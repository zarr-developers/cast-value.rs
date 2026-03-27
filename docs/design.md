# Design

## Motivation

The `cast_value` codec converts array elements between numeric data types as
part of the Zarr v3 codec pipeline. The per-element pipeline is:
scalar map lookup, rounding, range checking, and type cast. In pure
Python/numpy each step is a separate array operation with its own allocation
and iteration pass. Rust fuses them into a single pass over contiguous memory.

## Crate structure

```
cast-value.rs/
  core/           # zarr-cast-value: pure Rust library, no Python dependency
    src/lib.rs
    Cargo.toml
  python/         # cast-value-rs: PyO3 bindings
    src/lib.rs
    cast_value_rs/
    Cargo.toml
    pyproject.toml
  Cargo.toml      # workspace
```

**`zarr-cast-value`** (library crate) -- pure Rust, independently testable
with `cargo test`. Can be used by any Rust consumer (e.g. zarrs) without
pulling in Python bindings. Contains conversion logic, type definitions,
trait definitions, and trait implementations for primitive numeric types.

**`cast-value-rs`** (cdylib crate) -- PyO3 + numpy bindings. Contains numpy
array I/O, dtype dispatch, argument parsing, and the `#[pymodule]` definition.

## Architecture: four conversion paths

Rather than a single generic function with runtime type-flag checks, the crate
uses four separate conversion functions -- one per source/target category.
Each function's trait bounds statically guarantee that only valid operations
are called:

| Path | Pipeline |
|---|---|
| float to int | scalar_map, reject NaN, round, range check/clamp/wrap, cast |
| int to int | scalar_map, range check/clamp/wrap, cast |
| float to float | scalar_map, NaN/Inf propagation, cast, rounding adjustment, overflow check |
| int to float | scalar_map, cast, rounding adjustment |

Each path exists at two levels:

- **Per-element** (`convert_float_to_int`, etc.) -- pure function, no
  allocations. Operates on native `Src` and `Dst` types.
- **Slice** (`convert_slice_float_to_int`, etc.) -- iterates a source
  `&[Src]` and writes into a caller-provided `&mut [Dst]`. Returns early
  on the first error.

Buffer management and dtype dispatch are the caller's responsibility.
The Python crate handles numpy array I/O; a zarrs integration would handle
`ArrayBytes::Fixed` reinterpretation.

## No intermediate types

Values are read in the source type, transformed in that type, and written in
the target type. There is no promotion to an intermediate like f64.

- Scalar map comparison uses `val == entry.src` in native `Src` type
- Rounding uses the source float's own precision
- Range checking compares against target bounds expressed in `Src` space
- The actual type conversion happens once, at the end

The only uses of `to_f64_lossy()` are: error messages (not on the hot path),
Python boundary (scalar map parsing, once at dispatch time), and rounding
adjustment inexactness detection (does not affect the output value).

## Trait design

```rust
trait CastNum: Copy + PartialEq + PartialOrd + Debug + ToPrimitive {}
trait CastInt: CastNum {}
trait CastFloat: CastNum + Float + One + Add + Sub {
    fn round_with_mode(self, mode: RoundingMode) -> Self;
    fn rem_euclid(self, rhs: Self) -> Self;
    fn next_up(self) -> Self;
    fn next_down(self) -> Self;
}
trait CastInto<Dst: CastNum>: CastNum {
    fn dst_min() -> Self;
    fn dst_max() -> Self;
    fn cast_into(self) -> Dst;
}
```

- **Separate int/float traits** eliminate `unreachable!()` branches
- **`CastInto<Dst>`** carries range bounds in source type space, so range
  checking needs no intermediate type
- **`MapEntry<Src, Dst>`** stores typed source and target fields; NaN matching
  is handled by checking `entry.src.is_nan()` at lookup time

## Dependencies

### Core crate

- `num-traits` -- `Float`, `ToPrimitive`, `One` traits
- `serde` -- serialization for `RoundingMode` and `OutOfRangeMode`
- `thiserror` -- error type derivation
- `half` (optional, `float16` feature) -- `f16` type

### Python crate

- `zarr-cast-value` with `float16` feature enabled
- `half` -- for `f16` in dispatch code
- `pyo3` -- Rust/Python bindings
- `numpy` with `half` feature -- numpy array interop including `f16`
