# zarr-cast-value: Design Document

## Motivation

The `cast_value` codec converts array elements between numeric data types as part
of the zarr v3 codec pipeline. The hot path involves per-element operations:
scalar map lookups, rounding, and range checking — all applied to every value in
the array. In pure Python/numpy this becomes a bottleneck because:

1. **Multiple passes** — scalar map, rounding, and range check are separate
   numpy operations, each allocating temporaries and iterating the full array.
2. **No fused loops** — numpy cannot fuse these steps into a single pass.
3. **GIL contention** — all work holds the GIL.

Rust (via PyO3 + the `numpy` crate) lets us fuse all steps into a single pass
over contiguous memory, avoid intermediate allocations, and release the GIL
during computation.

Note: `scale_offset` stays in pure Python/numpy — it's a single multiply + add,
which numpy handles efficiently in one fused operation.

---

## Crate Structure

The project is split into two crates:

```
zarr-cast-value/
  core/           # zarr-cast-value-core: pure Rust library, no Python dependency
    src/lib.rs
    Cargo.toml
  python/         # zarr-cast-value: PyO3 bindings, depends on core
    src/lib.rs
    Cargo.toml
    pyproject.toml
  Cargo.toml      # workspace
```

### `zarr-cast-value-core` (library crate)

Pure Rust. No PyO3, no numpy crate, no Python dependency. Contains:

- Layers 1 and 2 (see below): four per-element conversion functions and
  four slice conversion functions
- Type definitions: `RoundingMode`, `OutOfRangeMode`, `MapEntry`, `CastError`
- Trait definitions: `CastNum`, `CastInt`, `CastFloat`, `CastInto`
- Trait implementations for built-in numeric types + external types behind
  feature flags

This crate is independently testable with `cargo test` and could be reused by
other Rust consumers (e.g. zarrs) without pulling in Python bindings.

### `zarr-cast-value` (cdylib crate)

PyO3 + numpy bindings. Depends on `zarr-cast-value-core`. Contains:

- A numpy-specific layer 3: numpy array I/O, dtype dispatch, `#[pyfunction]`
- Argument parsing (strings → Rust enums)
- The `#[pymodule]` definition

---

## Architecture: Four Layers

### Layer 1: Per-element transform (four conversion functions)

*Crate: core*

Takes a source value in its native type and produces a target value in the
target type. There is no intermediate representation — each function operates
directly on `Src` and `Dst`.

Rather than a single generic `convert_element` that uses runtime `IS_FLOAT` /
`IS_INTEGER` flags to decide which steps apply, we use **four separate
functions** — one per conversion category. This eliminates runtime type-flag
checks and `unreachable!` default implementations: each function's trait bounds
statically guarantee that only valid operations are called.

```
fn convert_float_to_int<Src: CastFloat, Dst: CastInt>(val, map, rounding, range) -> Result<Dst>:
    1. scalar_map lookup (first match wins)
    2. reject NaN
    3. round
    4. range check + clamp/wrap (wrap uses rem_euclid on the source float)
    5. cast Src → Dst

fn convert_int_to_int<Src: CastInt, Dst: CastInt>(val, map, range) -> Result<Dst>:
    1. scalar_map lookup (first match wins)
    2. range check + clamp/wrap (wrap uses bit-truncating `as` cast)
    3. cast Src → Dst

fn convert_float_to_float<Src: CastFloat, Dst: CastFloat>(val, map, rounding, range) -> Result<Dst>:
    1. scalar_map lookup (first match wins)
    2. NaN/Inf propagation (IEEE 754; signed zero preserved)
    3. cast Src → Dst (nearest-even by default)
    4. rounding adjustment (if mode ≠ nearest-even and cast was inexact)
    5. range check: if finite source overflowed to ±Inf → clamp or error

fn convert_int_to_float<Src: CastInt, Dst: CastFloat>(val, map, rounding) -> Result<Dst>:
    1. scalar_map lookup (first match wins)
    2. cast Src → Dst (nearest-even by default)
    3. rounding adjustment (if mode ≠ nearest-even and cast was inexact)
```

Each function is a pure function with no allocations. The trait bounds ensure:
- Round-to-integer (`round`) is only callable on `CastFloat` sources.
- NaN/Inf checks (`is_nan`, `is_infinite`) are only callable on `CastFloat` sources.
- Float-only arithmetic (`rem_euclid`, `sub`, `add`, `one`) is only available
  on `CastFloat` types — no `unreachable!` fallback needed.
- Precision-loss rounding applies to float→int, float→float, and int→float
  paths (any cast where the target type cannot exactly represent the input).
- Range checking applies when the target is `CastInt` or when float→float
  narrowing can overflow (e.g. f64→f32 beyond f32::MAX).

Steps operate on the source type. Rounding uses the source float's own
precision. Range checking compares against the target type's bounds. The actual
type conversion happens once, at the end, after all validation has passed.

For example, `float64 → uint8`: the f64 value gets scalar-mapped, rounded, and
range-checked — all as f64 — then converted to u8. For `int32 → uint8`: the i32
value gets range-checked as i32, then converted to u8. No intermediate buffer.

### Layer 2: Slice transform (`convert_slice_*`)

*Crate: core*

Four slice-level functions mirror the four element-level functions:
`convert_slice_float_to_int`, `convert_slice_int_to_int`,
`convert_slice_float_to_float`, `convert_slice_int_to_float`.

Each iterates over a source slice `&[Src]` and writes converted values into a
caller-provided output slice `&mut [Dst]`. Returns early on the first error
(e.g. out-of-range value with no clamp/wrap, or unconvertible NaN/Inf).

```
fn convert_slice_float_to_int<Src: CastFloat, Dst: CastInt>(
    src: &[Src], dst: &mut [Dst], map, rounding, range,
) -> Result<()>:
    for (in_val, out_slot) in src.iter().zip(dst.iter_mut()):
        *out_slot = convert_float_to_int(in_val, ...)?
    Ok(())
```

This layer owns the iteration but not the allocation. The caller provides both
buffers, so this function is agnostic to how memory is managed (stack, heap,
numpy-owned, etc.). Early termination means we don't waste time converting
remaining elements after the first failure.

When no scalar map entries are present, each function uses a tighter inner loop
that skips the map lookup entirely, allowing better optimization.

### Layer 3: Buffer management + dtype dispatch

*Crate: each consumer implements their own*

This layer is not in the core crate — it's specific to each consumer's buffer
format. It handles:

- Reading input data as a typed `&[Src]` slice
- Allocating or accepting an output `&mut [Dst]` slice
- Dispatching on `(src_dtype, tgt_dtype)` to monomorphize `convert_slice`
- Constructing `MapEntry<Src, Dst>` from the caller's pre-parsed scalars

For the **Python crate**, this means numpy array I/O:

```
fn cast_array_impl(input, src_dtype, tgt_dtype, ...):
    src_slice: &[Src] = view input numpy array as typed slice
    output: PyArrayDyn<Dst> = allocate(same shape)
    dst_slice: &mut [Dst] = view output numpy array as typed slice
    convert_slice(src_slice, dst_slice, ...)?
    return output
```

For **zarrs**, this would mean `ArrayBytes::Fixed` reinterpretation:

```
fn cast_bytes(bytes: &[u8], output: &mut [u8], src_dtype, tgt_dtype, ...):
    src_slice: &[Src] = reinterpret bytes as typed slice
    dst_slice: &mut [Dst] = reinterpret output as typed slice
    convert_slice(src_slice, dst_slice, ...)?
```

Dispatch on `(src_dtype, tgt_dtype)` produces `N × N` match arms, but each arm
is a one-liner calling `convert_slice<Src, Dst>` — the logic lives in the core
crate.

### Layer 4: Entry point

*Crate: each consumer implements their own*

For the **Python crate**: the `#[pyfunction]` that Python calls. Parses string
arguments into Rust enums, extracts the source dtype from the numpy array, and
delegates to layer 3.

For **zarrs**: the `ArrayToArrayCodecTraits::encode`/`decode` methods.

---

## No intermediate types

A core design principle: **avoid promoting values to an intermediate type (e.g.
f64) during conversion.** Each element should be read in its source type,
transformed in that type, and written in the target type. Intermediate
promotion risks precision loss, adds unnecessary computation, and makes it
harder to reason about correctness.

### Where this applies

- **Scalar map lookup**: Comparison uses `val == entry.src` in native `Src`
  type. Map entries are stored as `MapEntry<Src, Dst>` with typed fields.
- **NaN/Inf checks**: `val.is_nan()` and `val.is_infinite()` on the source type.
- **Rounding (float→int)**: Uses the source float's own precision. For f32
  sources, rounding happens in f32 — no promotion to f64.
- **Rounding adjustment (float→float, int→float)**: After the `as` cast
  (nearest-even), inexact results are adjusted using `next_up()`/`next_down()`
  on the target float type. The inexactness check uses `to_f64_lossy()` — this
  is acceptable because it only affects whether an adjustment is needed, not the
  final value.
- **Range checking**: Compares source values against target bounds expressed in
  `Src` space via `CastInto<Dst>::dst_min()` / `dst_max()`.
- **Clamping**: Clamps in source type, then casts once.
- **Int→int wrap**: Uses Rust's `as` cast (bit truncation = modular arithmetic).
  No intermediate type needed.
- **Float→int wrap**: Uses `rem_euclid` on the source float type directly,
  available because the function's `CastFloat` bound guarantees this operation
  exists. No promotion to f64.
- **Final cast**: A single `Src → Dst` conversion at the end of the pipeline.

### Per-path summary

- **int→int** (e.g. i32→u8): range check on the i32 value, then cast to u8.
  No precision loss possible. No rounding.
- **float→int** (e.g. f64→u8): scalar map, round to integer, range check — all
  on the f64 value — then cast to u8.
- **int→float** (e.g. i64→f32): scalar map (if any) on the i64 value, cast to
  f32, then rounding adjustment if the cast was inexact and mode ≠ nearest-even.
- **float→float** (e.g. f64→f32): scalar map (if any), NaN/Inf/zero
  propagation, cast, rounding adjustment if inexact, overflow check.

### Acceptable exceptions

Three places where `to_f64_lossy()` is used:

1. **Error reporting**: `CastError::OutOfRange` and `CastError::NanOrInf` carry
   f64 values for human-readable error messages. This only runs on the error
   path, never during successful conversion.
2. **Python boundary**: `parse_map_entries` receives f64 pairs from Python and
   converts them to typed `MapEntry<Src, Dst>` via `FromF64`. This runs once
   at dispatch time, not per element.
3. **Rounding adjustment (float→float, int→float)**: After the `as` cast, we
   compare `result.to_f64_lossy()` against `val.to_f64_lossy()` to detect
   whether rounding occurred. This is in the per-element path but only executes
   when the rounding mode is not `nearest-even` (the common default). It does
   not affect the output value — the adjustment uses `next_up()`/`next_down()`
   on the target type, which are exact operations.

---

## Python API

Two functions are exposed to Python: one that allocates the output, and one
that writes into a caller-provided buffer.

### `cast_array` — allocating variant

```python
def cast_array(
    arr: numpy.ndarray,
    target_dtype: str,
    rounding_mode: str,
    out_of_range_mode: str | None = None,
    scalar_map_entries: list[list[float]] | None = None,
) -> numpy.ndarray: ...
```

Allocates a new output array with the target dtype and same shape as `arr`.

### `cast_array_into` — pre-allocated variant

```python
def cast_array_into(
    arr: numpy.ndarray,
    out: numpy.ndarray,
    rounding_mode: str,
    out_of_range_mode: str | None = None,
    scalar_map_entries: list[list[float]] | None = None,
) -> None: ...
```

Writes converted values into `out`. The target dtype is inferred from
`out.dtype`. Raises `ValueError` if `out.shape != arr.shape`. Raises
`TypeError` if `out` is not contiguous or not writeable.

This variant avoids allocation when the caller already has a buffer (e.g.
reusing a pre-allocated chunk buffer in the codec pipeline).

### Common parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `arr` | `numpy.ndarray` | Input array (any supported numeric dtype) |
| `target_dtype` | `str` | Numpy dtype name (allocating variant only) |
| `out` | `numpy.ndarray` | Pre-allocated output array (into variant only) |
| `rounding_mode` | `str` | `"nearest-even"`, `"towards-zero"`, `"towards-positive"`, `"towards-negative"`, `"nearest-away"` |
| `out_of_range_mode` | `str \| None` | `"clamp"`, `"wrap"`, or `None` (error on out-of-range) |
| `scalar_map_entries` | `list[[float, float]] \| None` | Pre-parsed `[source, target]` pairs as f64 values |

### Design decisions

- **`scalar_map_entries` are pre-parsed as f64 pairs**: The Python side uses
  `ZDType.from_json_scalar` to parse JSON scalars into numpy scalars, then
  converts them to f64 pairs for the Rust boundary. The python crate's layer 3
  converts these f64 pairs into typed `MapEntry<Src, Dst>` values for each
  monomorphized `convert_slice` call. For integer map entries, this means the
  f64 is cast to the concrete `Src`/`Dst` type at dispatch time — safe because
  `CastValue.validate()` rejects int types whose range exceeds f64 precision.
- **`rounding_mode` is always required**: The Python caller passes the codec's
  configured mode (defaulting to `"nearest-even"`). No default in Rust.
- **Source dtype is inferred**: Read from `arr.dtype.name` inside Rust, so the
  caller doesn't need to pass it separately.
- **Both variants delegate to the same layer 2** (`convert_slice`). The only
  difference is whether layer 3 allocates the output or receives it.

---

## Supported Types

The core crate must support all integer and floating-point data types defined in
the zarr v3 core spec and registered extensions. Complex, boolean, and string
types are rejected by `CastValue.validate()` before data reaches this code.

### Core spec types (always available)

| Category | Types | Rust types |
|----------|-------|------------|
| Signed integers | `int8`, `int16`, `int32`, `int64` | `i8`, `i16`, `i32`, `i64` |
| Unsigned integers | `uint8`, `uint16`, `uint32`, `uint64` | `u8`, `u16`, `u32`, `u64` |
| Floats | `float16`, `float32`, `float64` | `half::f16`, `f32`, `f64` |

### Extension types (behind feature flags)

| Category | Types | Rust crate | Feature flag |
|----------|-------|------------|-------------|
| Specialized float | `bfloat16` | `half::bf16` | `bfloat16` |
| Low-bit integers | `int2`, `int4`, `uint2`, `uint4` | TBD | `low-bit-int` |
| 8-bit floats | `float8_e4m3`, `float8_e5m2`, etc. | TBD | `float8` |
| 4/6-bit floats | `float4_e2m1fn`, `float6_e2m3fn`, etc. | TBD | `low-bit-float` |

### Trait design for type extensibility

The conversion functions are generic over `Src` and `Dst`. Rather than a single
`CastType` trait with boolean `IS_FLOAT`/`IS_INTEGER` flags and `unreachable!`
default implementations for float-only operations, we use **separate traits for
integer and float types**. This lets the type system enforce which operations
are valid, eliminating runtime type checks and dead code.

```rust
/// Common base for all numeric types that participate in cast_value conversions.
/// Provides only the operations shared by both integers and floats.
trait CastNum: Copy + PartialEq + PartialOrd + Debug {
    /// Convert to f64 for error reporting only (not in the conversion hot path).
    fn to_f64_lossy(self) -> f64;
}

/// An integer numeric type. Implemented by i8..i64, u8..u64.
/// No NaN, no Inf, no rounding needed.
trait CastInt: CastNum {}

/// A floating-point numeric type. Implemented by f32, f64.
/// Carries all float-specific operations that the old CastType put behind
/// `unreachable!` defaults.
trait CastFloat: CastNum {
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
    fn round(self, mode: RoundingMode) -> Self;
    fn rem_euclid(self, rhs: Self) -> Self;
    fn one() -> Self;
    fn next_up(self) -> Self;    // IEEE 754 nextUp
    fn next_down(self) -> Self;  // IEEE 754 nextDown
    fn abs(self) -> Self;
    // Standard arithmetic via core::ops::Add/Sub
}

/// Conversion from Src to Dst. Implemented per (Src, Dst) pair.
trait CastInto<Dst>: CastNum {
    fn dst_min() -> Self;
    fn dst_max() -> Self;
    fn cast_into(self) -> Dst;
}
```

Key design points:

- **Separate int/float traits eliminate `unreachable!`**: Float-only operations
  (`is_nan`, `is_infinite`, `round`, `rem_euclid`, `one`) live exclusively on
  `CastFloat`. Integer types don't implement them at all — calling them on an
  integer is a compile error, not a runtime panic. Similarly, the four conversion
  functions use trait bounds to select the right path at compile time instead of
  runtime `IS_FLOAT`/`IS_INTEGER` checks.

- **No `to_f64`/`from_f64` on the hot path**: Scalar map comparison uses
  `PartialEq` on `Src` directly. Map entries are typed as `MapEntry<Src, Dst>`
  — the source field is `Src`, not f64. This avoids precision loss for
  int64/uint64 values > 2^53.

- **`CastInto<Dst>` carries the range bounds in `Src` space**: `dst_min()` and
  `dst_max()` return `Src` values, so range checking compares source values
  against target bounds without converting to an intermediate type. For example,
  `CastInto<u8> for i32` has `dst_min() -> i32 = 0` and `dst_max() -> i32 = 255`.

- **`CastInto` is implemented per `(Src, Dst)` pair**: This is the one place
  where the `N × N` combinatorics show up in the core crate. But each impl is
  trivial — just an `as` cast after the range check has already passed.

- **Scalar map `MapEntry<Src, Dst>`**: The source field is `Src` (not f64), so
  comparison is `val == entry.src` in native type space. The target field is
  `Dst`, so substitution is direct assignment with no conversion. The caller
  (layer 3) is responsible for constructing typed map entries from whatever
  format it receives (f64 pairs from Python, typed bytes from zarrs).

These traits are implemented for all built-in Rust numeric types in the core
crate. External types (`half::f16`, `half::bf16`, float8 types) get
implementations behind feature flags, so the core crate only pulls in those
dependencies when needed.

### Required external crate dependencies

```toml
# zarr-cast-value-core/Cargo.toml
[features]
default = []
float16 = ["dep:half"]       # f16 — part of core spec, but needs external crate
bfloat16 = ["dep:half"]      # bf16 — extension type
float8 = ["dep:float8"]      # float8 variants — extension types
low-bit-int = []              # int2/4, uint2/4 — may be representable as u8

[dependencies]
half = { version = "2", optional = true }
float8 = { version = "...", optional = true }  # TBD — no mature crate yet
```

Note: `float16` is in the zarr v3 core spec but requires the `half` crate in
Rust since there's no native `f16` type. This should probably be a default
feature.

---

## Rounding Modes

Applied when casting to a data type with insufficient numerical precision. Per
the zarr cast_value spec, this includes:

- **float→int**: rounding a float to the nearest integer value
- **float→float**: narrowing precision (e.g. f64→f32) when the value is not
  exactly representable in the target type
- **int→float**: large integers that exceed the mantissa precision of the
  target float (e.g. int64→float32 for values > 2^24)

Int→int casts never require rounding (integers are always exact).

For float→int, rounding operates on the source float value directly using
`.round()`, `.trunc()`, `.ceil()`, `.floor()`, or `.copysign()`. For
float→float and int→float, Rust's `as` cast always uses nearest-even; when a
different mode is requested and the cast is inexact, the result is adjusted
using `next_up()`/`next_down()` on the target float type.

| Mode | Behavior |
|------|----------|
| `nearest-even` | Round to nearest, ties to even (IEEE 754 `roundTiesToEven`) |
| `towards-zero` | Round towards zero |
| `towards-positive` | Round towards +∞ (ceiling) |
| `towards-negative` | Round towards −∞ (floor) |
| `nearest-away` | Round to nearest, ties away from zero |

For **float→int**, these are implemented directly on the source float:
`.round_ties_even()`, `.trunc()`, `.ceil()`, `.floor()`, or
`.copysign((.abs() + 0.5).floor(), self)`.

For **float→float** and **int→float**, Rust's `as` cast always uses
nearest-even. When a different mode is requested and the cast is inexact,
the result is adjusted by one ULP using `next_up()`/`next_down()` on the
target float type. For `nearest-away`, ties are detected by checking if the
original value falls exactly at the midpoint between the result and the
adjacent representable value.

These methods are available on `f32` and `f64` natively. For `f16` and `bf16`
(which lack native arithmetic), the value is promoted to `f32` for the rounding
operation — this is lossless since both fit within f32's precision.

---

## Out-of-Range Handling

Applied when a value exceeds the representable range of the target type.

### Integer targets (float→int and int→int)

| Mode | Behavior |
|------|----------|
| `None` | Return error on first out-of-range value |
| `"clamp"` | `val.clamp(dst_min, dst_max)` then cast |
| `"wrap"` | modular arithmetic into target range, then cast |

Range bounds (`dst_min`, `dst_max`) are provided by `CastInto<Dst>::dst_min()`
and `CastInto<Dst>::dst_max()`, expressed in the source type. This means no
intermediate representation is needed for the comparison.

### Float targets (float→float only)

When narrowing from a wider float to a narrower one (e.g. f64→f32), a finite
source value may overflow to ±Infinity. Per the spec:

| Mode | Behavior |
|------|----------|
| `None` | Return error on overflow (finite→Inf) |
| `"clamp"` | Map to ±Infinity (which is what the `as` cast already produces) |
| `"wrap"` | **Not permitted** for float targets (errors) |

Source ±Infinity propagates to target ±Infinity unconditionally (IEEE 754), and
does not trigger the out-of-range check. Only finite-to-infinite overflow is
considered out-of-range.

Int→float casts never overflow (all integer ranges fit within f32/f64's finite
range), so out-of-range does not apply.

---

## Scalar Map

### Runtime behavior

Applied before rounding and range checking. Each entry is a `(source, target)`
pair stored as `MapEntry<Src, Dst>`. For each element:

1. If `entry.src.is_nan()` → match any NaN value (float sources only)
2. Otherwise → exact equality: `val == entry.src` (in native `Src` type)

On match, the element is replaced with `entry.tgt` (native `Dst` type) and
the conversion function returns immediately — no rounding or range check needed
since the target value was explicitly chosen.

First matching entry wins; unmatched values continue through the pipeline.

### JSON representation (zarr metadata)

In zarr metadata, scalar map entries are JSON values interpreted according to
the source and target data types:

```json
{
  "scalar_map": {
    "encode": [["NaN", 0], ["+Infinity", 255], [-1, 0]],
    "decode": [[0, "NaN"], [255, "+Infinity"]]
  }
}
```

Each value can be:
- A JSON number: `0`, `255`, `-1`, `3.14`
- A JSON string for special floats: `"NaN"`, `"+Infinity"`, `"-Infinity"`
- A hex-encoded NaN payload string: `"0x7fc00001"` (preserves NaN bit pattern)

The source side of each pair is parsed using the source data type, and the
target side using the target data type. This is necessary because:
- An integer `0` in a uint8 context is different from `0.0` in a float64 context
- A hex NaN string must be parsed with the correct float width to preserve the
  payload bits

### Parsing is the caller's responsibility

JSON scalar parsing is **not** in the core crate. Both zarrs and zarr-python
already implement this:

- **zarrs**: `DataTypeTraits::fill_value` converts JSON scalars to typed bytes.
  The zarrs codec adapter would use this to parse scalar map entries into typed
  values before passing them to `convert_slice`.
- **zarr-python**: `ZDType.from_json_scalar` converts JSON scalars to numpy
  scalars. The Python bindings convert these to the appropriate Rust types.

Reimplementing JSON scalar parsing in the core crate would duplicate existing
functionality in both consumers. Instead, the core crate accepts pre-parsed,
typed `MapEntry<Src, Dst>` values:

```rust
/// A parsed scalar map entry, typed on Src and Dst.
struct MapEntry<Src, Dst> {
    src: Src,
    tgt: Dst,
}
```

NaN matching is handled implicitly: `apply_scalar_map_float` calls
`entry.src.is_nan()` to detect NaN entries at lookup time, so no separate
boolean flag is needed.

Each caller is responsible for constructing `MapEntry` values using their own
JSON parsing infrastructure. For the Python boundary specifically, this means
the python crate's layer 3 receives f64 pairs and converts them to typed
`MapEntry<Src, Dst>` inside each `(src_dtype, tgt_dtype)` dispatch arm.

---

## Integration with zarr-python

The `CastValue` codec in `src/zarr/codecs/cast_value.py` calls the Rust function
as a drop-in replacement for `_cast_array_impl`:

```python
from zarr_cast_value import cast_array as _rust_cast_array

# In _cast_array:
result = _rust_cast_array(
    arr,
    target_dtype=str(tgt_np_dtype),
    rounding_mode=rounding,
    out_of_range_mode=out_of_range,
    scalar_map_entries=[[float(s), float(t)] for s, t in parsed_entries],
)
```

The Python side remains responsible for:

- Metadata validation (precision checks, dtype compatibility)
- Parsing `scalar_map` JSON via `ZDType.from_json_scalar` (existing code)
- Converting parsed scalars to f64 pairs for the Rust boundary
- Fill value casting (reuses the same `_cast_array` path)
- Codec lifecycle (`from_dict`, `to_dict`, `resolve_metadata`)

---

## Build

Workspace build with [maturin](https://www.maturin.rs/):

```bash
cd zarr-cast-value
uv run --with maturin maturin develop
```

### Core crate dependencies

No external dependencies. The `CastFloat` and `CastInt` traits define exactly
the operations needed for conversion, using only `std` types and operations.
The `num-traits` crate was evaluated but is not needed — our custom traits
provide a tighter interface than `num_traits::Float` / `num_traits::PrimInt`,
and avoid pulling in unused functionality.

### Python crate dependencies

- `zarr-cast-value-core` (path dependency)
- `pyo3 = "0.23"` — Rust↔Python bindings
- `numpy = "0.23"` — numpy array interop for PyO3

---

## Integration with zarrs

The core crate has no Python dependency, so zarrs can use it directly.

### How zarrs would use the core crate

zarrs would depend on `zarr-cast-value-core` and implement the codec adapter
on their side. The core crate exposes the four per-element conversion functions,
the four slice conversion functions, and the supporting types and traits
(`RoundingMode`, `OutOfRangeMode`, `MapEntry`, `CastError`, `CastNum`,
`CastInt`, `CastFloat`, `CastInto`, and the four per-path config structs).

A zarrs-side `CastValue` struct would implement `ArrayToArrayCodecTraits`. Its
`encode`/`decode` methods would:

1. Determine `src_dtype` and `tgt_dtype` from the array's `data_type` and the codec's `data_type` field
2. Parse scalar map entries from JSON via `DataTypeTraits::fill_value` into
   typed `MapEntry<Src, Dst>` values
3. Reinterpret `ArrayBytes::Fixed` as a typed `&[Src]` slice
4. Allocate an output `Vec<u8>` for the target type, view as `&mut [Dst]`
5. Call the appropriate `zarr_cast_value_core::convert_slice_*` function
6. Return the output as `ArrayBytes::Fixed`

Steps 1–4 and 6 are zarrs-specific (byte reinterpretation, `DataType` dispatch,
JSON scalar parsing). Step 5 is the core crate — the same `convert_slice_*` functions that
the Python bindings use.

This approach keeps the core crate minimal and avoids version coupling between
our crate and zarrs' trait definitions. zarrs owns the
`ArrayToArrayCodecTraits` implementation alongside their other codecs.

---

## Future Considerations

- **Extended type support**: The current implementation supports 10 types
  (i8/i16/i32/i64/u8/u16/u32/u64/f32/f64). A future phase should add support
  for the full set of zarr numeric types:
  - **Sub-byte integers**: `int2`, `int4` (stored as 1-byte unpacked by ml_dtypes)
  - **IEEE 754 float16**: `float16` (Rust: `half::f16`)
  - **bfloat16**: `bfloat16` (Rust: `half::bf16`)
  - **8-bit floats**: `float8_e3m4`, `float8_e4m3`, `float8_e4m3b11fnuz`,
    `float8_e4m3fnuz`, `float8_e5m2`, `float8_e5m2fnuz`, `float8_e8m0fnu`
  - **Sub-byte floats**: `float4_e2m1fn`, `float6_e2m3fn`, `float6_e3m2fn`

  Key challenges:
  - The Python binding's `(kind, itemsize)` dispatch can't distinguish types that
    share the same numpy storage (e.g. float8 variants all report `('f', 1)`).
    Dispatch must use dtype name strings instead.
  - Rust crate coverage is uneven: `half` covers f16/bf16, `float8` covers
    F8E4M3/F8E5M2, `float4` covers F4E2M1. Other float8/float6 variants and
    int2/int4 lack mature Rust crates and may need custom bit manipulation.
  - N×N dispatch grows from 100 to 256 match arms (16×16 types). Consider
    code generation or a widen-at-boundary strategy to manage this.
  - The feature flag structure in the extension types table above was designed
    for this expansion.

- **Parallelism**: `convert_slice` over `&mut [Dst]` is amenable to rayon-based
  chunk parallelism for large arrays. The core crate could expose a
  `convert_slice_par` variant.
- **SIMD**: Rounding and clamping could benefit from explicit SIMD intrinsics,
  though LLVM auto-vectorization may already handle simple cases.
