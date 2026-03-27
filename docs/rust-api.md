# Rust API

The full Rust API documentation is hosted on docs.rs:

**[`zarr-cast-value` on docs.rs](https://docs.rs/zarr-cast-value)**

## Overview

The crate exposes four slice conversion functions (one per conversion category)
and four per-element conversion functions:

### Slice functions

| Function | Source | Target |
|---|---|---|
| `convert_slice_float_to_int` | `CastFloat` | `CastInt` |
| `convert_slice_int_to_int` | `CastInt` | `CastInt` |
| `convert_slice_float_to_float` | `CastFloat` | `CastFloat` |
| `convert_slice_int_to_float` | `CastInt` | `CastFloat` |

Each takes a source slice, a mutable destination slice, and a configuration
struct. SIMD acceleration is used automatically when the configuration allows
it.

### Per-element functions

| Function | Source | Target |
|---|---|---|
| `convert_float_to_int` | `CastFloat` | `CastInt` |
| `convert_int_to_int` | `CastInt` | `CastInt` |
| `convert_float_to_float` | `CastFloat` | `CastFloat` |
| `convert_int_to_float` | `CastInt` | `CastFloat` |

### Configuration structs

Each conversion path has its own config struct:

- `FloatToIntConfig` -- rounding mode, out-of-range mode, scalar map entries
- `IntToIntConfig` -- out-of-range mode, scalar map entries
- `FloatToFloatConfig` -- rounding mode, out-of-range mode, scalar map entries
- `IntToFloatConfig` -- rounding mode, scalar map entries

### Types and traits

| Type/Trait | Description |
|---|---|
| `RoundingMode` | Enum: `NearestEven`, `TowardsZero`, `TowardsPositive`, `TowardsNegative`, `NearestAway` |
| `OutOfRangeMode` | Enum: `Clamp`, `Wrap` |
| `MapEntry<Src, Dst>` | A scalar map entry with typed source and target fields |
| `CastError` | Error type for conversion failures |
| `CastNum` | Base trait for all numeric types |
| `CastInt` | Marker trait for integer types |
| `CastFloat` | Trait for float types with rounding and IEEE 754 operations |
| `CastInto<Dst>` | Conversion trait with range bounds and cast operation |

## Feature flags

| Feature | Description | Dependency |
|---|---|---|
| `float16` | Adds `f16` type support | `half` |
