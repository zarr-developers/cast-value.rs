# cast-value.rs

Rust implementation of the [`cast_value` Zarr codec](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value) with Python bindings.

## What is this?

The `cast_value` codec converts array elements between numeric data types as part of the Zarr v3 codec pipeline. This project provides:

- **[`zarr-cast-value`](https://crates.io/crates/zarr-cast-value)** -- a pure Rust crate with no Python dependency, usable by any Rust consumer (e.g. [zarrs](https://github.com/LDeakin/zarrs))
- **[`cast-value-rs`](https://pypi.org/project/cast-value-rs/)** -- Python bindings via PyO3, exposing the Rust implementation to zarr-python

## Why Rust?

The hot path involves per-element operations -- scalar map lookups, rounding, and range checking -- applied to every value in the array. In pure Python/numpy this requires multiple passes over the data. Rust fuses all steps into a single pass over contiguous memory, avoiding intermediate allocations.

## Features

- All five rounding modes: `nearest-even`, `towards-zero`, `towards-positive`, `towards-negative`, `nearest-away`
- Out-of-range handling: `clamp`, `wrap`, or error
- Scalar map for special values (e.g. NaN to sentinel)
- float16 support via the `half` crate (feature-gated)

## Supported types

| Category | Types |
|---|---|
| Signed integers | `int8`, `int16`, `int32`, `int64` |
| Unsigned integers | `uint8`, `uint16`, `uint32`, `uint64` |
| Floats | `float16`, `float32`, `float64` |
