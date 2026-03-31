# Getting started

## Python

Install the package from PyPI:

```sh
pip install cast-value-rs
```

Basic usage:

```python
import numpy as np
from cast_value_rs import cast_array

# Convert float64 to uint8 with clamping
data = np.array([1.7, 2.3, 300.0, -5.0], dtype=np.float64)
result = cast_array(
    data,
    target_dtype="uint8",
    rounding_mode="nearest-even",
    out_of_range_mode="clamp",
)
print(result)  # [2, 2, 255, 0]
```

### Accepting numpy dtypes

The `target_dtype` parameter accepts strings, numpy dtype objects, or numpy
scalar types:

```python
cast_array(data, target_dtype="uint8", ...)
cast_array(data, target_dtype=np.dtype("uint8"), ...)
cast_array(data, target_dtype=np.uint8, ...)
```

### Using scalar maps

Scalar maps let you handle special values like NaN before conversion:

```python
from math import nan, inf

result = cast_array(
    np.array([1.0, nan, inf], dtype=np.float64),
    target_dtype="uint8",
    rounding_mode="nearest-even",
    out_of_range_mode="clamp",
    scalar_map_entries={nan: 0, inf: 255},
)
print(result)  # [1, 0, 255]
```

### Pre-allocated output

Use `cast_array_into` to write into an existing array:

```python
from cast_value_rs import cast_array_into

src = np.array([1.0, 2.0, 3.0], dtype=np.float64)
dst = np.zeros(3, dtype=np.uint8)
cast_array_into(
    src, dst,
    rounding_mode="nearest-even",
    out_of_range_mode="clamp",
)
print(dst)  # [1, 2, 3]
```

## Rust

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
zarr-cast-value = "0.2"
```

For float16 support:

```toml
[dependencies]
zarr-cast-value = { version = "0.2", features = ["float16"] }
```

Basic usage:

```rust
use zarr_cast_value::{
    convert_slice_float_to_int, FloatToIntConfig,
    RoundingMode, OutOfRangeMode,
};

let src = [1.7_f64, 2.3, 300.0, -5.0];
let mut dst = [0u8; 4];
let config = FloatToIntConfig {
    map_entries: vec![],
    rounding: RoundingMode::NearestEven,
    out_of_range: Some(OutOfRangeMode::Clamp),
};
convert_slice_float_to_int(&src, &mut dst, &config).unwrap();
assert_eq!(dst, [2, 2, 255, 0]);
```
