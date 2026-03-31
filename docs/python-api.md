# Python API

::: cast_value_rs.cast_array

---

::: cast_value_rs._cast_value_rs.cast_array_into

---

## Rounding modes

| Mode | Behavior |
|---|---|
| `"nearest-even"` | Round to nearest, ties to even (IEEE 754 default) |
| `"towards-zero"` | Truncate towards zero |
| `"towards-positive"` | Round towards +infinity (ceiling) |
| `"towards-negative"` | Round towards -infinity (floor) |
| `"nearest-away"` | Round to nearest, ties away from zero |

## Out-of-range modes

| Mode | Behavior |
|---|---|
| `None` | Error on the first out-of-range value |
| `"clamp"` | Clamp to the target type's range |
| `"wrap"` | Modular arithmetic (wrapping) |

## Scalar map

Scalar map entries are applied before rounding and range checking. Each entry
maps a source value to a target value. NaN entries match any NaN value. First
match wins.

Scalar map values must have types compatible with the source and target dtypes.
For example, when casting `float64` to `int32`, the source value should be a
float and the target value should be an integer:

```python
# Correct: float source, int target
scalar_map_entries=[(float("nan"), 0), (float("inf"), 2147483647)]

# Error: float target value for int dtype
scalar_map_entries=[(float("nan"), 0.0)]  # TypeError
```

## Supported dtypes

`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`,
`float16`, `float32`, `float64`
