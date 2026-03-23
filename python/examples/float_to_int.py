"""
An example that converts an array of floats bounded between [0, 1000] with NaN values to uint16.
To make the conversion lossless, an offset of 1 is added to the floats before converting, and the
scalar_map parameter maps NaN to 0. On decoding, the scalar_map parameter maps 0 back to NaN, and
the offset of 1 is subtracted from the decoded integers to get back the original floats.
"""

import numpy as np
from math import nan
from zarr_cast_value import cast_array

# Original float64 data with NaN representing missing values.
# All finite values are integers in [0, 1000].
data = np.array([0.0, 1.0, 500.0, nan, 999.0, 1000.0, nan, 42.0], dtype=np.float64)
print(f"original float64: {data}")

# --- Encode: float64 -> uint16 ---
# Add offset of 1 so that the range [0, 1000] maps to [1, 1001],
# reserving 0 as the NaN sentinel in the integer domain.
shifted = data + 1.0  # NaN + 1.0 stays NaN
print(f"shifted (NaN + 1 = NaN): {shifted}")

# scalar_map: NaN in the source maps to 0 in the target.
# Rounding mode is towards-zero (all values are already integers, so
# rounding mode doesn't matter here, but we set it explicitly).
encoded = cast_array(
    shifted,
    target_dtype="uint16",
    rounding_mode="towards-zero",
    out_of_range_mode="clamp",
    scalar_map_entries=[(nan, 0)],
)
print(f"encoded uint16:   {encoded}")

# --- Decode: uint16 -> float64 ---
# scalar_map: 0 in the source maps to NaN in the target.
decoded = cast_array(
    encoded,
    target_dtype="float64",
    rounding_mode="nearest-even",
    scalar_map_entries=[(0, nan)],
)

# Subtract the offset to recover the original values.
recovered = decoded - 1.0  # NaN - 1.0 stays NaN
print(f"recovered float64: {recovered}")

# --- Verify round-trip ---
# np.testing.assert_array_equal handles NaN==NaN correctly.
np.testing.assert_array_equal(data, recovered)
print("\nround-trip verified: original == recovered")