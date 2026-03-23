from collections.abc import Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt

RoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]

OutOfRangeMode = Literal["clamp", "wrap"]

DTypeName = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]

def cast_array(
    arr: npt.NDArray[np.generic],
    *,
    target_dtype: DTypeName,
    rounding_mode: RoundingMode,
    out_of_range_mode: OutOfRangeMode | None = None,
    scalar_map_entries: dict[float, float] | Iterable[tuple[float, float]] | None = None,
) -> npt.NDArray[np.generic]:
    """Cast a numpy array to a new dtype, allocating a new output array.

    Parameters
    ----------
    arr
        Input numpy array.
    target_dtype
        Target dtype name (e.g. ``"uint8"``, ``"float32"``).
    rounding_mode
        How to round values during conversion.
    out_of_range_mode
        How to handle values outside the target type's range.
        ``None`` means out-of-range values raise an error.
    scalar_map_entries
        Mapping of special source values to target values, applied before
        any other conversion logic. Accepts a dict or any iterable of
        ``(source, target)`` pairs (tuples, lists, etc.). Useful for
        mapping ``NaN`` or ``Inf`` to sentinel integer values.

    Returns
    -------
    npt.NDArray[np.generic]
        A new numpy array with the target dtype.

    Raises
    ------
    ValueError
        If a value cannot be converted (e.g. NaN to int without a
        scalar_map entry, or out-of-range without a mode set).
    TypeError
        If the source or target dtype is unsupported.
    """
    ...

def cast_array_into(
    arr: npt.NDArray[np.generic],
    out: npt.NDArray[np.generic],
    *,
    rounding_mode: RoundingMode,
    out_of_range_mode: OutOfRangeMode | None = None,
    scalar_map_entries: dict[float, float] | Iterable[tuple[float, float]] | None = None,
) -> None:
    """Cast a numpy array into a pre-allocated output array.

    Parameters
    ----------
    arr
        Input numpy array.
    out
        Pre-allocated output numpy array. Must have the same shape as
        ``arr``.
    rounding_mode
        How to round values during conversion.
    out_of_range_mode
        How to handle values outside the target type's range.
        ``None`` means out-of-range values raise an error.
    scalar_map_entries
        Mapping of special source values to target values. Accepts a dict
        or any iterable of ``(source, target)`` pairs.

    Raises
    ------
    ValueError
        If shapes don't match, or a value cannot be converted.
    TypeError
        If the source or target dtype is unsupported.
    """
    ...
