"""Tests for scalar_map_entries parsing and behavior."""

from collections.abc import Mapping
from math import inf, nan

import numpy as np
import pytest

from cast_value_rs import cast_array

from .conftest import Expect, ExpectFail, nan_eq


def _cast_f64_to_u8(scalar_map_entries):
    """Helper: cast a fixed float64 array to uint8 with the given scalar map."""
    return cast_array(
        np.array([1.0, nan, inf], dtype=np.float64),
        target_dtype="uint8",
        rounding_mode="nearest-even",
        out_of_range_mode="clamp",
        scalar_map_entries=scalar_map_entries,
    )

class ScalarMapping(Mapping[object, object]):
    """A custom mapping type for testing dict-like input acceptance."""

    def __init__(self, data: Mapping[object, object]):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

# ---------------------------------------------------------------------------
# Input format acceptance
# ---------------------------------------------------------------------------

FORMAT_CASES = [
    Expect(
        input={nan: 0, inf: 255},
        expected=np.array([1, 0, 255], dtype=np.uint8),
        id="dict",
    ),
    Expect(
        input={nan: np.uint8(0), inf: np.uint8(255)},
        expected=np.array([1, 0, 255], dtype=np.uint8),
        id="dict-of-numpy-scalars",
    ),
    Expect(
        input=ScalarMapping({nan: 0, inf: 255}),
        expected=np.array([1, 0, 255], dtype=np.uint8),
        id="custom-mapping",
    ),
    Expect(
        input=[(nan, 0), (inf, 255)],
        expected=np.array([1, 0, 255], dtype=np.uint8),
        id="list-of-tuples",
    ),
    Expect(
        input=[[nan, 0], [inf, 255]],
        expected=np.array([1, 0, 255], dtype=np.uint8),
        id="list-of-lists",
    ),
    Expect(
        input=((nan, 0), (inf, 255)),
        expected=np.array([1, 0, 255], dtype=np.uint8),
        id="tuple-of-tuples",
    ),
]


@pytest.mark.parametrize("case", FORMAT_CASES, ids=[c.id for c in FORMAT_CASES])
def test_format(case: Expect):
    result = _cast_f64_to_u8(case.input)
    case.check(result)


def test_generator():
    def gen():
        yield (nan, 0)
        yield (inf, 255)

    result = _cast_f64_to_u8(gen())
    expected = np.array([1, 0, 255], dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_none():
    arr = np.array([1.0], dtype=np.float64)
    result = cast_array(
        arr,
        target_dtype="uint8",
        rounding_mode="nearest-even",
        scalar_map_entries=None,
    )
    assert np.array_equal(result, np.array([1], dtype=np.uint8))


def test_empty():
    arr = np.array([1.0], dtype=np.float64)
    result = cast_array(
        arr,
        target_dtype="uint8",
        rounding_mode="nearest-even",
        scalar_map_entries=[],
    )
    assert np.array_equal(result, np.array([1], dtype=np.uint8))


# ---------------------------------------------------------------------------
# Matching behavior
# ---------------------------------------------------------------------------

BEHAVIOR_CASES = [
    Expect(
        input=dict(
            arr=np.array([float("nan"), np.float64("nan")], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            scalar_map_entries=[(nan, 42)],
        ),
        expected=np.array([42, 42], dtype=np.uint8),
        id="nan-matches-any-nan",
    ),
    Expect(
        input=dict(
            arr=np.array([42.0, 10.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            scalar_map_entries=[(42.0, 99)],
        ),
        expected=np.array([99, 10], dtype=np.uint8),
        id="exact-value-match",
    ),
    Expect(
        input=dict(
            arr=np.array([0, 1, 2], dtype=np.int32),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            scalar_map_entries=[(0, 255)],
        ),
        expected=np.array([255, 1, 2], dtype=np.uint8),
        id="int-to-int",
    ),
    Expect(
        input=dict(
            arr=np.array([1.0, nan], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
            scalar_map_entries=[(nan, nan)],
        ),
        expected=np.array([1.0, nan], dtype=np.float32),
        eq=nan_eq,
        id="float-nan-to-float-nan",
    ),
]


@pytest.mark.parametrize(
    "case", BEHAVIOR_CASES, ids=[c.id for c in BEHAVIOR_CASES]
)
def test_behavior(case: Expect):
    result = cast_array(**case.input)
    case.check(result)


# ---------------------------------------------------------------------------
# Numpy scalar values in entries
# ---------------------------------------------------------------------------


def test_numpy_scalar_entries_float():
    """Scalar map entries using numpy float scalars."""
    result = _cast_f64_to_u8(
        [(np.float64("nan"), np.uint8(0)), (np.float64("inf"), np.uint8(255))]
    )
    assert np.array_equal(result, np.array([1, 0, 255], dtype=np.uint8))


def test_numpy_scalar_entries_dict():
    """Scalar map dict with numpy scalar keys and values."""
    result = _cast_f64_to_u8(
        {np.float64("nan"): np.uint8(0), np.float64("inf"): np.uint8(255)}
    )
    assert np.array_equal(result, np.array([1, 0, 255], dtype=np.uint8))


def test_numpy_scalar_entries_int():
    """Scalar map entries using numpy int scalars."""
    result = cast_array(
        np.array([0, 1, 2], dtype=np.int32),
        target_dtype="uint8",
        rounding_mode="nearest-even",
        scalar_map_entries=[(np.int32(0), np.uint8(99))],
    )
    assert np.array_equal(result, np.array([99, 1, 2], dtype=np.uint8))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

ERROR_CASES = [
    ExpectFail(
        input=dict(
            arr=np.array([1.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            scalar_map_entries=42,
        ),
        exception=TypeError, match="iterable",
        id="not-iterable",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([1.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            scalar_map_entries=[(1.0, 2.0, 3.0)],
        ),
        exception=ValueError, match="pair",
        id="wrong-length",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([1.0, np.nan, 3.0], dtype=np.float64),
            target_dtype="int32",
            rounding_mode="nearest-even",
            scalar_map_entries=[(np.float64(np.nan), np.float64(0.0))],
        ),
        exception=TypeError, match=r"scalar_map target value.*target dtype int32",
        id="float-target-value-for-int-dtype",
    ),
]


@pytest.mark.parametrize("case", ERROR_CASES, ids=[c.id for c in ERROR_CASES])
def test_errors(case: ExpectFail):
    case.check(cast_array)
