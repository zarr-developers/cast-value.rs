"""Tests for scalar_map_entries parsing and behavior."""

from math import inf, nan

import numpy as np
import pytest

from zarr_cast_value import cast_array

from .conftest import Expect, ExpectFail, ExpectedError, nan_eq


def _cast_f64_to_u8(scalar_map_entries):
    """Helper: cast a fixed float64 array to uint8 with the given scalar map."""
    return cast_array(
        np.array([1.0, nan, inf], dtype=np.float64),
        target_dtype="uint8",
        rounding_mode="nearest-even",
        out_of_range_mode="clamp",
        scalar_map_entries=scalar_map_entries,
    )


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
        error=ExpectedError(TypeError, "iterable"),
        id="not-iterable",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([1.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            scalar_map_entries=[(1.0, 2.0, 3.0)],
        ),
        error=ExpectedError(ValueError, "pair"),
        id="wrong-length",
    ),
]


@pytest.mark.parametrize("case", ERROR_CASES, ids=[c.id for c in ERROR_CASES])
def test_errors(case: ExpectFail):
    case.check(cast_array)
