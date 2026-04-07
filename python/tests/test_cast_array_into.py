"""Tests for cast_array_into (pre-allocated output variant)."""

from math import nan

import numpy as np
import pytest

from cast_value_rs import cast_array_into

from .conftest import Expect, ExpectFail


def _run_into(case: Expect) -> None:
    """Call cast_array_into with the input kwargs and check the output buffer."""
    kwargs = case.input.copy()
    out = kwargs.pop("out")
    cast_array_into(**kwargs, out=out)
    case.check(out)


CAST_INTO_CASES = [
    Expect(
        input=dict(
            arr=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            out=np.zeros(3, dtype=np.uint8),
            rounding_mode="nearest-even",
        ),
        expected=np.array([1, 2, 3], dtype=np.uint8),
        id="float-to-int",
    ),
    Expect(
        input=dict(
            arr=np.array([1, 2, 3], dtype=np.int32),
            out=np.zeros(3, dtype=np.uint8),
            rounding_mode="nearest-even",
        ),
        expected=np.array([1, 2, 3], dtype=np.uint8),
        id="int-to-int",
    ),
    Expect(
        input=dict(
            arr=np.array([1.0, 2.0], dtype=np.float64),
            out=np.zeros(2, dtype=np.float32),
            rounding_mode="nearest-even",
        ),
        expected=np.array([1.0, 2.0], dtype=np.float32),
        id="float-to-float",
    ),
    Expect(
        input=dict(
            arr=np.array([1, 2], dtype=np.int32),
            out=np.zeros(2, dtype=np.float64),
            rounding_mode="nearest-even",
        ),
        expected=np.array([1.0, 2.0], dtype=np.float64),
        id="int-to-float",
    ),
    Expect(
        input=dict(
            arr=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            out=np.zeros((2, 2), dtype=np.uint8),
            rounding_mode="nearest-even",
        ),
        expected=np.array([[1, 2], [3, 4]], dtype=np.uint8),
        id="multidimensional",
    ),
    Expect(
        input=dict(
            arr=np.array([300.0, -10.0], dtype=np.float64),
            out=np.zeros(2, dtype=np.uint8),
            rounding_mode="nearest-even",
            out_of_range_mode="clamp",
        ),
        expected=np.array([255, 0], dtype=np.uint8),
        id="with-clamp",
    ),
    Expect(
        input=dict(
            arr=np.array([nan, 1.0], dtype=np.float64),
            out=np.zeros(2, dtype=np.uint8),
            rounding_mode="nearest-even",
            scalar_map_entries=[(nan, 0)],
        ),
        expected=np.array([0, 1], dtype=np.uint8),
        id="with-scalar-map",
    ),
]


@pytest.mark.parametrize(
    "case", CAST_INTO_CASES, ids=[c.id for c in CAST_INTO_CASES]
)
def test_cast_array_into(case: Expect):
    _run_into(case)


def test_cast_array_into_returns_none():
    arr = np.array([1.0], dtype=np.float64)
    out = np.zeros(1, dtype=np.uint8)
    result = cast_array_into(arr, out, rounding_mode="nearest-even")
    assert result is None


CAST_INTO_FAIL_CASES = [
    ExpectFail(
        input=dict(
            arr=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            out=np.zeros(2, dtype=np.uint8),
            rounding_mode="nearest-even",
        ),
        exception=ValueError, match="Shape mismatch",
        id="shape-mismatch",
    ),
]


@pytest.mark.parametrize(
    "case", CAST_INTO_FAIL_CASES, ids=[c.id for c in CAST_INTO_FAIL_CASES]
)
def test_cast_array_into_errors(case: ExpectFail):
    case.check(cast_array_into)
