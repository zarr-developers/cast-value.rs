"""Tests for cast_array (allocating variant)."""

from math import inf, nan

import numpy as np
import pytest

from zarr_cast_value import cast_array

from .conftest import Expect, ExpectFail, ExpectedError, nan_eq

# ---------------------------------------------------------------------------
# float -> int
# ---------------------------------------------------------------------------

FLOAT_TO_INT_CASES = [
    Expect(
        input=dict(
            arr=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([1, 2, 3], dtype=np.uint8),
        id="basic",
    ),
    Expect(
        input=dict(
            arr=np.array([2.5, 3.5], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([2, 4], dtype=np.uint8),
        id="nearest-even",
    ),
    Expect(
        input=dict(
            arr=np.array([2.7, -2.7], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="towards-zero",
        ),
        expected=np.array([2, -2], dtype=np.int8),
        id="towards-zero",
    ),
    Expect(
        input=dict(
            arr=np.array([2.1, -2.7], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="towards-positive",
        ),
        expected=np.array([3, -2], dtype=np.int8),
        id="towards-positive",
    ),
    Expect(
        input=dict(
            arr=np.array([2.7, -2.1], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="towards-negative",
        ),
        expected=np.array([2, -3], dtype=np.int8),
        id="towards-negative",
    ),
    Expect(
        input=dict(
            arr=np.array([2.5, -2.5], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="nearest-away",
        ),
        expected=np.array([3, -3], dtype=np.int8),
        id="nearest-away",
    ),
    Expect(
        input=dict(
            arr=np.array([300.0, -10.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            out_of_range_mode="clamp",
        ),
        expected=np.array([255, 0], dtype=np.uint8),
        id="clamp",
    ),
    Expect(
        input=dict(
            arr=np.array([200.0], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="nearest-even",
            out_of_range_mode="wrap",
        ),
        expected=np.array([-56], dtype=np.int8),
        id="wrap",
    ),
    Expect(
        input=dict(
            arr=np.array([inf, -inf], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            out_of_range_mode="clamp",
        ),
        expected=np.array([255, 0], dtype=np.uint8),
        id="inf-clamp",
    ),
    Expect(
        input=dict(
            arr=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            target_dtype="int16",
            rounding_mode="nearest-even",
        ),
        expected=np.array([1, 2, 3], dtype=np.int16),
        id="f32-source",
    ),
]


@pytest.mark.parametrize(
    "case",
    FLOAT_TO_INT_CASES,
    ids=[c.id for c in FLOAT_TO_INT_CASES],
)
def test_float_to_int(case: Expect):
    result = cast_array(**case.input)
    case.check(result)


FLOAT_TO_INT_FAIL_CASES = [
    ExpectFail(
        input=dict(
            arr=np.array([300.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        error=ExpectedError(ValueError, "out of range"),
        id="out-of-range",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([nan], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        error=ExpectedError(ValueError, "NaN"),
        id="nan",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([inf], dtype=np.float64),
            target_dtype="int8",
            rounding_mode="nearest-even",
            out_of_range_mode="wrap",
        ),
        error=ExpectedError(ValueError, "cast"),
        id="inf-wrap",
    ),
]


@pytest.mark.parametrize(
    "case",
    FLOAT_TO_INT_FAIL_CASES,
    ids=[c.id for c in FLOAT_TO_INT_FAIL_CASES],
)
def test_float_to_int_errors(case: ExpectFail):
    case.check(cast_array)


# ---------------------------------------------------------------------------
# int -> int
# ---------------------------------------------------------------------------

INT_TO_INT_CASES = [
    Expect(
        input=dict(
            arr=np.array([1, 2, 3], dtype=np.int32),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([1, 2, 3], dtype=np.uint8),
        id="basic",
    ),
    Expect(
        input=dict(
            arr=np.array([300, -1], dtype=np.int32),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            out_of_range_mode="clamp",
        ),
        expected=np.array([255, 0], dtype=np.uint8),
        id="clamp",
    ),
    Expect(
        input=dict(
            arr=np.array([300, -1], dtype=np.int32),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            out_of_range_mode="wrap",
        ),
        expected=np.array([44, 255], dtype=np.uint8),
        id="wrap",
    ),
    Expect(
        input=dict(
            arr=np.array([0, 127, -128], dtype=np.int8),
            target_dtype="int32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([0, 127, -128], dtype=np.int32),
        id="widening",
    ),
]


@pytest.mark.parametrize(
    "case", INT_TO_INT_CASES, ids=[c.id for c in INT_TO_INT_CASES]
)
def test_int_to_int(case: Expect):
    result = cast_array(**case.input)
    case.check(result)


INT_TO_INT_FAIL_CASES = [
    ExpectFail(
        input=dict(
            arr=np.array([300], dtype=np.int32),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        error=ExpectedError(ValueError, "out of range"),
        id="out-of-range",
    ),
]


@pytest.mark.parametrize(
    "case", INT_TO_INT_FAIL_CASES, ids=[c.id for c in INT_TO_INT_FAIL_CASES]
)
def test_int_to_int_errors(case: ExpectFail):
    case.check(cast_array)


# ---------------------------------------------------------------------------
# float -> float
# ---------------------------------------------------------------------------

FLOAT_TO_FLOAT_CASES = [
    Expect(
        input=dict(
            arr=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        id="f64-to-f32",
    ),
    Expect(
        input=dict(
            arr=np.array([nan], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([nan], dtype=np.float32),
        eq=nan_eq,
        id="nan-propagates",
    ),
    Expect(
        input=dict(
            arr=np.array([inf, -inf], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([inf, -inf], dtype=np.float32),
        id="inf-propagates",
    ),
    Expect(
        input=dict(
            arr=np.array([-0.0], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        expected=np.array([-0.0], dtype=np.float32),
        eq=lambda a, b: bool(
            np.array_equal(a, b) and np.signbit(a[0]) == np.signbit(b[0])
        ),
        id="signed-zero",
    ),
]


@pytest.mark.parametrize(
    "case", FLOAT_TO_FLOAT_CASES, ids=[c.id for c in FLOAT_TO_FLOAT_CASES]
)
def test_float_to_float(case: Expect):
    result = cast_array(**case.input)
    case.check(result)


_overflow_val = float(np.finfo(np.float32).max) * 1.5

FLOAT_TO_FLOAT_FAIL_CASES = [
    ExpectFail(
        input=dict(
            arr=np.array([_overflow_val], dtype=np.float64),
            target_dtype="float32",
            rounding_mode="nearest-even",
        ),
        error=ExpectedError(ValueError, "out of range"),
        id="overflow",
    ),
]


@pytest.mark.parametrize(
    "case",
    FLOAT_TO_FLOAT_FAIL_CASES,
    ids=[c.id for c in FLOAT_TO_FLOAT_FAIL_CASES],
)
def test_float_to_float_errors(case: ExpectFail):
    case.check(cast_array)


def test_float_to_float_overflow_clamp():
    result = cast_array(
        np.array([_overflow_val], dtype=np.float64),
        target_dtype="float32",
        rounding_mode="nearest-even",
        out_of_range_mode="clamp",
    )
    assert np.isinf(result[0]) and result[0] > 0


# ---------------------------------------------------------------------------
# int -> float
# ---------------------------------------------------------------------------

INT_TO_FLOAT_CASES = [
    Expect(
        input=dict(
            arr=np.array([0, -1, 32767, -32768], dtype=np.int16),
            target_dtype="float64",
            rounding_mode="nearest-even",
        ),
        expected=np.array([0.0, -1.0, 32767.0, -32768.0], dtype=np.float64),
        id="lossless-widening",
    ),
]


@pytest.mark.parametrize(
    "case", INT_TO_FLOAT_CASES, ids=[c.id for c in INT_TO_FLOAT_CASES]
)
def test_int_to_float(case: Expect):
    result = cast_array(**case.input)
    case.check(result)


def test_int_to_float_rounding():
    """i64 values near 2^24 can't all be exactly represented in f32."""
    val = (1 << 24) + 1  # 16777217
    arr = np.array([val], dtype=np.int64)

    result_pos = cast_array(
        arr, target_dtype="float32", rounding_mode="towards-positive"
    )
    assert float(result_pos[0]) >= val

    result_tz = cast_array(
        arr, target_dtype="float32", rounding_mode="towards-zero"
    )
    assert float(result_tz[0]) <= val


# ---------------------------------------------------------------------------
# shape / misc
# ---------------------------------------------------------------------------

SHAPE_CASES = [
    Expect(
        input=dict(
            arr=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([[1, 2], [3, 4]], dtype=np.uint8),
        id="2d",
    ),
    Expect(
        input=dict(
            arr=np.array([], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        expected=np.array([], dtype=np.uint8),
        id="empty",
    ),
]


@pytest.mark.parametrize("case", SHAPE_CASES, ids=[c.id for c in SHAPE_CASES])
def test_shape(case: Expect):
    result = cast_array(**case.input)
    assert result.shape == case.expected.shape
    case.check(result)
