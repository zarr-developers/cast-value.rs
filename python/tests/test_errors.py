"""Tests for error handling and argument validation."""

import numpy as np
import pytest

from zarr_cast_value import cast_array, cast_array_into

from .conftest import ExpectFail, ExpectedError

# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------

INVALID_ARG_CASES = [
    ExpectFail(
        input=dict(
            arr=np.array([1.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="bad",
        ),
        error=ExpectedError(ValueError, "Unknown rounding mode"),
        id="bad-rounding-mode",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([1.0], dtype=np.float64),
            target_dtype="uint8",
            rounding_mode="nearest-even",
            out_of_range_mode="bad",
        ),
        error=ExpectedError(ValueError, "Unknown out_of_range mode"),
        id="bad-out-of-range-mode",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([True, False], dtype=np.bool_),
            target_dtype="uint8",
            rounding_mode="nearest-even",
        ),
        error=ExpectedError(TypeError, "Unsupported dtype"),
        id="unsupported-source-dtype",
    ),
    ExpectFail(
        input=dict(
            arr=np.array([1.0], dtype=np.float64),
            target_dtype="bool",
            rounding_mode="nearest-even",
        ),
        error=ExpectedError(TypeError, "Unsupported target dtype"),
        id="unsupported-target-dtype",
    ),
]


@pytest.mark.parametrize(
    "case", INVALID_ARG_CASES, ids=[c.id for c in INVALID_ARG_CASES]
)
def test_invalid_args(case: ExpectFail):
    case.check(cast_array)


# ---------------------------------------------------------------------------
# Keyword-only enforcement
# ---------------------------------------------------------------------------

KEYWORD_ONLY_CASES = [
    ExpectFail(
        input=dict(args=(np.array([1.0], dtype=np.float64), "uint8", "nearest-even")),
        error=ExpectedError(TypeError, ""),
        id="cast-array-positional",
    ),
]


def test_cast_array_positional_args_rejected():
    with pytest.raises(TypeError):
        cast_array(np.array([1.0], dtype=np.float64), "uint8", "nearest-even")


def test_cast_array_into_positional_args_rejected():
    with pytest.raises(TypeError):
        cast_array_into(
            np.array([1.0], dtype=np.float64),
            np.zeros(1, dtype=np.uint8),
            "nearest-even",
        )


# ---------------------------------------------------------------------------
# Non-contiguous input
# ---------------------------------------------------------------------------


def test_non_contiguous_input():
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)[::2]
    assert not arr.flags["C_CONTIGUOUS"]
    case = ExpectFail(
        input=dict(arr=arr, target_dtype="uint8", rounding_mode="nearest-even"),
        error=ExpectedError(ValueError, "contiguous"),
    )
    case.check(cast_array)
