from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest


def default_eq(a: Any, b: Any) -> bool:
    """Default equality: element-wise comparison via numpy."""
    return bool(np.array_equal(a, b))


def nan_eq(a: Any, b: Any) -> bool:
    """Equality that treats NaN == NaN as True."""
    return bool(np.array_equal(a, b, equal_nan=True))


@dataclass
class Expect:
    """A test case: call the function on `input`, check it matches `expected`."""

    input: Any
    expected: Any
    eq: Callable[[Any, Any], bool] = field(default=default_eq)
    id: str | None = None

    def check(self, actual: Any) -> None:
        assert self.eq(actual, self.expected), f"expected {self.expected!r}, got {actual!r}"


@dataclass
class ExpectFail:
    """A test case that should raise an exception."""

    input: Any
    exception: type[Exception]
    match: str
    id: str | None = None

    def check(self, fn: Callable[..., Any]) -> None:
        with pytest.raises(self.exception, match=self.match):
            fn(**self.input)
