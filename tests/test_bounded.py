"""utils/bounded: a call that may never return gets a wall-clock bound."""
import time

import pytest

from utils.bounded import call_bounded


def test_a_quick_call_returns_its_value():
    assert call_bounded(lambda a, b=0: a + b, 1.0, 2, b=3) == 5


def test_the_callee_s_exception_is_the_caller_s():
    def boom():
        raise ValueError("no")
    with pytest.raises(ValueError, match="no"):
        call_bounded(boom, 1.0)


def test_a_hung_call_raises_timeout_at_the_bound():
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        call_bounded(time.sleep, 0.2, 5)
    assert time.monotonic() - t0 < 2.0
