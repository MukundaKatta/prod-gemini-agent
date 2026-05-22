from __future__ import annotations

import pytest

from prod_gemini_agent.breaker import (
    BreakerOpen,
    BreakerState,
    CircuitBreaker,
)


def _ticking_clock(start: float = 0.0):
    """Returns a (now, advance) pair backed by a mutable cell."""
    cell = {"t": start}

    def now() -> float:
        return cell["t"]

    def advance(seconds: float) -> None:
        cell["t"] += seconds

    return now, advance


def test_breaker_opens_after_threshold_failures() -> None:
    """N consecutive failures trip the breaker to OPEN."""
    breaker = CircuitBreaker(failure_threshold=3, cooldown_s=0.1)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.trips == 1


def test_breaker_blocks_calls_while_open() -> None:
    """``call`` short-circuits with ``BreakerOpen`` while OPEN and pre-cooldown."""
    now, _advance = _ticking_clock()
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=10.0, clock=now)
    for _ in range(2):
        breaker.record_failure()
    with pytest.raises(BreakerOpen):
        breaker.call(lambda: "should not run")


def test_breaker_half_opens_after_cooldown() -> None:
    """After cooldown the breaker probes with one trial call."""
    now, advance = _ticking_clock()
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=0.5, clock=now)
    for _ in range(2):
        breaker.record_failure()
    advance(1.0)
    assert breaker.allow() is True  # Move to HALF_OPEN.
    assert breaker.state is BreakerState.HALF_OPEN


def test_breaker_recovers_on_half_open_success() -> None:
    """A successful trial call snaps the breaker back to CLOSED."""
    now, advance = _ticking_clock()
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=0.5, clock=now)
    for _ in range(2):
        breaker.record_failure()
    advance(1.0)
    result = breaker.call(lambda: "alive")
    assert result == "alive"
    assert breaker.state is BreakerState.CLOSED


def test_breaker_reopens_on_half_open_failure() -> None:
    """A failing trial call sends the breaker back to OPEN with a fresh timer."""
    now, advance = _ticking_clock()
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=0.5, clock=now)
    for _ in range(2):
        breaker.record_failure()
    advance(1.0)
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.trips == 2  # The half-open failure counts as a new trip.
