from __future__ import annotations

import os
import sys
import unittest

# Make ``src`` importable under plain ``python3 -m unittest discover -s tests``.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

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


class CircuitBreakerTests(unittest.TestCase):
    def test_breaker_opens_after_threshold_failures(self) -> None:
        """N consecutive failures trip the breaker to OPEN."""
        breaker = CircuitBreaker(failure_threshold=3, cooldown_s=0.1)
        for _ in range(3):
            breaker.record_failure()
        self.assertIs(breaker.state, BreakerState.OPEN)
        self.assertEqual(breaker.trips, 1)

    def test_breaker_blocks_calls_while_open(self) -> None:
        """``call`` short-circuits with ``BreakerOpen`` while OPEN and pre-cooldown."""
        now, _advance = _ticking_clock()
        breaker = CircuitBreaker(failure_threshold=2, cooldown_s=10.0, clock=now)
        for _ in range(2):
            breaker.record_failure()
        with self.assertRaises(BreakerOpen):
            breaker.call(lambda: "should not run")

    def test_breaker_half_opens_after_cooldown(self) -> None:
        """After cooldown the breaker probes with one trial call."""
        now, advance = _ticking_clock()
        breaker = CircuitBreaker(failure_threshold=2, cooldown_s=0.5, clock=now)
        for _ in range(2):
            breaker.record_failure()
        advance(1.0)
        self.assertTrue(breaker.allow())  # Move to HALF_OPEN.
        self.assertIs(breaker.state, BreakerState.HALF_OPEN)

    def test_breaker_recovers_on_half_open_success(self) -> None:
        """A successful trial call snaps the breaker back to CLOSED."""
        now, advance = _ticking_clock()
        breaker = CircuitBreaker(failure_threshold=2, cooldown_s=0.5, clock=now)
        for _ in range(2):
            breaker.record_failure()
        advance(1.0)
        result = breaker.call(lambda: "alive")
        self.assertEqual(result, "alive")
        self.assertIs(breaker.state, BreakerState.CLOSED)

    def test_breaker_reopens_on_half_open_failure(self) -> None:
        """A failing trial call sends the breaker back to OPEN with a fresh timer."""
        now, advance = _ticking_clock()
        breaker = CircuitBreaker(failure_threshold=2, cooldown_s=0.5, clock=now)
        for _ in range(2):
            breaker.record_failure()
        advance(1.0)
        self.assertTrue(breaker.allow())
        breaker.record_failure()
        self.assertIs(breaker.state, BreakerState.OPEN)
        self.assertEqual(breaker.trips, 2)  # The half-open failure counts as a new trip.

    def test_breaker_propagates_underlying_error_and_counts_failure(self) -> None:
        """A raising ``fn`` records a failure and re-raises the original error."""
        breaker = CircuitBreaker(failure_threshold=2, cooldown_s=10.0)

        def boom() -> str:
            raise RuntimeError("upstream blew up")

        with self.assertRaises(RuntimeError):
            breaker.call(boom)
        self.assertEqual(breaker.consecutive_failures, 1)
        self.assertIs(breaker.state, BreakerState.CLOSED)  # below threshold


if __name__ == "__main__":
    unittest.main()
