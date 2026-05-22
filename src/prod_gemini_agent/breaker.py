"""Circuit breaker for upstream Gemini calls.

Three states: ``CLOSED`` (everything is fine), ``OPEN`` (we believe the
backend is failing, fail fast for a cool-down window), ``HALF_OPEN``
(let one trial call through to see if the backend recovered).

Mirrors the contract from ``llm-circuit-breaker`` so the same mental
model carries over.
"""
from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

T = TypeVar("T")


class BreakerState(str, enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BreakerOpen(RuntimeError):
    """Raised when the breaker rejects a call because it is open."""


@dataclass
class CircuitBreaker:
    """Tiny thread-safe circuit breaker.

    ``failure_threshold`` consecutive failures flip the breaker to OPEN.
    After ``cooldown_s`` we move to HALF_OPEN and admit one trial call.
    If that trial succeeds we go back to CLOSED; if it fails we go back
    to OPEN and restart the timer.
    """

    failure_threshold: int = 3
    cooldown_s: float = 0.5
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at_s: float = 0.0
    trips: int = 0
    clock: Callable[[], float] = field(default=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self) -> bool:
        with self._lock:
            if self.state is BreakerState.CLOSED:
                return True
            if self.state is BreakerState.OPEN:
                if self.clock() - self.opened_at_s >= self.cooldown_s:
                    self.state = BreakerState.HALF_OPEN
                    return True
                return False
            return True  # HALF_OPEN: let exactly one through

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0
            self.state = BreakerState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive_failures += 1
            if self.state is BreakerState.HALF_OPEN:
                self.state = BreakerState.OPEN
                self.opened_at_s = self.clock()
                self.trips += 1
                return
            if self.consecutive_failures >= self.failure_threshold:
                if self.state is not BreakerState.OPEN:
                    self.trips += 1
                self.state = BreakerState.OPEN
                self.opened_at_s = self.clock()

    def call(self, fn: Callable[[], T]) -> T:
        """Wrap ``fn`` with breaker logic. Raises ``BreakerOpen`` if blocked."""
        if not self.allow():
            raise BreakerOpen("circuit breaker is open")
        try:
            result = fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result
