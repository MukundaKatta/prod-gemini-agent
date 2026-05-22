"""Sliding-window USD budget guard.

Drops a call before it goes out the door if it would push the recent
spend over the configured cap inside a rolling window. Mirrors
``llm-budget-window`` (and the ``token-budget-pool`` shared-cap idea)
on the user's published stack.

We keep timestamps + costs in a deque so the eviction step is O(1)
amortized. ``snapshot()`` returns a stable view for the trace report
without taking the lock for long.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Tuple


class BudgetExceeded(RuntimeError):
    """Raised by ``record`` when admitting the cost would breach the cap."""


@dataclass
class BudgetWindow:
    """Sliding USD budget.

    ``cap_usd`` is the max spend admitted inside any ``window_s`` seconds.
    Calls ``check_and_reserve(cost)`` before the upstream call, then
    ``commit(cost, ts)`` after success. If the call fails you should
    skip the commit so the reservation drops off naturally with the window.
    """

    cap_usd: float = 1.0
    window_s: float = 60.0
    clock: Callable[[], float] = field(default=time.monotonic)
    _records: Deque[Tuple[float, float]] = field(default_factory=collections.deque)
    _reserved: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _evict_locked(self) -> None:
        now = self.clock()
        cutoff = now - self.window_s
        while self._records and self._records[0][0] < cutoff:
            self._records.popleft()

    def spent(self) -> float:
        with self._lock:
            self._evict_locked()
            return sum(c for _, c in self._records) + self._reserved

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent())

    def check_and_reserve(self, estimated_usd: float) -> None:
        with self._lock:
            self._evict_locked()
            current = sum(c for _, c in self._records) + self._reserved
            if current + estimated_usd > self.cap_usd:
                raise BudgetExceeded(
                    f"reserving ${estimated_usd:.6f} would exceed cap "
                    f"${self.cap_usd:.6f}; current=${current:.6f}"
                )
            self._reserved += estimated_usd

    def commit(self, actual_usd: float, reserved_usd: float) -> None:
        with self._lock:
            self._evict_locked()
            self._reserved = max(0.0, self._reserved - reserved_usd)
            self._records.append((self.clock(), actual_usd))

    def cancel_reservation(self, reserved_usd: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - reserved_usd)
