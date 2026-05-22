"""Bounded retry with full-jitter exponential backoff.

This mirrors the contract used by ``llm-retry`` (the user's Rust crate)
and ``llm-retry`` style policies seen across his published libs. The
policy is intentionally tiny so callers can compose it with the breaker
and the budget without surprises.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from .client import ProviderError

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Retry budget and backoff curve.

    ``max_attempts`` includes the first try. Sleep grows as
    ``min(max_delay_s, base_delay_s * 2**attempt)`` and then gets
    multiplied by a uniform [0, 1) jitter so retried calls do not
    line up on the same wall-clock tick.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.05
    max_delay_s: float = 2.0
    sleep_fn: Callable[[float], None] = field(default=time.sleep)
    rng: random.Random = field(default_factory=lambda: random.Random(13))


def retry_call(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    on_retry: Callable[[int, ProviderError], None] | None = None,
) -> T:
    """Call ``fn`` with bounded retries on retryable ``ProviderError``.

    A non-retryable error is re-raised on the spot. Anything that is not
    a ``ProviderError`` bubbles up untouched so unrelated bugs surface fast.
    """
    p = policy or RetryPolicy()
    last_exc: ProviderError | None = None
    for attempt in range(p.max_attempts):
        try:
            return fn()
        except ProviderError as exc:
            if not exc.retryable:
                raise
            last_exc = exc
            if on_retry is not None:
                on_retry(attempt, exc)
            if attempt >= p.max_attempts - 1:
                break
            delay = min(p.max_delay_s, p.base_delay_s * (2 ** attempt))
            p.sleep_fn(delay * p.rng.random())
    assert last_exc is not None  # mypy hint; loop guarantees this.
    raise last_exc
