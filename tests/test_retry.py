from __future__ import annotations

import pytest

from prod_gemini_agent.client import ProviderError
from prod_gemini_agent.retry import RetryPolicy, retry_call


def test_retry_succeeds_after_transient_failures() -> None:
    """A bounded number of transient errors still lets a call land."""
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("transient", retryable=True)
        return "ok"

    policy = RetryPolicy(max_attempts=4, base_delay_s=0.0, sleep_fn=lambda _s: None)
    assert retry_call(fn, policy=policy) == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_after_max_attempts() -> None:
    """Once the policy is exhausted the last error is re-raised."""
    def fn() -> str:
        raise ProviderError("always fails", retryable=True)

    policy = RetryPolicy(max_attempts=3, base_delay_s=0.0, sleep_fn=lambda _s: None)
    with pytest.raises(ProviderError):
        retry_call(fn, policy=policy)


def test_retry_does_not_retry_non_retryable() -> None:
    """Permanent errors bubble up immediately and skip the budget."""
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise ProviderError("bad input", retryable=False)

    policy = RetryPolicy(max_attempts=5, base_delay_s=0.0, sleep_fn=lambda _s: None)
    with pytest.raises(ProviderError):
        retry_call(fn, policy=policy)
    assert calls["n"] == 1


def test_retry_on_retry_callback_fires() -> None:
    """Observability hook reports each retry attempt to the caller."""
    seen: list[int] = []

    def fn() -> str:
        if len(seen) < 2:
            raise ProviderError("transient", retryable=True)
        return "ok"

    def on_retry(attempt: int, _exc: ProviderError) -> None:
        seen.append(attempt)

    policy = RetryPolicy(max_attempts=5, base_delay_s=0.0, sleep_fn=lambda _s: None)
    retry_call(fn, policy=policy, on_retry=on_retry)
    assert seen == [0, 1]
