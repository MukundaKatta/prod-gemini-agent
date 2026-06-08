from __future__ import annotations

import os
import sys
import unittest

# Make ``src`` importable under plain ``python3 -m unittest discover -s tests``.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prod_gemini_agent.client import ProviderError
from prod_gemini_agent.retry import RetryPolicy, retry_call


class RetryTests(unittest.TestCase):
    def test_retry_succeeds_after_transient_failures(self) -> None:
        """A bounded number of transient errors still lets a call land."""
        calls = {"n": 0}

        def fn() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderError("transient", retryable=True)
            return "ok"

        policy = RetryPolicy(max_attempts=4, base_delay_s=0.0, sleep_fn=lambda _s: None)
        self.assertEqual(retry_call(fn, policy=policy), "ok")
        self.assertEqual(calls["n"], 3)

    def test_retry_gives_up_after_max_attempts(self) -> None:
        """Once the policy is exhausted the last error is re-raised."""

        def fn() -> str:
            raise ProviderError("always fails", retryable=True)

        policy = RetryPolicy(max_attempts=3, base_delay_s=0.0, sleep_fn=lambda _s: None)
        with self.assertRaises(ProviderError):
            retry_call(fn, policy=policy)

    def test_retry_does_not_retry_non_retryable(self) -> None:
        """Permanent errors bubble up immediately and skip the budget."""
        calls = {"n": 0}

        def fn() -> str:
            calls["n"] += 1
            raise ProviderError("bad input", retryable=False)

        policy = RetryPolicy(max_attempts=5, base_delay_s=0.0, sleep_fn=lambda _s: None)
        with self.assertRaises(ProviderError):
            retry_call(fn, policy=policy)
        self.assertEqual(calls["n"], 1)

    def test_retry_on_retry_callback_fires(self) -> None:
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
        self.assertEqual(seen, [0, 1])

    def test_retry_rejects_non_positive_attempt_budget(self) -> None:
        """A non-positive attempt budget is a config error, not a silent no-op.

        With ``max_attempts <= 0`` the loop body never runs. ``retry_call``
        must raise a clear ``ValueError`` rather than the opaque
        ``AssertionError`` (or, under ``python -O``, a ``TypeError``) that the
        old internal assertion produced. ``fn`` must never be called.
        """

        def fn() -> str:
            raise AssertionError("fn must not be called when max_attempts <= 0")

        policy = RetryPolicy(max_attempts=0, base_delay_s=0.0, sleep_fn=lambda _s: None)
        with self.assertRaises(ValueError):
            retry_call(fn, policy=policy)


if __name__ == "__main__":
    unittest.main()
