from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make ``src`` importable when run as plain ``python3 -m unittest discover -s tests``
# (that invocation imports test modules top-level, so the package's bootstrap is
# skipped). Inserting here keeps the suite dependency-free and runner-agnostic.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prod_gemini_agent import (
    BudgetWindow,
    CircuitBreaker,
    FakeGeminiProvider,
    Fleet,
    ProductionAgent,
    ResponseCache,
    RetryPolicy,
    RunTrace,
    run_raw_gemini_baseline,
)
from prod_gemini_agent.client import GeminiResult, ProviderError


class _CountingProvider:
    """Provider that fails the first N calls and succeeds after."""

    def __init__(self, fail_first: int = 2) -> None:
        self._fail_first = fail_first
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def call(self, prompt: str) -> GeminiResult:
        self._calls += 1
        if self._calls <= self._fail_first:
            raise ProviderError(f"transient {self._calls}", retryable=True)
        return GeminiResult(
            text=f"summary[{self._calls}]", input_tokens=10, output_tokens=20, latency_ms=42.0
        )


def _serial_agent(provider, **kwargs) -> ProductionAgent:
    """Build a single-threaded agent so call ordering is deterministic in tests."""
    params = dict(
        fleet=Fleet(max_workers=1),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_s=0.0, sleep_fn=lambda _s: None),
    )
    params.update(kwargs)
    return ProductionAgent(provider=provider, **params)


class ProductionAgentTests(unittest.TestCase):
    def test_production_agent_lands_call_after_retries(self) -> None:
        """The composed stack survives 2 transient failures via the retry layer."""
        agent = _serial_agent(
            _CountingProvider(fail_first=2),
            retry_policy=RetryPolicy(max_attempts=5, base_delay_s=0.0, sleep_fn=lambda _s: None),
        )
        report = agent.run([("doc-1", "summarize this")])
        self.assertEqual(report.success, 1)
        self.assertEqual(report.failed, 0)
        self.assertGreaterEqual(report.retries, 1)

    def test_production_agent_uses_cache_on_repeat(self) -> None:
        """A repeated prompt hits the cache; the provider is only called once."""
        provider = _CountingProvider(fail_first=0)
        agent = _serial_agent(provider)
        report = agent.run([("doc-1", "same prompt"), ("doc-2", "same prompt")])
        self.assertGreaterEqual(report.cache_hits, 1)
        # The cache means exactly one real provider call for two identical prompts.
        self.assertEqual(provider.calls, 1)

    def test_production_agent_respects_budget_cap(self) -> None:
        """Budget exhaustion blocks further calls instead of silently overspending."""
        agent = _serial_agent(
            _CountingProvider(fail_first=0),
            budget=BudgetWindow(cap_usd=0.0000002, window_s=60.0),
        )
        report = agent.run(
            [
                ("doc-1", "first call"),
                ("doc-2", "second call"),
                ("doc-3", "third call"),
            ]
        )
        self.assertGreaterEqual(report.budget_blocks, 1)

    def test_production_agent_writes_audit_log(self) -> None:
        """The audit log lands on disk with one line per call."""
        agent = _serial_agent(_CountingProvider(fail_first=0))
        agent.run([("doc-1", "first"), ("doc-2", "second")])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit.jsonl"
            agent.write_audit_log(out)
            self.assertTrue(out.exists())
            self.assertEqual(len(out.read_text().splitlines()), 2)

    def test_baseline_is_unforgiving(self) -> None:
        """The notebook-grade baseline does not retry, so transient errors stick."""
        provider = _CountingProvider(fail_first=1)
        report = run_raw_gemini_baseline(
            [("doc-1", "first call"), ("doc-2", "second call")],
            provider,
            max_workers=1,
        )
        # One of the two calls hit the transient failure; the baseline does not retry.
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.success, 1)

    def test_fake_provider_demo_seed_reproduces(self) -> None:
        """End-to-end run with the demo's seed produces a non-empty summary."""
        agent = ProductionAgent(
            provider=FakeGeminiProvider(seed=7, error_rate=0.0, burst_failure_after=None),
            fleet=Fleet(max_workers=4),
            retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, sleep_fn=lambda _s: None),
            breaker=CircuitBreaker(failure_threshold=3, cooldown_s=0.1),
            budget=BudgetWindow(cap_usd=1.0, window_s=60.0),
            cache=ResponseCache(),
            trace=RunTrace(),
        )
        prompts = [(f"d{i}", f"prompt {i}") for i in range(8)]
        report = agent.run(prompts)
        self.assertEqual(report.success, 8)
        self.assertGreater(report.total_usd, 0.0)

    def test_report_outputs_surface_per_prompt_summaries(self) -> None:
        """``run`` returns one output slot per prompt, in submission order."""
        agent = _serial_agent(_CountingProvider(fail_first=0))
        report = agent.run([("doc-1", "alpha"), ("doc-2", "beta")])
        self.assertEqual(len(report.outputs), 2)
        self.assertTrue(all(out for out in report.outputs))


if __name__ == "__main__":
    unittest.main()
