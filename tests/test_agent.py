from __future__ import annotations

from pathlib import Path

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

    def call(self, prompt: str):
        self._calls += 1
        if self._calls <= self._fail_first:
            raise ProviderError(f"transient {self._calls}", retryable=True)
        return GeminiResult(
            text=f"summary[{self._calls}]", input_tokens=10, output_tokens=20, latency_ms=42.0
        )


def test_production_agent_lands_call_after_retries() -> None:
    """The composed stack survives 2 transient failures via the retry layer."""
    agent = ProductionAgent(
        provider=_CountingProvider(fail_first=2),
        fleet=Fleet(max_workers=1),
        retry_policy=RetryPolicy(max_attempts=5, base_delay_s=0.0, sleep_fn=lambda _s: None),
    )
    report = agent.run([("doc-1", "summarize this")])
    assert report.success == 1
    assert report.failed == 0
    assert report.retries >= 1


def test_production_agent_uses_cache_on_repeat() -> None:
    """A repeated prompt hits the cache; the call counter does not move."""
    provider = _CountingProvider(fail_first=0)
    agent = ProductionAgent(
        provider=provider,
        fleet=Fleet(max_workers=1),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_s=0.0, sleep_fn=lambda _s: None),
    )
    report = agent.run([("doc-1", "same prompt"), ("doc-2", "same prompt")])
    # One real call, one cache hit. We assert the cache hit count via the report.
    assert report.cache_hits >= 1


def test_production_agent_respects_budget_cap() -> None:
    """Budget exhaustion blocks further calls instead of silently overspending."""
    # Tiny cap so the second call's reservation must fail.
    agent = ProductionAgent(
        provider=_CountingProvider(fail_first=0),
        fleet=Fleet(max_workers=1),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_s=0.0, sleep_fn=lambda _s: None),
        budget=BudgetWindow(cap_usd=0.0000002, window_s=60.0),
    )
    report = agent.run([
        ("doc-1", "first call"),
        ("doc-2", "second call"),
        ("doc-3", "third call"),
    ])
    assert report.budget_blocks >= 1


def test_production_agent_writes_audit_log(tmp_path: Path) -> None:
    """The audit log lands on disk with one line per call."""
    agent = ProductionAgent(
        provider=_CountingProvider(fail_first=0),
        fleet=Fleet(max_workers=1),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_s=0.0, sleep_fn=lambda _s: None),
    )
    agent.run([("doc-1", "first"), ("doc-2", "second")])
    out = tmp_path / "audit.jsonl"
    agent.write_audit_log(out)
    assert out.exists()
    assert len(out.read_text().splitlines()) == 2


def test_baseline_is_unforgiving() -> None:
    """The notebook-grade baseline does not retry, so transient errors stick."""
    provider = _CountingProvider(fail_first=1)
    report = run_raw_gemini_baseline(
        [("doc-1", "first call"), ("doc-2", "second call")],
        provider,
        max_workers=1,
    )
    # One of the two calls hit the transient failure; the baseline does not retry.
    assert report.failed == 1
    assert report.success == 1


def test_fake_provider_demo_seed_reproduces() -> None:
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
    assert report.success == 8
    assert report.total_usd > 0.0
