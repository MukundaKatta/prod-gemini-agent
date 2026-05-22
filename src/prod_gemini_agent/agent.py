"""ProductionAgent: the governed wrapper around a Gemini provider.

This is the file the README points to first. It wires the seven small
governance pieces into one composed object:

* cache (cachebench-style hit/miss tracking)
* budget (llm-budget-window-style USD cap)
* breaker (llm-circuit-breaker-style fast-fail)
* retry (llm-retry-style bounded backoff)
* fleet (llmfleet-style concurrent dispatch)
* trace (agenttrace-style audit + summary)
* client (Gemini 2.0 Flash, real or fake)

A baseline ``run_raw_gemini_baseline`` is provided so the demo can show
the same task without any of the governance layers : that is the
"notebook-grade" scene in the before/after pitch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .breaker import BreakerOpen, CircuitBreaker
from .budget import BudgetExceeded, BudgetWindow
from .cache import ResponseCache
from .client import (
    FakeGeminiProvider,
    GeminiProvider,
    GeminiResult,
    ProviderError,
    GEMINI_2_FLASH_INPUT_USD_PER_MTOK,
    GEMINI_2_FLASH_OUTPUT_USD_PER_MTOK,
)
from .fleet import Fleet, FleetTaskFailure
from .retry import RetryPolicy, retry_call
from .trace import CallRecord, RunTrace


@dataclass
class RunReport:
    """Printable report produced by ``ProductionAgent.run`` and the baseline."""

    label: str
    total_calls: int
    success: int
    failed: int
    total_usd: float
    p50_ms: float
    p95_ms: float
    retries: int
    cache_hits: int
    breaker_blocks: int
    budget_blocks: int
    wall_seconds: float
    budget_remaining_usd: Optional[float] = None
    breaker_trips: int = 0
    cache_hit_ratio: float = 0.0
    cache_usd_saved: float = 0.0
    outputs: List[str] = field(default_factory=list)

    def print(self) -> None:
        print(f"\n=== {self.label} ===")
        print(f"  calls           : {self.total_calls}  (success {self.success} / failed {self.failed})")
        print(f"  total cost (USD): ${self.total_usd:.6f}")
        print(f"  latency p50/p95 : {self.p50_ms:.1f}ms / {self.p95_ms:.1f}ms")
        print(f"  retries         : {self.retries}")
        print(f"  cache hits      : {self.cache_hits}  (hit ratio {self.cache_hit_ratio:.1%}, saved ${self.cache_usd_saved:.6f})")
        print(f"  breaker trips   : {self.breaker_trips}  (blocked {self.breaker_blocks})")
        print(f"  budget blocks   : {self.budget_blocks}")
        if self.budget_remaining_usd is not None:
            print(f"  budget remaining: ${self.budget_remaining_usd:.6f}")
        print(f"  wall time       : {self.wall_seconds:.2f}s")


def _estimate_cost_usd(prompt: str) -> float:
    """Conservative cost estimate used for budget reservation.

    We assume 1 input token ~ 4 chars and budget for a fixed-size output
    reply. This is a deliberate over-estimate so the budget reservation
    is safe even if the real reply is longer than expected.
    """
    in_tok = max(8, len(prompt) // 4)
    out_tok = 256
    in_cost = (in_tok / 1_000_000.0) * GEMINI_2_FLASH_INPUT_USD_PER_MTOK
    out_cost = (out_tok / 1_000_000.0) * GEMINI_2_FLASH_OUTPUT_USD_PER_MTOK
    return in_cost + out_cost


@dataclass
class ProductionAgent:
    """Composed governance stack around a Gemini provider."""

    provider: GeminiProvider
    fleet: Fleet = field(default_factory=lambda: Fleet(max_workers=8))
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    budget: BudgetWindow = field(default_factory=lambda: BudgetWindow(cap_usd=0.05, window_s=60.0))
    cache: ResponseCache = field(default_factory=ResponseCache)
    trace: RunTrace = field(default_factory=RunTrace)

    def _single_call(self, prompt_id: str, prompt: str) -> CallRecord:
        started = time.perf_counter()
        # 1) cache lookup
        cached = self.cache.get(prompt)
        if cached is not None:
            return CallRecord(
                prompt_id=prompt_id,
                started_at=started,
                duration_ms=0.0,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                usd_cost=0.0,  # cache hit, no new cost
                success=True,
                cache_hit=True,
            )

        # 2) budget reservation
        estimated = _estimate_cost_usd(prompt)
        try:
            self.budget.check_and_reserve(estimated)
        except BudgetExceeded as exc:
            return CallRecord(
                prompt_id=prompt_id,
                started_at=started,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                success=False,
                budget_blocked=True,
                error=str(exc),
            )

        retries_used = 0

        def _provider_call() -> GeminiResult:
            return self.provider.call(prompt)

        def _on_retry(attempt: int, _exc: ProviderError) -> None:
            nonlocal retries_used
            retries_used = attempt + 1

        try:
            result = self.breaker.call(
                lambda: retry_call(_provider_call, policy=self.retry_policy, on_retry=_on_retry)
            )
        except BreakerOpen as exc:
            self.budget.cancel_reservation(estimated)
            return CallRecord(
                prompt_id=prompt_id,
                started_at=started,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                success=False,
                retries=retries_used,
                breaker_blocked=True,
                error=str(exc),
            )
        except ProviderError as exc:
            self.budget.cancel_reservation(estimated)
            return CallRecord(
                prompt_id=prompt_id,
                started_at=started,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                success=False,
                retries=retries_used,
                error=str(exc),
            )

        # 3) success path: cache the result and commit the real cost.
        self.cache.put(prompt, result)
        self.budget.commit(actual_usd=result.usd_cost, reserved_usd=estimated)
        return CallRecord(
            prompt_id=prompt_id,
            started_at=started,
            duration_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            usd_cost=result.usd_cost,
            success=True,
            retries=retries_used,
        )

    def run(self, prompts: Sequence[tuple[str, str]], *, label: str = "ProductionAgent") -> RunReport:
        """Run a list of ``(prompt_id, prompt_text)`` pairs and return a report."""
        wall_started = time.perf_counter()

        def _worker(item: tuple[str, str]) -> CallRecord:
            record = self._single_call(item[0], item[1])
            self.trace.record(record)
            return record

        outcomes = self.fleet.run(prompts, _worker)
        wall_seconds = time.perf_counter() - wall_started

        # Capture the actual text outputs for the demo. We re-pull from
        # the cache after the run so successful prompts surface their summary.
        outputs: List[str] = []
        for prompt_id, prompt in prompts:
            cached = self.cache.get(prompt)
            outputs.append(cached.text if cached is not None else f"<no result for {prompt_id}>")

        snap = self.trace.snapshot()
        return RunReport(
            label=label,
            total_calls=snap["total_calls"],
            success=snap["success"],
            failed=snap["failed"],
            total_usd=snap["total_usd"],
            p50_ms=snap["p50_ms"],
            p95_ms=snap["p95_ms"],
            retries=snap["retries"],
            cache_hits=snap["cache_hits"],
            breaker_blocks=snap["breaker_blocks"],
            budget_blocks=snap["budget_blocks"],
            wall_seconds=wall_seconds,
            budget_remaining_usd=self.budget.remaining(),
            breaker_trips=self.breaker.trips,
            cache_hit_ratio=self.cache.stats.hit_ratio,
            cache_usd_saved=self.cache.stats.usd_saved,
            outputs=outputs,
        )

    def write_audit_log(self, path: Path | str) -> None:
        """Persist the per-call trace to a JSONL file for after-the-fact review."""
        self.trace.write_jsonl(Path(path))


def run_raw_gemini_baseline(
    prompts: Sequence[tuple[str, str]],
    provider: GeminiProvider,
    *,
    max_workers: int = 8,
    label: str = "Raw Gemini (notebook-grade)",
) -> RunReport:
    """Run the same batch without any governance layer.

    This is the "before" scene in the pitch. It shows what most teams
    actually ship in week one: one provider call per item, no retry,
    no breaker, no budget, no cache, no trace. The number of failed
    calls and the cost variance is the whole point.
    """
    wall_started = time.perf_counter()
    trace = RunTrace()
    fleet = Fleet(max_workers=max_workers)

    def _worker(item: tuple[str, str]) -> None:
        prompt_id, prompt = item
        started = time.perf_counter()
        try:
            result = provider.call(prompt)
        except ProviderError as exc:
            trace.record(
                CallRecord(
                    prompt_id=prompt_id,
                    started_at=started,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    success=False,
                    error=str(exc),
                )
            )
            return
        trace.record(
            CallRecord(
                prompt_id=prompt_id,
                started_at=started,
                duration_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                usd_cost=result.usd_cost,
                success=True,
            )
        )

    outcomes = fleet.run(prompts, _worker)
    wall_seconds = time.perf_counter() - wall_started
    snap = trace.snapshot()
    return RunReport(
        label=label,
        total_calls=snap["total_calls"],
        success=snap["success"],
        failed=snap["failed"],
        total_usd=snap["total_usd"],
        p50_ms=snap["p50_ms"],
        p95_ms=snap["p95_ms"],
        retries=snap["retries"],
        cache_hits=0,
        breaker_blocks=0,
        budget_blocks=0,
        wall_seconds=wall_seconds,
        budget_remaining_usd=None,
        breaker_trips=0,
    )
