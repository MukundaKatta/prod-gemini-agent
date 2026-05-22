"""Per-run trace + audit log.

Mirrors the contract from ``agenttrace`` (Python) and ``agenttrace-rs``:
collect per-call records, compute totals (cost, p50/p95 latency, retries,
error counts), and expose a stable ``snapshot`` for downstream reports.

This is the single source of truth the ``ProductionAgent`` reads when it
prints its before/after report and when it writes the JSONL audit log.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class CallRecord:
    """One upstream call (regardless of retry count)."""

    prompt_id: str
    started_at: float
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    usd_cost: float = 0.0
    success: bool = True
    retries: int = 0
    cache_hit: bool = False
    breaker_blocked: bool = False
    budget_blocked: bool = False
    error: Optional[str] = None


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


@dataclass
class RunTrace:
    """Captures every call inside one batch run.

    Use ``record`` from worker threads; the lock keeps the list consistent.
    Call ``snapshot`` at the end for the printable summary, and optionally
    ``write_jsonl`` to persist an audit trail to disk for the demo.
    """

    started_at: float = field(default_factory=time.time)
    records: List[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, call: CallRecord) -> None:
        with self._lock:
            self.records.append(call)

    def snapshot(self) -> dict:
        with self._lock:
            successes = [r for r in self.records if r.success]
            failures = [r for r in self.records if not r.success]
            latencies = sorted(r.duration_ms for r in successes)
            total_cost = sum(r.usd_cost for r in successes)
            total_retries = sum(r.retries for r in self.records)
            cache_hits = sum(1 for r in self.records if r.cache_hit)
            breaker_blocks = sum(1 for r in self.records if r.breaker_blocked)
            budget_blocks = sum(1 for r in self.records if r.budget_blocked)
            return {
                "total_calls": len(self.records),
                "success": len(successes),
                "failed": len(failures),
                "total_usd": round(total_cost, 6),
                "p50_ms": round(_percentile(latencies, 0.50), 2),
                "p95_ms": round(_percentile(latencies, 0.95), 2),
                "retries": total_retries,
                "cache_hits": cache_hits,
                "breaker_blocks": breaker_blocks,
                "budget_blocks": budget_blocks,
            }

    def write_jsonl(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("w", encoding="utf-8") as fh:
                for r in self.records:
                    fh.write(json.dumps(asdict(r)) + "\n")
