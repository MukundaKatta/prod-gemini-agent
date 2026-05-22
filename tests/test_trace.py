from __future__ import annotations

import json
from pathlib import Path

import pytest

from prod_gemini_agent.trace import CallRecord, RunTrace


def test_trace_snapshot_computes_percentiles_and_totals() -> None:
    """The summary numbers the demo prints come from this snapshot."""
    trace = RunTrace()
    trace.record(CallRecord(prompt_id="a", started_at=0, duration_ms=100, usd_cost=0.001, success=True))
    trace.record(CallRecord(prompt_id="b", started_at=0, duration_ms=200, usd_cost=0.002, success=True))
    trace.record(CallRecord(prompt_id="c", started_at=0, duration_ms=400, usd_cost=0.004, success=True))
    snap = trace.snapshot()
    assert snap["total_calls"] == 3
    assert snap["success"] == 3
    assert snap["total_usd"] == pytest.approx(0.007)
    assert snap["p50_ms"] == 200.0
    assert snap["p95_ms"] > 200.0


def test_trace_writes_jsonl(tmp_path: Path) -> None:
    """JSONL audit log is the single artifact the demo points at."""
    trace = RunTrace()
    trace.record(CallRecord(prompt_id="a", started_at=0, duration_ms=1, success=True))
    trace.record(CallRecord(prompt_id="b", started_at=0, duration_ms=2, success=False, error="boom"))
    out = tmp_path / "audit.jsonl"
    trace.write_jsonl(out)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[1]["error"] == "boom"
