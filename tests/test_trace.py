from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make ``src`` importable under plain ``python3 -m unittest discover -s tests``.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prod_gemini_agent.trace import CallRecord, RunTrace


class TraceTests(unittest.TestCase):
    def test_trace_snapshot_computes_percentiles_and_totals(self) -> None:
        """The summary numbers the demo prints come from this snapshot."""
        trace = RunTrace()
        trace.record(CallRecord(prompt_id="a", started_at=0, duration_ms=100, usd_cost=0.001, success=True))
        trace.record(CallRecord(prompt_id="b", started_at=0, duration_ms=200, usd_cost=0.002, success=True))
        trace.record(CallRecord(prompt_id="c", started_at=0, duration_ms=400, usd_cost=0.004, success=True))
        snap = trace.snapshot()
        self.assertEqual(snap["total_calls"], 3)
        self.assertEqual(snap["success"], 3)
        self.assertAlmostEqual(snap["total_usd"], 0.007)
        self.assertEqual(snap["p50_ms"], 200.0)
        self.assertGreater(snap["p95_ms"], 200.0)

    def test_trace_snapshot_empty_is_all_zeros(self) -> None:
        """A trace with no records yields a clean zeroed snapshot, not a crash."""
        snap = RunTrace().snapshot()
        self.assertEqual(snap["total_calls"], 0)
        self.assertEqual(snap["success"], 0)
        self.assertEqual(snap["failed"], 0)
        self.assertEqual(snap["total_usd"], 0.0)
        self.assertEqual(snap["p50_ms"], 0.0)
        self.assertEqual(snap["p95_ms"], 0.0)

    def test_trace_snapshot_separates_success_and_failure(self) -> None:
        """Cost and latency percentiles only count successful calls."""
        trace = RunTrace()
        trace.record(CallRecord(prompt_id="ok", started_at=0, duration_ms=50, usd_cost=0.01, success=True))
        trace.record(
            CallRecord(
                prompt_id="bad",
                started_at=0,
                duration_ms=999,
                usd_cost=0.0,
                success=False,
                error="boom",
            )
        )
        snap = trace.snapshot()
        self.assertEqual(snap["success"], 1)
        self.assertEqual(snap["failed"], 1)
        self.assertAlmostEqual(snap["total_usd"], 0.01)
        # The failed call's 999ms latency must not leak into the percentile.
        self.assertEqual(snap["p50_ms"], 50.0)

    def test_trace_writes_jsonl(self) -> None:
        """JSONL audit log is the single artifact the demo points at."""
        trace = RunTrace()
        trace.record(CallRecord(prompt_id="a", started_at=0, duration_ms=1, success=True))
        trace.record(
            CallRecord(prompt_id="b", started_at=0, duration_ms=2, success=False, error="boom")
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit.jsonl"
            trace.write_jsonl(out)
            lines = out.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        parsed = [json.loads(line) for line in lines]
        self.assertEqual(parsed[1]["error"], "boom")

    def test_trace_write_jsonl_creates_missing_parent_dirs(self) -> None:
        """The audit log path is created even when its directory does not exist yet."""
        trace = RunTrace()
        trace.record(CallRecord(prompt_id="a", started_at=0, duration_ms=1, success=True))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "dir" / "audit.jsonl"
            trace.write_jsonl(out)
            self.assertTrue(out.exists())
            self.assertEqual(len(out.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
