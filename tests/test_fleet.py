from __future__ import annotations

import os
import sys
import unittest

# Make ``src`` importable under plain ``python3 -m unittest discover -s tests``.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prod_gemini_agent.fleet import Fleet, FleetTaskFailure


class FleetTests(unittest.TestCase):
    def test_fleet_runs_items_concurrently_and_preserves_order(self) -> None:
        """Result list aligns with input order even when workers complete out of order."""
        fleet = Fleet(max_workers=4)
        results = fleet.run([1, 2, 3, 4, 5], lambda x: x * 2)
        self.assertEqual(results, [2, 4, 6, 8, 10])

    def test_fleet_captures_worker_exceptions(self) -> None:
        """A worker failure becomes a ``FleetTaskFailure`` in its slot."""

        def worker(x: int) -> int:
            if x == 2:
                raise RuntimeError("boom on 2")
            return x

        fleet = Fleet(max_workers=4)
        results = fleet.run([1, 2, 3], worker)
        self.assertEqual(results[0], 1)
        self.assertIsInstance(results[1], FleetTaskFailure)
        self.assertIsInstance(results[1].error, RuntimeError)
        self.assertEqual(results[1].item, 2)
        self.assertEqual(results[2], 3)

    def test_fleet_rejects_invalid_concurrency(self) -> None:
        with self.assertRaises(ValueError):
            Fleet(max_workers=0)

    def test_fleet_empty_input_short_circuits(self) -> None:
        """Empty input returns an empty list without spinning the pool."""
        fleet = Fleet(max_workers=4)
        self.assertEqual(fleet.run([], lambda x: x), [])


if __name__ == "__main__":
    unittest.main()
