from __future__ import annotations

import os
import sys
import unittest

# Make ``src`` importable under plain ``python3 -m unittest discover -s tests``.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prod_gemini_agent.budget import BudgetExceeded, BudgetWindow


def _ticking_clock(start: float = 0.0):
    cell = {"t": start}
    return (lambda: cell["t"]), (lambda dt: cell.__setitem__("t", cell["t"] + dt))


class BudgetWindowTests(unittest.TestCase):
    def test_budget_admits_calls_below_cap(self) -> None:
        """Reservation under the cap moves the spent counter."""
        b = BudgetWindow(cap_usd=1.0, window_s=10.0)
        b.check_and_reserve(0.4)
        b.commit(actual_usd=0.4, reserved_usd=0.4)
        self.assertAlmostEqual(b.spent(), 0.4)
        self.assertAlmostEqual(b.remaining(), 0.6)

    def test_budget_rejects_calls_over_cap(self) -> None:
        """A reservation that would push the total over the cap raises."""
        b = BudgetWindow(cap_usd=1.0, window_s=10.0)
        b.check_and_reserve(0.7)
        b.commit(0.7, 0.7)
        with self.assertRaises(BudgetExceeded):
            b.check_and_reserve(0.5)

    def test_budget_window_slides(self) -> None:
        """Old spend falls out of the sliding window once enough time passes."""
        now, advance = _ticking_clock()
        b = BudgetWindow(cap_usd=1.0, window_s=10.0, clock=now)
        b.check_and_reserve(0.9)
        b.commit(0.9, 0.9)
        advance(11.0)
        self.assertAlmostEqual(b.spent(), 0.0)
        b.check_and_reserve(0.9)  # Should not raise.

    def test_budget_cancel_reservation_releases_capacity(self) -> None:
        """A failed call's reservation is released so the budget stays accurate."""
        b = BudgetWindow(cap_usd=1.0, window_s=10.0)
        b.check_and_reserve(0.6)
        b.cancel_reservation(0.6)
        self.assertAlmostEqual(b.spent(), 0.0)

    def test_budget_reservation_blocks_concurrent_overspend(self) -> None:
        """Reservations stack so two in-flight calls cannot both slip past the cap."""
        b = BudgetWindow(cap_usd=1.0, window_s=10.0)
        b.check_and_reserve(0.6)
        # A second reservation that fits only if the first were ignored must fail.
        with self.assertRaises(BudgetExceeded):
            b.check_and_reserve(0.6)
        # Reserved-but-not-committed spend is still counted as spent.
        self.assertAlmostEqual(b.spent(), 0.6)


if __name__ == "__main__":
    unittest.main()
