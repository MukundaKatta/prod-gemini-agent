from __future__ import annotations

import pytest

from prod_gemini_agent.budget import BudgetExceeded, BudgetWindow


def _ticking_clock(start: float = 0.0):
    cell = {"t": start}
    return (lambda: cell["t"]), (lambda dt: cell.__setitem__("t", cell["t"] + dt))


def test_budget_admits_calls_below_cap() -> None:
    """Reservation under the cap moves the spent counter."""
    b = BudgetWindow(cap_usd=1.0, window_s=10.0)
    b.check_and_reserve(0.4)
    b.commit(actual_usd=0.4, reserved_usd=0.4)
    assert b.spent() == pytest.approx(0.4)
    assert b.remaining() == pytest.approx(0.6)


def test_budget_rejects_calls_over_cap() -> None:
    """A reservation that would push the total over the cap raises."""
    b = BudgetWindow(cap_usd=1.0, window_s=10.0)
    b.check_and_reserve(0.7)
    b.commit(0.7, 0.7)
    with pytest.raises(BudgetExceeded):
        b.check_and_reserve(0.5)


def test_budget_window_slides() -> None:
    """Old spend falls out of the sliding window once enough time passes."""
    now, advance = _ticking_clock()
    b = BudgetWindow(cap_usd=1.0, window_s=10.0, clock=now)
    b.check_and_reserve(0.9)
    b.commit(0.9, 0.9)
    advance(11.0)
    assert b.spent() == pytest.approx(0.0)
    b.check_and_reserve(0.9)  # Should not raise.


def test_budget_cancel_reservation_releases_capacity() -> None:
    """A failed call's reservation is released so the budget stays accurate."""
    b = BudgetWindow(cap_usd=1.0, window_s=10.0)
    b.check_and_reserve(0.6)
    b.cancel_reservation(0.6)
    assert b.spent() == pytest.approx(0.0)
