from __future__ import annotations

import pytest

from prod_gemini_agent.fleet import Fleet, FleetTaskFailure


def test_fleet_runs_items_concurrently_and_preserves_order() -> None:
    """Result list aligns with input order even when workers complete out of order."""
    fleet = Fleet(max_workers=4)
    results = fleet.run([1, 2, 3, 4, 5], lambda x: x * 2)
    assert results == [2, 4, 6, 8, 10]


def test_fleet_captures_worker_exceptions() -> None:
    """A worker failure becomes a ``FleetTaskFailure`` in its slot."""
    def worker(x: int) -> int:
        if x == 2:
            raise RuntimeError("boom on 2")
        return x

    fleet = Fleet(max_workers=4)
    results = fleet.run([1, 2, 3], worker)
    assert results[0] == 1
    assert isinstance(results[1], FleetTaskFailure)
    assert isinstance(results[1].error, RuntimeError)
    assert results[2] == 3


def test_fleet_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError):
        Fleet(max_workers=0)


def test_fleet_empty_input_short_circuits() -> None:
    """Empty input returns an empty list without spinning the pool."""
    fleet = Fleet(max_workers=4)
    assert fleet.run([], lambda x: x) == []
