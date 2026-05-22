"""Batched concurrent dispatch.

Mirrors the design of ``llmfleet``: a small pool that takes a list of
work items and a worker function, runs them concurrently with a hard
parallelism cap, and returns results in submission order. Failures bubble
up as ``FleetTaskFailure`` so callers see exactly which item failed.

We use ``concurrent.futures.ThreadPoolExecutor`` because Gemini calls
are I/O bound. For a real deployment swap to async, but threads keep
this reference project easy to read.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Callable, Generic, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class FleetTaskFailure(Generic[T]):
    """Captures a worker exception alongside the input that triggered it."""

    item: T
    error: BaseException


class Fleet:
    """Thin concurrent runner with a hard parallelism cap.

    ``run(items, worker)`` returns a list aligned with ``items``: each
    slot is either the worker's return value or a ``FleetTaskFailure``.
    """

    def __init__(self, *, max_workers: int = 8) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._max_workers = max_workers

    def run(
        self,
        items: Sequence[T],
        worker: Callable[[T], R],
    ) -> List[R | FleetTaskFailure[T]]:
        if not items:
            return []
        results: List[R | FleetTaskFailure[T]] = [None] * len(items)  # type: ignore
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_idx = {pool.submit(worker, item): i for i, item in enumerate(items)}
            for fut in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except BaseException as exc:  # noqa: BLE001 - intentional
                    results[idx] = FleetTaskFailure(item=items[idx], error=exc)
        return results
