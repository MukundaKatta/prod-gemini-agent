"""Response cache + hit-ratio observability.

Maps the contract used by ``cachebench`` (the user's published lib): the
cache itself is dumb, but it tracks the metrics that matter for production
agents : hits, misses, cost saved, and miss-aware retry hints.

For batch summarization there is real value in caching: in production the
same document gets re-summarized across many users, and replays during
debugging hit the same prompts repeatedly.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from .client import GeminiResult


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    usd_saved: float = 0.0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total


@dataclass
class ResponseCache:
    """Process-local cache.

    ``maxsize`` does FIFO eviction; nothing fancier : a real deployment
    would point this at Redis or a local LRU. We expose ``stats`` so the
    trace report can show hit-ratio without callers wiring anything up.
    """

    maxsize: int = 256
    _store: Dict[str, GeminiResult] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    stats: CacheStats = field(default_factory=CacheStats)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _key(prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}::{prompt}".encode("utf-8")).hexdigest()

    def get(self, prompt: str, *, model: str = "gemini-2.0-flash") -> Optional[GeminiResult]:
        k = self._key(prompt, model)
        with self._lock:
            cached = self._store.get(k)
            if cached is None:
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            self.stats.usd_saved += cached.usd_cost
            return cached

    def put(self, prompt: str, result: GeminiResult) -> None:
        k = self._key(prompt, result.model)
        with self._lock:
            if k in self._store:
                return
            if len(self._order) >= self.maxsize:
                evict = self._order.pop(0)
                self._store.pop(evict, None)
            self._store[k] = result
            self._order.append(k)
            self.stats.inserts += 1
