from __future__ import annotations

import os
import sys
import unittest

# Make ``src`` importable under plain ``python3 -m unittest discover -s tests``.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prod_gemini_agent.cache import ResponseCache
from prod_gemini_agent.client import GeminiResult


def _result(text: str = "out") -> GeminiResult:
    return GeminiResult(text=text, input_tokens=10, output_tokens=20, latency_ms=50.0)


class ResponseCacheTests(unittest.TestCase):
    def test_cache_returns_none_on_miss(self) -> None:
        cache = ResponseCache()
        self.assertIsNone(cache.get("prompt-1"))
        self.assertEqual(cache.stats.misses, 1)

    def test_cache_returns_value_on_hit_and_counts_savings(self) -> None:
        """Hit ratio + USD saved are the metrics the demo prints."""
        cache = ResponseCache()
        result = _result()
        cache.put("prompt-1", result)
        cached = cache.get("prompt-1")
        self.assertIsNotNone(cached)
        assert cached is not None  # narrow for type checkers
        self.assertEqual(cached.text, "out")
        self.assertEqual(cache.stats.hits, 1)
        self.assertGreater(cache.stats.usd_saved, 0.0)

    def test_cache_hit_ratio_reflects_hits_and_misses(self) -> None:
        """``hit_ratio`` is 0.0 with no traffic and tracks hits over total."""
        cache = ResponseCache()
        self.assertEqual(cache.stats.hit_ratio, 0.0)
        cache.put("p", _result())
        cache.get("p")        # hit
        cache.get("absent")   # miss
        self.assertAlmostEqual(cache.stats.hit_ratio, 0.5)

    def test_cache_eviction_fifo(self) -> None:
        """At capacity, the oldest entry leaves and the newest stays."""
        cache = ResponseCache(maxsize=2)
        cache.put("a", _result("a"))
        cache.put("b", _result("b"))
        cache.put("c", _result("c"))
        self.assertIsNone(cache.get("a"))  # Evicted.
        self.assertIsNotNone(cache.get("b"))
        self.assertIsNotNone(cache.get("c"))

    def test_cache_keyed_per_model(self) -> None:
        """Different models keep their own slots so we don't cross-contaminate."""
        cache = ResponseCache()
        flash = GeminiResult(
            text="flash", input_tokens=1, output_tokens=1, latency_ms=1.0, model="gemini-2.0-flash"
        )
        pro = GeminiResult(
            text="pro", input_tokens=1, output_tokens=1, latency_ms=1.0, model="gemini-2.0-pro"
        )
        cache.put("same prompt", flash)
        cache.put("same prompt", pro)
        self.assertIsNotNone(cache.get("same prompt", model="gemini-2.0-flash"))
        self.assertIsNotNone(cache.get("same prompt", model="gemini-2.0-pro"))


if __name__ == "__main__":
    unittest.main()
