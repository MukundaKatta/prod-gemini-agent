from __future__ import annotations

from prod_gemini_agent.cache import ResponseCache
from prod_gemini_agent.client import GeminiResult


def _result(text: str = "out") -> GeminiResult:
    return GeminiResult(text=text, input_tokens=10, output_tokens=20, latency_ms=50.0)


def test_cache_returns_none_on_miss() -> None:
    cache = ResponseCache()
    assert cache.get("prompt-1") is None
    assert cache.stats.misses == 1


def test_cache_returns_value_on_hit_and_counts_savings() -> None:
    """Hit ratio + USD saved are the metrics the demo prints."""
    cache = ResponseCache()
    result = _result()
    cache.put("prompt-1", result)
    cached = cache.get("prompt-1")
    assert cached is not None
    assert cached.text == "out"
    assert cache.stats.hits == 1
    assert cache.stats.usd_saved > 0.0


def test_cache_eviction_fifo() -> None:
    """At capacity, the oldest entry leaves and the newest stays."""
    cache = ResponseCache(maxsize=2)
    cache.put("a", _result("a"))
    cache.put("b", _result("b"))
    cache.put("c", _result("c"))
    assert cache.get("a") is None  # Evicted.
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_cache_keyed_per_model() -> None:
    """Different models keep their own slots so we don't cross-contaminate."""
    cache = ResponseCache()
    flash = GeminiResult(text="flash", input_tokens=1, output_tokens=1, latency_ms=1.0, model="gemini-2.0-flash")
    pro = GeminiResult(text="pro", input_tokens=1, output_tokens=1, latency_ms=1.0, model="gemini-2.0-pro")
    cache.put("same prompt", flash)
    cache.put("same prompt", pro)
    assert cache.get("same prompt", model="gemini-2.0-flash") is not None
    assert cache.get("same prompt", model="gemini-2.0-pro") is not None
