from __future__ import annotations

import pytest

from prod_gemini_agent.client import (
    FakeGeminiProvider,
    GeminiResult,
    ProviderError,
    GEMINI_2_FLASH_INPUT_USD_PER_MTOK,
    GEMINI_2_FLASH_OUTPUT_USD_PER_MTOK,
)


def test_fake_provider_is_deterministic() -> None:
    """Same seed + same prompts = identical text outputs."""
    a = FakeGeminiProvider(seed=7, error_rate=0.0, burst_failure_after=None)
    b = FakeGeminiProvider(seed=7, error_rate=0.0, burst_failure_after=None)
    prompts = ["hello world", "foo bar baz", "another doc"]
    for p in prompts:
        assert a.call(p).text == b.call(p).text


def test_fake_provider_injects_burst_failures() -> None:
    """The first burst-failure window flips the provider's mood."""
    provider = FakeGeminiProvider(
        seed=7, error_rate=0.0, burst_failure_after=2
    )
    provider.call("warm-up call 1")
    with pytest.raises(ProviderError):
        provider.call("triggers burst")
    with pytest.raises(ProviderError):
        provider.call("still in burst")
    with pytest.raises(ProviderError):
        provider.call("end of burst")


def test_gemini_result_cost_uses_published_rates() -> None:
    """Cost math matches the publicly listed Gemini 2.0 Flash rates."""
    r = GeminiResult(text="hi", input_tokens=1_000_000, output_tokens=1_000_000, latency_ms=10.0)
    expected = GEMINI_2_FLASH_INPUT_USD_PER_MTOK + GEMINI_2_FLASH_OUTPUT_USD_PER_MTOK
    assert r.usd_cost == pytest.approx(expected)


def test_provider_error_carries_retryable_flag() -> None:
    """The retry layer relies on the ``retryable`` discriminator."""
    err = ProviderError("nope", retryable=False)
    assert err.retryable is False
