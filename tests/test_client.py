from __future__ import annotations

import os
import sys
import unittest

# Make ``src`` importable under plain ``python3 -m unittest discover -s tests``.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prod_gemini_agent.client import (
    FakeGeminiProvider,
    GeminiResult,
    ProviderError,
    GEMINI_2_FLASH_INPUT_USD_PER_MTOK,
    GEMINI_2_FLASH_OUTPUT_USD_PER_MTOK,
)


class ClientTests(unittest.TestCase):
    def test_fake_provider_is_deterministic(self) -> None:
        """Same seed + same prompts = identical text outputs."""
        a = FakeGeminiProvider(seed=7, error_rate=0.0, burst_failure_after=None)
        b = FakeGeminiProvider(seed=7, error_rate=0.0, burst_failure_after=None)
        prompts = ["hello world", "foo bar baz", "another doc"]
        for p in prompts:
            self.assertEqual(a.call(p).text, b.call(p).text)

    def test_fake_provider_injects_burst_failures(self) -> None:
        """The first burst-failure window flips the provider's mood."""
        provider = FakeGeminiProvider(seed=7, error_rate=0.0, burst_failure_after=2)
        provider.call("warm-up call 1")
        with self.assertRaises(ProviderError):
            provider.call("triggers burst")
        with self.assertRaises(ProviderError):
            provider.call("still in burst")
        with self.assertRaises(ProviderError):
            provider.call("end of burst")

    def test_gemini_result_cost_uses_published_rates(self) -> None:
        """Cost math matches the publicly listed Gemini 2.0 Flash rates."""
        r = GeminiResult(
            text="hi", input_tokens=1_000_000, output_tokens=1_000_000, latency_ms=10.0
        )
        expected = GEMINI_2_FLASH_INPUT_USD_PER_MTOK + GEMINI_2_FLASH_OUTPUT_USD_PER_MTOK
        self.assertAlmostEqual(r.usd_cost, expected)

    def test_gemini_result_cost_scales_with_token_split(self) -> None:
        """Input and output tokens are priced on their own published rates."""
        in_only = GeminiResult(
            text="x", input_tokens=1_000_000, output_tokens=0, latency_ms=1.0
        )
        out_only = GeminiResult(
            text="x", input_tokens=0, output_tokens=1_000_000, latency_ms=1.0
        )
        self.assertAlmostEqual(in_only.usd_cost, GEMINI_2_FLASH_INPUT_USD_PER_MTOK)
        self.assertAlmostEqual(out_only.usd_cost, GEMINI_2_FLASH_OUTPUT_USD_PER_MTOK)
        # Output tokens are the pricier side of the bill.
        self.assertGreater(out_only.usd_cost, in_only.usd_cost)

    def test_provider_error_carries_retryable_flag(self) -> None:
        """The retry layer relies on the ``retryable`` discriminator."""
        err = ProviderError("nope", retryable=False)
        self.assertFalse(err.retryable)
        self.assertTrue(ProviderError("transient").retryable)  # defaults to True


if __name__ == "__main__":
    unittest.main()
