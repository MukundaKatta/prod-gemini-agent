"""Gemini client surface.

This module exposes the contract every downstream lib expects: a small
sync ``call`` method that takes a prompt and returns a ``GeminiResult``.

Two implementations are provided:

* ``FakeGeminiProvider`` : deterministic, seeded latency/error injection.
  Used everywhere in tests and in the default demo so a contributor can
  run the project without a key. Maps the API surface of Gemini 2.0 Flash.
* ``GeminiClient`` : thin shim over ``google.generativeai``. Imported lazily
  so the package stays usable without the optional dependency.

Pricing constants mirror the public Gemini 2.0 Flash rates as of May 2026
(input $0.10 / 1M tokens, output $0.40 / 1M tokens). Update if Google
publishes new tiers; the agent surfaces a ``RunReport`` that reads from here.
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Optional, Protocol


# Public pricing. Per 1M tokens, USD. Mirrors Gemini 2.0 Flash list price.
GEMINI_2_FLASH_INPUT_USD_PER_MTOK = 0.10
GEMINI_2_FLASH_OUTPUT_USD_PER_MTOK = 0.40


class ProviderError(RuntimeError):
    """Raised by a Gemini provider for retryable or fatal failures.

    ``retryable=True`` covers 429/5xx-style transient faults. ``False``
    covers permanent failures (bad input, auth) and skips the retry loop.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class GeminiResult:
    """Normalized response from any Gemini provider implementation."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str = "gemini-2.0-flash"

    @property
    def usd_cost(self) -> float:
        in_cost = (self.input_tokens / 1_000_000.0) * GEMINI_2_FLASH_INPUT_USD_PER_MTOK
        out_cost = (self.output_tokens / 1_000_000.0) * GEMINI_2_FLASH_OUTPUT_USD_PER_MTOK
        return in_cost + out_cost


class GeminiProvider(Protocol):
    """Minimum surface every provider must implement."""

    def call(self, prompt: str) -> GeminiResult: ...


class FakeGeminiProvider:
    """Deterministic fake Gemini.

    Uses a seeded PRNG to inject realistic latency and a configurable
    transient error rate. The same seed and the same prompt list always
    produce the same trace, so demos and tests stay reproducible.

    Set ``error_rate`` between 0 and 1. Errors are tagged ``retryable=True``
    so they exercise the retry + breaker path naturally.
    """

    def __init__(
        self,
        *,
        seed: int = 7,
        base_latency_ms: float = 180.0,
        latency_jitter_ms: float = 220.0,
        error_rate: float = 0.18,
        model: str = "gemini-2.0-flash",
        burst_failure_after: Optional[int] = 6,
    ) -> None:
        self._rng = random.Random(seed)
        self._base = base_latency_ms
        self._jitter = latency_jitter_ms
        self._error_rate = error_rate
        self._model = model
        self._calls = 0
        self._burst_after = burst_failure_after

    def call(self, prompt: str) -> GeminiResult:
        self._calls += 1

        # Synthetic latency. We sleep a tiny amount so wall-clock numbers
        # in the demo feel real without slowing tests too much.
        latency_ms = self._base + self._rng.random() * self._jitter
        time.sleep(latency_ms / 10_000.0)  # 10x faster than real for demo speed.

        # Inject transient failures. Once we cross ``burst_after`` we
        # raise three errors in a row to make the breaker trip in the demo.
        if self._burst_after is not None and self._calls in (
            self._burst_after,
            self._burst_after + 1,
            self._burst_after + 2,
        ):
            raise ProviderError("simulated 503 from gemini backend", retryable=True)

        if self._rng.random() < self._error_rate:
            raise ProviderError("simulated rate-limit (429)", retryable=True)

        # Hash the prompt for a stable output and synthetic token counts.
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        text = f"summary[{digest}]: {prompt[:80]}..."
        input_tokens = max(8, len(prompt) // 4)
        output_tokens = max(16, len(text) // 4)
        return GeminiResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=self._model,
        )


class GeminiClient:
    """Thin shim over ``google.generativeai.GenerativeModel``.

    We keep the import lazy so the package installs cleanly without the
    optional ``gemini`` extra. The shim translates Google's exceptions
    into ``ProviderError`` so the retry/breaker libs see a single error
    type and can decide on their own policies.
    """

    def __init__(self, *, api_key: str, model: str = "gemini-2.0-flash") -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised by the user.
            raise ImportError(
                "google-generativeai is not installed. "
                "Install with: pip install 'prod-gemini-agent[gemini]'"
            ) from exc
        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(model)

    def call(self, prompt: str) -> GeminiResult:  # pragma: no cover - real network.
        started = time.perf_counter()
        try:
            response = self._model.generate_content(prompt)
        except Exception as exc:  # google client raises many distinct types.
            msg = str(exc).lower()
            retryable = any(tok in msg for tok in ("429", "503", "rate", "deadline"))
            raise ProviderError(str(exc), retryable=retryable) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        # google-generativeai exposes usage_metadata on the response.
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return GeminiResult(
            text=getattr(response, "text", ""),
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            model=self._model_name,
        )
