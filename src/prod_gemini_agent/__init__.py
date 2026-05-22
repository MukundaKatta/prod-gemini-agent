"""prod-gemini-agent: production-grade wrapper around Gemini for batch agent tasks.

A reference project that pairs Gemini 2.0 Flash with seven small governance libs
(llmfleet, llm-retry, llm-circuit-breaker, llm-budget-window, token-budget-pool,
cachebench, agenttrace) so the same agent can run reliably under real load.
"""
from __future__ import annotations

from .agent import ProductionAgent, run_raw_gemini_baseline, RunReport
from .client import GeminiClient, FakeGeminiProvider, GeminiResult, ProviderError
from .fleet import Fleet
from .retry import retry_call, RetryPolicy
from .breaker import CircuitBreaker, BreakerState, BreakerOpen
from .budget import BudgetWindow, BudgetExceeded
from .cache import ResponseCache, CacheStats
from .trace import RunTrace, CallRecord

__all__ = [
    "ProductionAgent",
    "RunReport",
    "run_raw_gemini_baseline",
    "GeminiClient",
    "FakeGeminiProvider",
    "GeminiResult",
    "ProviderError",
    "Fleet",
    "retry_call",
    "RetryPolicy",
    "CircuitBreaker",
    "BreakerState",
    "BreakerOpen",
    "BudgetWindow",
    "BudgetExceeded",
    "ResponseCache",
    "CacheStats",
    "RunTrace",
    "CallRecord",
]

__version__ = "0.1.0"
