# prod-gemini-agent

Same Gemini batch task, two scenes. Before: raw notebook agent. After: governed production agent.

This is a reference project for the Google for Startups AI Agents Challenge (Track B, "optimize an existing prototype for production"). It wraps Gemini 2.0 Flash in seven small governance pieces and prints the difference.

## Quickstart

```bash
git clone https://github.com/MukundaKatta/prod-gemini-agent.git
cd prod-gemini-agent
python3 -m pytest tests/         # 33 tests, ~0.5 s
python3 examples/batch_summarize.py
```

No API key needed. The demo runs against `FakeGeminiProvider(seed=7)` so it is deterministic. Set `GEMINI_API_KEY` to run the same script against real Gemini 2.0 Flash.

## What the demo prints

```
=== Raw Gemini (notebook-grade) ===
  calls           : 20  (success 14 / failed 6)
  total cost (USD): $0.000229
  latency p50/p95 : 266.9ms / 337.5ms
  retries         : 0
  cache hits      : 0
  breaker trips   : 0
  budget blocks   : 0
  wall time       : 0.10s

=== ProductionAgent (governed) ===
  calls           : 20  (success 20 / failed 0)
  total cost (USD): $0.000311
  latency p50/p95 : 299.2ms / 369.8ms
  retries         : 8
  cache hits      : 1  (hit ratio 52.5%, saved $0.000344)
  breaker trips   : 0  (blocked 0)
  budget blocks   : 0
  budget remaining: $0.049689
  wall time       : 0.14s
```

Same input, same provider, same 18% transient failure rate. The baseline drops six summaries. The governed run keeps all twenty, hard-caps spend at $0.05, and writes an audit log to disk.

## Architecture

![prod-gemini-agent architecture: a batch of (doc-id, prompt) tasks is dispatched by a concurrent fleet, and each Gemini call passes through a per-call governance pipeline (cache, sliding USD budget, circuit breaker, bounded retry, the Gemini call with cost math, and a trace). The run emits a report (cost, latency p50/p95, retries, cache hits, breaker trips, budget) and an audit log. A raw notebook agent drops 6 of 20 calls; the governed agent keeps all 20, caps spend, and audits everything.](docs/architecture.png)

Each Gemini call is wrapped by seven small governance pieces: concurrent dispatch, a sliding USD budget, a circuit breaker, bounded retry with jitter, response caching, cost math, and a per-call trace. Same input and same transient-failure rate, the baseline drops summaries while the governed run keeps all of them under a hard spend cap.

## Governance map

Each governance feature lives in one small module and mirrors a published lib:

| Concern | Module | Mirrored lib |
| --- | --- | --- |
| Concurrent batch dispatch | `src/prod_gemini_agent/fleet.py` | [llmfleet](https://pypi.org/project/llmfleet/) |
| Bounded retry with jitter | `src/prod_gemini_agent/retry.py` | [llm-retry](https://crates.io/crates/llm-retry) |
| Circuit breaker | `src/prod_gemini_agent/breaker.py` | [llm-circuit-breaker](https://crates.io/crates/llm-circuit-breaker) |
| Sliding USD budget | `src/prod_gemini_agent/budget.py` | [llm-budget-window](https://crates.io/crates/llm-budget-window), [token-budget-py](https://pypi.org/project/token-budget-py/) |
| Cache hit-ratio observability | `src/prod_gemini_agent/cache.py` | [cachebench](https://pypi.org/project/cachebench/) |
| Per-run audit + cost/latency report | `src/prod_gemini_agent/trace.py` | [agenttrace](https://pypi.org/project/agenttrace/) |
| Cost math | `src/prod_gemini_agent/client.py` | [claude-cost](https://crates.io/crates/claude-cost), [bedrock-cost](https://crates.io/crates/bedrock-cost) |

These libs are tiny on purpose. Each one solves one problem and stays under 400 LOC. This project shows them composed.

## Swap to real Gemini in five lines

The demo defaults to `FakeGeminiProvider`. To call the real Gemini 2.0 Flash API:

```python
import os
from prod_gemini_agent import GeminiClient, ProductionAgent

provider = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
agent = ProductionAgent(provider=provider)
report = agent.run([("doc-1", "Summarize: ...")])
report.print()
```

For Vertex AI and Cloud Run, see [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Layout

```
src/prod_gemini_agent/
  client.py     # FakeGeminiProvider + GeminiClient + cost math
  fleet.py      # concurrent batch dispatch
  retry.py      # exponential backoff with jitter
  breaker.py    # CLOSED / OPEN / HALF_OPEN circuit breaker
  budget.py     # sliding-window USD cap
  cache.py      # response cache with hit-ratio stats
  trace.py      # per-call audit log + p50/p95 report
  agent.py      # ProductionAgent (composed) + raw baseline
examples/
  batch_summarize.py   # 90-second demo
tests/                 # 33 tests, FakeGeminiProvider only
docs/
  DEPLOY.md            # Vertex AI + Cloud Run notes
  DEMO_SCRIPT.md       # 90s video script
SUBMISSION.md          # Devpost write-up
```

## License

MIT. Author: Mukunda Katta.
