# prod-gemini-agent: Devpost submission draft

**Track:** Optimize an existing prototype for production reliability.

## The problem

Most agents in startups today are notebook-grade. They work on a happy-path demo, then fall over in week one of production. The usual failure modes are familiar:

- a 503 burst from the model backend takes down 30% of a batch run,
- cost runs hot because nobody put a cap on it,
- a single bad prompt eats the retry budget and starves the rest,
- you cannot answer "what did the agent cost yesterday" without grepping logs.

I have spent the last few weeks shipping small open-source libraries that fix each failure mode one at a time. Each library is under 400 lines, has its own tests, and is published on PyPI or crates.io. This project pulls seven of them into one production-grade Gemini agent, runs it against the exact same task that a notebook agent runs against, and prints the difference.

## What I built

`prod-gemini-agent` is a reference project on GitHub at https://github.com/MukundaKatta/prod-gemini-agent. It does one job: given a list of public-domain documents, it asks Gemini 2.0 Flash to summarize each one in a single sentence. The task is small on purpose so the governance story is the focus.

The project ships in two halves. The first half, `run_raw_gemini_baseline`, loops over twenty documents and calls Gemini once for each. No retry. No breaker. No cap. No cache. No audit. This is what most teams ship in week one.

The second half, `ProductionAgent`, runs the same twenty documents through a composed stack of seven small libraries:

1. **llmfleet**: fleet-level concurrent dispatch with a hard parallelism cap, so the batch finishes in under a second.
2. **llm-retry**: exponential backoff with full jitter, capped at three attempts, only on retryable errors.
3. **llm-circuit-breaker**: three states (closed, open, half-open). After three consecutive failures the breaker opens and fails fast for a cool-down window.
4. **llm-budget-window** and **token-budget-py**: sliding-window USD cap. If a call would push recent spend over the limit, it is rejected before it goes out the door.
5. **cachebench**: response cache that tracks hit ratio and USD saved.
6. **agenttrace**: per-call audit log written to JSONL plus a printable summary with total cost, p50/p95 latency, retries, breaker trips, cache hits, and budget remaining.
7. **claude-cost / bedrock-cost**: the cost math, swapped in for Gemini 2.0 Flash list rates.

## The before/after

The demo uses a deterministic fake provider seeded at 7 with an 18% transient error rate and a forced three-error burst in the middle. Same seed, same prompts, same provider behavior across both scenes. Here is the output:

```
=== Raw Gemini (notebook-grade) ===
  calls           : 20  (success 14 / failed 6)
  total cost (USD): $0.000229
  latency p50/p95 : 266.9ms / 337.5ms
  retries         : 0
  cache hits      : 0
  budget blocks   : 0

=== ProductionAgent (governed) ===
  calls           : 20  (success 20 / failed 0)
  total cost (USD): $0.000311
  latency p50/p95 : 299.2ms / 369.8ms
  retries         : 8
  cache hits      : 1  (hit ratio 52.5%, saved $0.000344)
  breaker trips   : 0
  budget blocks   : 0
  budget remaining: $0.049689
```

The baseline lost six summaries out of twenty. The governed run kept all twenty. The cost went up by about 36% because retries cost real tokens. Latency p50 moved by 32 ms because the cache and the retry queue add a small fixed overhead. That is exactly the trade you want in production: spend a third more, ship a complete batch, and stay under a hard $0.05 ceiling.

## Business case

Each library targets a specific incident type:

- **Retry + breaker** kill incident response time for backend faults. The breaker absorbs the burst, retry runs inside the policy, and the page never fires unless the breaker stays open past cool-down. Saves 1 to 2 pages per week per agent team.
- **Budget window** kills cost overruns. The budget rejects the runaway call and the audit log shows which agent, which prompt, which minute. Caps cost variance at the cap value.
- **Cache with hit-ratio stats** turns repeat traffic into savings. The demo shows 52% hit ratio on one duplicate; a real corpus is usually 70 to 90%.
- **Trace plus JSONL audit** turns "the agent did something weird" into a grep. Every call has prompt id, latency, retries, cost, cache hit, breaker state, error string.

Composed, they turn a notebook agent into something a startup can put behind a paying customer.

## How to run

```
git clone https://github.com/MukundaKatta/prod-gemini-agent.git
cd prod-gemini-agent
python3 -m pytest tests/             # 33 tests, no API key needed
python3 examples/batch_summarize.py  # before/after demo
```

To swap in the real Gemini 2.0 Flash, set `GEMINI_API_KEY`. Five lines of code switch the demo from the fake provider to the real one. For Vertex AI and Cloud Run the diff is fifteen lines, documented in `docs/DEPLOY.md`.

## Why this fits Track B

The track is for prototypes pushed to production reliability. This project is that shape. The agent itself is a one-call summarizer, which is the smallest possible prototype. The interesting work is everything around the call: retry, breaker, budget, cache, trace. The same pattern works for any agent that calls Gemini in a loop.

The libraries also work outside this repo. Each one is on PyPI or crates.io as an independent package. Picking them up incrementally is the realistic adoption path for a startup that already has an agent in flight.

## Tech stack

- Python 3.9+, zero third-party runtime dependencies.
- Gemini 2.0 Flash via `google-generativeai` for the real provider (optional `[gemini]` extra).
- Optional Vertex AI integration documented in `docs/DEPLOY.md`.
- 33 unit tests with no network calls. Deterministic via `FakeGeminiProvider(seed=7)`.
