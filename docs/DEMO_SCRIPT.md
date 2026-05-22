# 90-second demo video script

Three shots. No voiceover gymnastics. Plain English, sounds like a real person.

## Shot 1: the problem (0:00 to 0:25)

**Visual:** terminal window with the repo open. Cursor on `examples/batch_summarize.py`. Pan to the top of `agent.py` showing the `ProductionAgent` class signature.

**Narration:**

> "Most agent demos work. Most agents in production do not. The same Gemini call that succeeds in your notebook fails one in five times when the backend has a bad afternoon. I built a small reference project that shows the difference."

## Shot 2: the before/after run (0:25 to 1:05)

**Visual:** clear terminal, then run:

```
python3 examples/batch_summarize.py
```

Let the output land. Mouse highlight the two `===` blocks side by side.

**Narration:**

> "Same twenty documents. Same fake Gemini backend with an 18% transient error rate. The top run is the notebook agent. Six summaries are missing. No retry. No audit. Now the bottom run, same task, through the production agent. Twenty out of twenty. Eight retries hidden inside the policy. Hard 5-cent budget cap. Cache hit on the duplicate document. Full JSONL audit log on disk. Three percent more compute, zero missing summaries."

## Shot 3: the governance map (1:05 to 1:30)

**Visual:** open `README.md`, scroll to the governance table. Briefly show `src/prod_gemini_agent/` with the eight Python files.

**Narration:**

> "Each row in this table is a small open-source library I have published. llmfleet, llm-retry, llm-circuit-breaker, llm-budget-window, cachebench, agenttrace. Each one solves one production problem. The project glues them around Gemini 2.0 Flash. Five lines of code switch from the fake provider to the real one. Fifteen lines switch to Vertex AI. Repo and Devpost write-up are linked below."

## Recording notes

- Use a clean terminal with a 14pt or larger monospaced font.
- The demo run takes well under one second; do not edit it for length, the speed is the point.
- For the README shot, hide the table of contents so the governance table is the first thing visible.
- Total target length: 75 to 90 seconds. Devpost cuts off long videos.
