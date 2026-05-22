"""90-second demo: same task, two scenes.

Before: 20 raw Gemini calls. Some fail, no audit, no cost cap.
After:  20 calls through ProductionAgent with retry, breaker, budget,
        cache, fleet, trace.

By default this uses ``FakeGeminiProvider`` with seed=7 so the demo
is byte-stable. Set ``GEMINI_API_KEY`` in the environment to swap in
the real Gemini 2.0 Flash endpoint.

Run:
    python examples/batch_summarize.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the demo importable without a pip install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from prod_gemini_agent import (  # noqa: E402
    BudgetWindow,
    CircuitBreaker,
    FakeGeminiProvider,
    Fleet,
    ProductionAgent,
    ResponseCache,
    RetryPolicy,
    RunTrace,
    run_raw_gemini_baseline,
)


# Public-domain document openings. Project Gutenberg first paragraphs.
DOCS: list[tuple[str, str]] = [
    ("pride_and_prejudice", "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."),
    ("moby_dick", "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, I thought I would sail about a little."),
    ("a_tale_of_two_cities", "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness."),
    ("anna_karenina", "Happy families are all alike; every unhappy family is unhappy in its own way."),
    ("alice_in_wonderland", "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do."),
    ("dracula", "3 May. Bistritz. Left Munich at 8:35 P.M., on 1st May, arriving at Vienna early next morning."),
    ("frankenstein", "You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have regarded with such evil forebodings."),
    ("the_metamorphosis", "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect."),
    ("don_quixote", "In a village of La Mancha, the name of which I have no desire to call to mind, there lived not long since one of those gentlemen."),
    ("the_great_gatsby", "In my younger and more vulnerable years my father gave me some advice that I have been turning over in my mind ever since."),
    ("crime_and_punishment", "On an exceptionally hot evening early in July a young man came out of the garret in which he lodged in S. Place."),
    ("the_picture_of_dorian_gray", "The studio was filled with the rich odour of roses, and when the light summer wind stirred amidst the trees of the garden."),
    ("ulysses", "Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed."),
    ("brave_new_world", "A squat grey building of only thirty-four stories. Over the main entrance the words, CENTRAL LONDON HATCHERY AND CONDITIONING CENTRE."),
    ("nineteen_eighty_four", "It was a bright cold day in April, and the clocks were striking thirteen."),
    ("the_odyssey", "Tell me, O muse, of that ingenious hero who travelled far and wide after he had sacked the famous town of Troy."),
    ("hamlet", "Who's there? Nay, answer me. Stand and unfold yourself. Long live the king!"),
    ("the_iliad", "Sing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans."),
    ("walden", "When I wrote the following pages, or rather the bulk of them, I lived alone, in the woods, a mile from any neighbour."),
    # Last item is a duplicate so the cache has something interesting to show.
    ("pride_and_prejudice_again", "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."),
]


def _build_prompts() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for doc_id, opening in DOCS:
        prompt = (
            "You are a careful editor. Summarize the opening below in one sentence "
            "of plain English. Keep the tone of the original.\n\n"
            f"Opening: {opening}"
        )
        out.append((doc_id, prompt))
    return out


def _select_provider():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        from prod_gemini_agent import GeminiClient
        print("\n[provider] GEMINI_API_KEY set, calling real Gemini 2.0 Flash.")
        return GeminiClient(api_key=api_key)
    print("\n[provider] No GEMINI_API_KEY found. Using FakeGeminiProvider(seed=7).")
    return FakeGeminiProvider(seed=7, error_rate=0.18, burst_failure_after=6)


def main() -> int:
    prompts = _build_prompts()
    print(f"Documents in batch: {len(prompts)}")

    # ---- Scene 1: notebook-grade, no governance ---------------------------
    baseline_provider = _select_provider()
    print("\n[scene 1] Raw Gemini, no retry, no breaker, no budget, no cache.")
    baseline_report = run_raw_gemini_baseline(prompts, baseline_provider, max_workers=8)
    baseline_report.print()

    # ---- Scene 2: ProductionAgent, full governance ------------------------
    governed_provider = _select_provider()
    agent = ProductionAgent(
        provider=governed_provider,
        fleet=Fleet(max_workers=8),
        # 5 attempts gives the retry layer enough room to ride through both
        # the 18% rate-limit error rate and the synthetic three-error burst.
        retry_policy=RetryPolicy(max_attempts=5, base_delay_s=0.02),
        breaker=CircuitBreaker(failure_threshold=4, cooldown_s=0.3),
        budget=BudgetWindow(cap_usd=0.05, window_s=60.0),
        cache=ResponseCache(maxsize=256),
        trace=RunTrace(),
    )
    print("\n[scene 2] ProductionAgent with retry + breaker + budget + cache + trace.")
    governed_report = agent.run(prompts, label="ProductionAgent (governed)")
    governed_report.print()

    audit_path = Path(__file__).resolve().parent.parent / "audit.jsonl"
    agent.write_audit_log(audit_path)
    print(f"\n[audit] Wrote per-call audit log to {audit_path}")

    # Headline delta the README points at.
    print("\n=== headline ===")
    print(f"  before: {baseline_report.failed} failed call(s), no audit, no cost cap.")
    print(f"  after : {governed_report.failed} failed call(s), audit log on disk, "
          f"${governed_report.budget_remaining_usd or 0:.6f} USD budget left.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
