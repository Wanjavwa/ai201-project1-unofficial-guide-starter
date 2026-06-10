"""Evaluation harness.

Runs a fixed set of test questions (with ground-truth answers) through the full
RAG pipeline and prints a report showing, for each question:
  - the question and the expected answer
  - what the system returned
  - which chunks were retrieved (source + distance)

You then judge retrieval quality and response accuracy by hand and fill in the
Evaluation Report table in README.md. The harness also writes the raw run to
eval_results.md so you can paste from it.

Run:  python evaluate.py
"""

from pathlib import Path

import config
import generate

# 5 test questions with ground-truth answers grounded in the documents/ corpus.
# Q5 is intentionally hard: it requires combining facts from THREE documents
# (which dorms have AC, the per-semester room costs, and meal-plan eligibility)
# plus arithmetic. It exists to surface a failure, per the project hints.
TEST_QUESTIONS = [
    {
        "question": "Is the MSU housing lottery actually random?",
        "expected": (
            "No. The selection time ('time ticket') is weighted by earned credit "
            "hours, so juniors/seniors pick before sophomores. Randomness only "
            "applies as a tiebreaker between students with the same credits. "
            "Honors students get an earlier priority window."
        ),
    },
    {
        "question": "Which dorm does not have air conditioning?",
        "expected": (
            "North Tower has no air conditioning; students are advised to bring a "
            "fan. Riverside and Westfield both have central AC."
        ),
    },
    {
        "question": "How much does laundry cost in North Tower?",
        "expected": (
            "$1.50 per wash plus $1.50 per dry, charged to your student card "
            "(no cash/coins). Laundry is in the basement."
        ),
    },
    {
        "question": "Can a first-year living in Riverside choose a smaller meal "
                    "plan instead of the unlimited one?",
        "expected": (
            "Yes. Because Riverside has a shared kitchen on every floor, Riverside "
            "first-years are allowed to pick the smaller Block 80 plan instead of "
            "the otherwise-required Unlimited plan. North Tower and Westfield "
            "first-years cannot."
        ),
    },
    {
        # Deliberately hard cross-document question designed to surface a failure.
        # The North Tower room rate (~$3,200/sem) lives in cost_comparison.txt, but
        # the off-campus terms in the query ("Maple Court", "off-campus") pull
        # retrieval toward the off-campus docs and the cost doc's off-campus chunk,
        # so the per-dorm-rates chunk often doesn't make the top-k.
        "question": "Over a full academic year, is it cheaper to live off-campus at "
                    "Maple Court or in North Tower? Give a total for each.",
        "expected": (
            "North Tower: ~$3,200/sem room (~$6,400/yr) plus the required first-year "
            "meal plan ~$2,750/sem (~$5,500/yr) ~= ~$11,900/yr. Maple Court: "
            "$650/mo x 12 (12-month lease) = $7,800/yr in rent, PLUS unlisted "
            "utilities and food (no meal plan). The corpus lacks utility/food "
            "figures, so a clean head-to-head isn't possible — but North Tower's "
            "room cost and the meal-plan requirement ARE in the documents, so the "
            "system should surface those, not claim the cost is unknown."
        ),
    },
]


def _format_hits(hits: list[dict]) -> str:
    if not hits:
        return "    (no chunks passed the relevance threshold)"
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"    [{i}] {h['source']} "
                     f"(chunk {h['chunk_index']}, distance {h['distance']})")
    return "\n".join(lines)


def main() -> None:
    report_lines = ["# Evaluation Run\n",
                    f"_Model: {config.GROQ_MODEL} · Embeddings: "
                    f"{config.EMBEDDING_MODEL} · top_k={config.TOP_K}_\n"]

    for i, item in enumerate(TEST_QUESTIONS, 1):
        print("\n" + "#" * 72)
        print(f"Q{i}: {item['question']}")
        print("#" * 72)

        result = generate.answer_query(item["question"])

        print(f"\nEXPECTED:\n  {item['expected']}\n")
        print(f"SYSTEM ANSWER:\n  {result['answer']}\n")
        print("RETRIEVED CHUNKS:")
        print(_format_hits(result["hits"]))

        # Accumulate markdown for eval_results.md
        report_lines += [
            f"\n## Q{i}: {item['question']}\n",
            f"**Expected:** {item['expected']}\n",
            f"**System answer:**\n\n> {result['answer']}\n",
            "**Retrieved chunks:**\n",
        ]
        if result["hits"]:
            for j, h in enumerate(result["hits"], 1):
                report_lines.append(
                    f"{j}. `{h['source']}` (chunk {h['chunk_index']}, "
                    f"distance {h['distance']})")
        else:
            report_lines.append("_(no chunks passed the relevance threshold)_")
        report_lines.append("")

    out = Path(config.ROOT) / "eval_results.md"
    out.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n\nRaw run written to {out}")
    print("Now judge each result and fill the Evaluation Report table in README.md.")


if __name__ == "__main__":
    main()
