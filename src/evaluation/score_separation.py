"""DEV only: top reranker score per question for the locked config, answerable vs unanswerable.

Shows whether the best reranked chunk's score separates questions the docs answer from ones
they don't. No threshold is chosen here.
"""

import json
import statistics
from pathlib import Path

from src.retrieval.pipeline import Resources, build_retrievers, load_config

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
SPLIT = "dev"


def main() -> None:
    config = load_config()
    with QUESTIONS.open(encoding="utf-8") as f:
        questions = [q for q in map(json.loads, f) if q["split"] == SPLIT]

    res = Resources(config)
    retriever = build_retrievers(config, res)["hybrid+bge-reranker"]
    act = getattr(res.reranker.model, "activation_fn", None)
    print(f"{config['chunker']['name']} + hybrid + {config['rerank']['model']}, split={SPLIT}; "
          f"score activation: {type(act).__name__ if act is not None else 'unknown'}")

    rows = []
    for q in questions:
        top = retriever.search(q["question"], k=1)[0]
        rows.append({"id": q["id"], "answerable": q["answerable"], "top_score": top["score"],
                     "top_page": top["doc_path"], "question": q["question"]})
    res.close()

    rows.sort(key=lambda r: -r["top_score"])
    print(f"\n  {'score':>7}  {'group':<12} {'id':<5} question")
    for r in rows:
        group = "answerable" if r["answerable"] else "UNANSWERABLE"
        print(f"  {r['top_score']:7.4f}  {group:<12} {r['id']:<5} {r['question'][:80]}")

    print()
    for label, flag in [("answerable", True), ("unanswerable", False)]:
        s = [r["top_score"] for r in rows if r["answerable"] is flag]
        print(f"  {label:<12} n={len(s):<2} min {min(s):.4f}  median {statistics.median(s):.4f}  max {max(s):.4f}")


if __name__ == "__main__":
    main()
