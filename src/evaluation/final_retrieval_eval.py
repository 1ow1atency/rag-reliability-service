"""Evaluate the locked retrieval config (configs/final_retrieval.json) and two reference setups.

Usage: python -m src.evaluation.final_retrieval_eval --split {dev,test}

Setups: the final config (fixed_512 + hybrid + bge-reranker), fixed_512 dense, and
heading + hybrid + bge-reranker. Answerable questions only. Page-level metrics on all of
them; page vs chunk-level side by side for questions with an evidence sentence; questions
without one are listed separately (page-level only). The final config hands its top
config["top_k"] chunks to the model, so R@3 is its operative number; deeper columns are
shown for comparison. Only the test run is saved.
"""

import argparse
import json
from pathlib import Path

from src.evaluation.retrieval_eval import (
    TOP_N,
    clean_evidence,
    evidence_is_whole,
    first_evidence_rank,
    first_hit_rank,
    load_evidence,
    print_side_by_side,
    print_table,
    rank_str,
    summarize,
)
from src.retrieval.pipeline import CONFIG, Resources, build_retrievers, load_config

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
OUT = ROOT / "results" / "metrics" / "retrieval_test_final.json"


def setups(config: dict) -> list[tuple[str, str, str]]:
    """(label, chunker, method) for each evaluated setup; the final config comes first."""
    final = config["chunker"]["name"]
    return [
        (f"final ({final} hybrid+bge)", final, "hybrid+bge-reranker"),
        (f"{final} dense", final, "dense"),
        ("heading hybrid+bge", "heading", "hybrid+bge-reranker"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    split = parser.parse_args().split

    config = load_config()
    with QUESTIONS.open(encoding="utf-8") as f:
        questions = [q for q in map(json.loads, f) if q["split"] == split and q["answerable"] is True]
    evidence_raw = load_evidence()
    evidence = {q["id"]: clean_evidence(evidence_raw[q["id"]]) for q in questions if q["id"] in evidence_raw}

    res = Resources(config)
    by_chunker = {c: build_retrievers(config, res, c) for c in {c for _, c, _ in setups(config)}}
    labels = [label for label, _, _ in setups(config)]

    split_evidence = {
        c: [qid for qid in evidence
            if not evidence_is_whole(evidence[qid], next(q for q in questions if q["id"] == qid)["source_file"],
                                     r["bm25"].chunks)]
        for c, r in by_chunker.items()
    }

    per_question = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "source_file": q["source_file"],
               "has_evidence": q["id"] in evidence, "page_rank": {}, "chunk_rank": {}, "top3_tokens": {}}
        for label, chunker, method in setups(config):
            retrieved = by_chunker[chunker][method].search(q["question"], k=TOP_N)
            row["page_rank"][label] = first_hit_rank(q, retrieved)
            row["chunk_rank"][label] = (first_evidence_rank(q, evidence[q["id"]], retrieved)
                                        if q["id"] in evidence else None)
            row["top3_tokens"][label] = sum(c["n_tokens"] for c in retrieved[:3])
        per_question.append(row)
    res.close()

    with_ev = [r for r in per_question if r["has_evidence"]]
    page_only = [r for r in per_question if not r["has_evidence"]]
    metrics = {
        label: {
            "page_level_all": summarize([r["page_rank"][label] for r in per_question]),
            "page_level_with_evidence": summarize([r["page_rank"][label] for r in with_ev]),
            "chunk_level_with_evidence": summarize([r["chunk_rank"][label] for r in with_ev]),
            "top3_tokens_per_question_mean": sum(r["top3_tokens"][label] for r in per_question) / len(per_question),
        }
        for label in labels
    }

    w = max(len(l) for l in labels)
    print(f"\nSplit: {split}. Answerable questions: {len(per_question)} "
          f"({len(with_ev)} with evidence, {len(page_only)} page-level only). "
          f"Final config returns top {config['top_k']}.")
    for c, ids in split_evidence.items():
        print(f"  {c}: evidence whole in some chunk for {len(evidence) - len(ids)}/{len(evidence)}"
              + (f"; split: {', '.join(ids)}" if ids else ""))
    print_table(f"Page-level, all {len(per_question)} answerable {split} questions",
                [(l, metrics[l]["page_level_all"]) for l in labels], w)
    print_side_by_side(f"Page-level vs chunk-level, {len(with_ev)} questions with an evidence sentence",
                       [(l, metrics[l]["page_level_with_evidence"], metrics[l]["chunk_level_with_evidence"])
                        for l in labels], w)
    print("\nPage-level only (no evidence sentence), rank of right page:")
    for r in page_only:
        ranks = "  ".join(f"{l}={rank_str(r['page_rank'][l])}" for l in labels)
        print(f"  {r['id']}  {ranks}   {r['source_file']}")
    print("\nTokens in the top 3 chunks, mean per question:")
    for l in labels:
        print(f"  {l:<{w}}  {metrics[l]['top3_tokens_per_question_mean']:6.0f}")

    if split != "test":
        print("\n(dev check only; nothing saved)")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "split": "test",
        "answerable_only": True,
        "config": str(CONFIG.relative_to(ROOT)),
        "setups": [{"label": l, "chunker": c, "method": m} for l, c, m in setups(config)],
        "hit_definitions": {
            "page_level": "a top-k chunk comes from the question's source_file",
            "chunk_level": "a top-k chunk from that page contains the whole evidence sentence (whitespace ignored)",
        },
        "n_questions": len(per_question),
        "n_with_evidence": len(with_ev),
        "page_level_only_ids": [r["id"] for r in page_only],
        "evidence_split_across_chunks": split_evidence,
        "metrics": metrics,
        "per_question": per_question,
    }, indent=2) + "\n")
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
