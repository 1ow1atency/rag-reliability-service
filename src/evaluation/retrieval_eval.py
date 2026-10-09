"""Page-level retrieval eval of dense retrieval over heading chunks, on answerable dev questions.

A question is a hit at k when any of the top-k chunks comes from its source_file.
MRR uses the chunk rank of the first chunk from the right page within the top 10 (0 if absent).
"""

import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
CHUNKS = ROOT / "data" / "processed" / "chunks_heading.jsonl"
COLLECTION = "chunks_heading_bge_small"
OUT = ROOT / "results" / "metrics" / "retrieval_heading_dense_dev.json"
DOCS_MARKER = "content/en/docs/"
KS = [1, 3, 5, 10]
MISS_K = 5


def normalize_doc_path(path: str) -> str:
    """Map a page path to the form used by source_file: relative to docs/, POSIX separators."""
    p = path.strip().replace("\\", "/")
    if DOCS_MARKER in p:
        p = p.split(DOCS_MARKER, 1)[1]
    while p.startswith("./"):
        p = p[2:]
    return str(PurePosixPath(p.lstrip("/")))


def first_hit_rank(question: dict, retrieved: list[dict]) -> int | None:
    target = normalize_doc_path(question["source_file"])
    for rank, chunk in enumerate(retrieved, 1):
        if normalize_doc_path(chunk["doc_path"]) == target:
            return rank
    return None


def load_dev_answerable() -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r["split"] == "dev" and r["answerable"] is True]


def main() -> None:
    from src.retrieval.dense import BATCH_SIZE, MODEL_NAME, QUERY_PREFIX, DenseRetriever

    retriever = DenseRetriever(CHUNKS, COLLECTION)
    print(f"model.max_seq_length = {retriever.model.max_seq_length}")
    built = retriever.build()
    print(f"Index {COLLECTION}: {'built' if built else 'cached, not re-embedded'} "
          f"({retriever.client.count(COLLECTION, exact=True).count} chunks)")

    questions = load_dev_answerable()
    n = len(questions)
    per_question = []
    for q in questions:
        retrieved = retriever.search(q["question"], k=max(KS))
        rank = first_hit_rank(q, retrieved)
        per_question.append({
            "id": q["id"],
            "question": q["question"],
            "source_file": q["source_file"],
            "rank": rank,
            "top_pages": [normalize_doc_path(c["doc_path"]) for c in retrieved],
        })
    retriever.close()

    recall = {}
    for k in KS:
        hits = sum(1 for r in per_question if r["rank"] is not None and r["rank"] <= k)
        recall[f"recall@{k}"] = {"hits": hits, "total": n, "rate": hits / n}
    mrr = sum(1 / r["rank"] for r in per_question if r["rank"]) / n

    print(f"\nDev answerable questions: {n}")
    for name, v in recall.items():
        print(f"  {name:<10} {v['hits']:>2}/{v['total']}  {v['rate']:6.1%}")
    print(f"  {'MRR@10':<10} {mrr:.3f}")

    misses = [r for r in per_question if r["rank"] is None or r["rank"] > MISS_K]
    print(f"\nMisses (right page not in top {MISS_K}): {len(misses)}")
    for r in misses:
        rank = f"rank {r['rank']}" if r["rank"] else "not in top 10"
        print(f"\n  {r['id']} ({rank}) {r['question']}")
        print(f"    expected: {r['source_file']}")
        for i, page in enumerate(r["top_pages"][:3], 1):
            print(f"    top {i}:    {page}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "retriever": "dense",
        "model": MODEL_NAME,
        "query_prefix": QUERY_PREFIX,
        "batch_size": BATCH_SIZE,
        "chunks": str(CHUNKS.relative_to(ROOT)),
        "split": "dev",
        "answerable_only": True,
        "hit_definition": "any top-k chunk from the question's source_file",
        "n_questions": n,
        **recall,
        "mrr@10": mrr,
        "per_question": per_question,
    }, indent=2) + "\n")
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
