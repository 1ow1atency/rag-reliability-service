"""Retrieval eval on answerable dev questions over heading chunks: dense, BM25, hybrid, and
hybrid reranked by cross-encoders (top 20 hybrid candidates reranked, top 10 kept).

Two hit measures, each ranked over the top 10 chunks:
  page-level:  a chunk from the question's source_file is retrieved.
  chunk-level: a chunk from that page contains the question's evidence sentence
               (from data/eval/candidates.jsonl, cleaned like the chunks, whitespace ignored).
Questions without an evidence sentence (the hand-written ones) get page-level only.
MRR@10 uses the rank of the first hit (0 if none in the top 10).
"""

import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
CANDIDATES = ROOT / "data" / "eval" / "candidates.jsonl"
CHUNKS = ROOT / "data" / "processed" / "chunks_heading.jsonl"
COLLECTION = "chunks_heading_bge_small"
OUT = ROOT / "results" / "metrics" / "retrieval_dev_comparison.json"
DOCS_MARKER = "content/en/docs/"
KS = [1, 3, 5, 10]
TOP_N = max(KS)
DISAGREE_K = 5
RERANKERS = {
    "hybrid+MiniLM": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "hybrid+bge-reranker": "BAAI/bge-reranker-base",
}


# ---------- matching ----------


def normalize_doc_path(path: str) -> str:
    """Map a page path to the form used by source_file: relative to docs/, POSIX separators."""
    p = path.strip().replace("\\", "/")
    if DOCS_MARKER in p:
        p = p.split(DOCS_MARKER, 1)[1]
    while p.startswith("./"):
        p = p[2:]
    return str(PurePosixPath(p.lstrip("/")))


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_evidence(evidence: str) -> str:
    """Clean a raw-markdown evidence sentence the way chunk text is cleaned, then normalize whitespace."""
    from src.ingest.chunking import clean_markdown

    return normalize_ws(clean_markdown(evidence)[1])


def first_hit_rank(question: dict, retrieved: list[dict]) -> int | None:
    """Rank of the first chunk from the question's page, or None."""
    target = normalize_doc_path(question["source_file"])
    for rank, chunk in enumerate(retrieved, 1):
        if normalize_doc_path(chunk["doc_path"]) == target:
            return rank
    return None


def first_evidence_rank(question: dict, evidence: str, retrieved: list[dict]) -> int | None:
    """Rank of the first chunk from the question's page whose text contains the cleaned evidence."""
    target = normalize_doc_path(question["source_file"])
    for rank, chunk in enumerate(retrieved, 1):
        if normalize_doc_path(chunk["doc_path"]) == target and evidence in normalize_ws(chunk["text"]):
            return rank
    return None


# ---------- metrics ----------


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"n": n}
    for k in KS:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        out[f"recall@{k}"] = {"hits": hits, "total": n, "rate": hits / n if n else 0.0}
    out["mrr@10"] = sum(1 / r for r in ranks if r) / n if n else 0.0
    return out


def load_dev_answerable() -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r["split"] == "dev" and r["answerable"] is True]


def load_evidence() -> dict[str, str]:
    with CANDIDATES.open(encoding="utf-8") as f:
        return {r["id"]: r["evidence"] for r in map(json.loads, f) if r.get("evidence")}


# ---------- reporting ----------


def fmt_cells(s: dict) -> str:
    cells = [f"{s[f'recall@{k}']['hits']}/{s['n']}" for k in KS]
    return "  ".join(f"{c:>6}" for c in cells) + f"  {s['mrr@10']:>6.3f}"


def header_cells() -> str:
    return "  ".join(f"{'R@' + str(k):>6}" for k in KS) + f"  {'MRR@10':>6}"


def print_table(title: str, rows: list[tuple[str, dict]], w: int) -> None:
    print(f"\n{title}")
    print(f"  {'method':<{w}} {header_cells()}")
    for label, s in rows:
        print(f"  {label:<{w}} {fmt_cells(s)}")


def print_side_by_side(title: str, rows: list[tuple[str, dict, dict]], w: int) -> None:
    width = len(header_cells())
    print(f"\n{title}")
    print(f"  {'':<{w}} {'page-level':^{width}}  |  {'chunk-level':^{width}}")
    print(f"  {'method':<{w}} {header_cells()}  |  {header_cells()}")
    for label, page, chunk in rows:
        print(f"  {label:<{w}} {fmt_cells(page)}  |  {fmt_cells(chunk)}")


def rank_str(r: int | None) -> str:
    return str(r) if r is not None else "-"


def main() -> None:
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import MODEL_NAME, DenseRetriever
    from src.retrieval.hybrid import DEPTH, RRF_K, HybridRetriever
    from src.retrieval.rerank import DEPTH as RERANK_DEPTH
    from src.retrieval.rerank import CrossEncoderReranker, RerankRetriever

    dense = DenseRetriever(CHUNKS, COLLECTION)
    print(f"dense index: {'built' if dense.build() else 'cached'}")
    bm25 = BM25Retriever(CHUNKS)
    print(f"bm25 index:  {'built' if bm25.build() else 'cached'}")
    hybrid = HybridRetriever(dense, bm25)
    retrievers = {"dense": dense, "bm25": bm25, "hybrid": hybrid}

    rerank_info, rerankers = {}, {}
    for label, model_name in RERANKERS.items():
        try:
            reranker = CrossEncoderReranker(model_name)
        except Exception as e:  # download or load failure: report and skip this model
            print(f"{label}: SKIPPED, could not load {model_name}: {type(e).__name__}: {e}")
            rerank_info[label] = {"model": model_name, "loaded": False, "error": f"{type(e).__name__}: {e}"}
            continue
        print(f"{label}: loaded {model_name} on {reranker.device}")
        rerankers[label] = reranker
        retrievers[label] = RerankRetriever(hybrid, reranker)
        rerank_info[label] = {"model": model_name, "loaded": True, "device": reranker.device}
    methods = list(retrievers)
    w = max(len(m) for m in methods)

    questions = load_dev_answerable()
    evidence_raw = load_evidence()
    evidence = {q["id"]: clean_evidence(evidence_raw[q["id"]]) for q in questions if q["id"] in evidence_raw}

    # Sanity check: an evidence sentence split across chunk boundaries can never be a chunk-level hit.
    source = {q["id"]: normalize_doc_path(q["source_file"]) for q in questions}
    unreachable = [
        qid for qid, ev in evidence.items()
        if not any(ev in normalize_ws(c["text"]) for c in bm25.chunks
                   if normalize_doc_path(c["doc_path"]) == source[qid])
    ]

    per_question = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "source_file": q["source_file"],
               "has_evidence": q["id"] in evidence, "page_rank": {}, "chunk_rank": {}, "top_pages": {}}
        for name, retriever in retrievers.items():
            retrieved = retriever.search(q["question"], k=TOP_N)
            row["page_rank"][name] = first_hit_rank(q, retrieved)
            row["chunk_rank"][name] = (first_evidence_rank(q, evidence[q["id"]], retrieved)
                                       if q["id"] in evidence else None)
            row["top_pages"][name] = [normalize_doc_path(c["doc_path"]) for c in retrieved]
        per_question.append(row)
    dense.close()
    for reranker in rerankers.values():
        reranker.save()

    with_ev = [r for r in per_question if r["has_evidence"]]
    page_only = [r for r in per_question if not r["has_evidence"]]
    metrics = {
        name: {
            "page_level_all": summarize([r["page_rank"][name] for r in per_question]),
            "page_level_with_evidence": summarize([r["page_rank"][name] for r in with_ev]),
            "chunk_level_with_evidence": summarize([r["chunk_rank"][name] for r in with_ev]),
        }
        for name in methods
    }

    print(f"\nDev answerable questions: {len(per_question)} "
          f"({len(with_ev)} with evidence, {len(page_only)} page-level only)")
    print_table(f"Page-level, all {len(per_question)} dev questions",
                [(m, metrics[m]["page_level_all"]) for m in methods], w)
    print_side_by_side(
        f"Page-level vs chunk-level, {len(with_ev)} questions with an evidence sentence",
        [(m, metrics[m]["page_level_with_evidence"], metrics[m]["chunk_level_with_evidence"]) for m in methods],
        w,
    )
    if unreachable:
        print(f"  note: evidence for {unreachable} is split across chunks, so chunk-level can't hit")

    print(f"\nPage-level only (no evidence sentence), rank of right page per method:")
    for r in page_only:
        ranks = "  ".join(f"{m}={rank_str(r['page_rank'][m])}" for m in methods)
        print(f"  {r['id']}  {ranks}   {r['source_file']}")

    disagreements = [
        r for r in per_question
        if len({r["page_rank"][m] is not None and r["page_rank"][m] <= DISAGREE_K for m in methods}) > 1
    ]
    print(f"\nDisagreements (right page in top {DISAGREE_K} for some methods but not others): {len(disagreements)}")
    for r in disagreements:
        print(f"\n  {r['id']} {r['question']}")
        print(f"    expected: {r['source_file']}")
        for m in methods:
            found = r["page_rank"][m] is not None and r["page_rank"][m] <= DISAGREE_K
            print(f"    {m:<{w}} page rank {rank_str(r['page_rank'][m]):>2}  {'found' if found else 'MISSED'}"
                  f"   top 1: {r['top_pages'][m][0]}")

    if rerankers:
        labels = list(rerankers)
        print(f"\nChunk-level rank per question, hybrid vs rerankers (- = not in top {TOP_N}):")
        print(f"  {'id':<5} {'hybrid':>6}  " + "  ".join(f"{l:>{len(l)}}" for l in labels))
        changes = {l: {"improved": 0, "worse": 0, "same": 0} for l in labels}
        for r in with_ev:
            base = r["chunk_rank"]["hybrid"]
            cells = []
            for l in labels:
                new = r["chunk_rank"][l]
                b, n = base or TOP_N + 1, new or TOP_N + 1
                tag = "better" if n < b else "worse" if n > b else "same"
                changes[l]["improved" if tag == "better" else tag] += 1
                cells.append(f"{rank_str(new) + ' ' + ('+' if tag == 'better' else '-' if tag == 'worse' else '='):>{len(l)}}")
            print(f"  {r['id']:<5} {rank_str(base):>6}  " + "  ".join(cells))
        for l, c in changes.items():
            print(f"  {l}: {c['improved']} improved, {c['worse']} worse, {c['same']} unchanged")
            rerank_info[l]["chunk_rank_changes_vs_hybrid"] = c

        print("\nReranking time per question (cross-encoder scoring of the top "
              f"{RERANK_DEPTH} hybrid chunks, excluding retrieval):")
        for l in labels:
            t = retrievers[l].timing_summary()
            rerank_info[l]["timing"] = t
            src = (f"all {t['n_queries']} from the score cache (times recorded when first computed)"
                   if t["from_cache"] == t["n_queries"] else f"{t['n_queries'] - t['from_cache']} computed this run")
            print(f"  {l:<{w}} mean {t['mean_s'] * 1000:7.1f} ms  median {t['median_s'] * 1000:7.1f} ms  "
                  f"max {t['max_s'] * 1000:7.1f} ms  on {rerank_info[l]['device']}  ({src})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "split": "dev",
        "answerable_only": True,
        "chunks": str(CHUNKS.relative_to(ROOT)),
        "dense_model": MODEL_NAME,
        "hybrid": {"method": "rrf", "rrf_k": RRF_K, "depth_per_retriever": DEPTH},
        "rerank": {"base": "hybrid", "depth": RERANK_DEPTH, "top_k_kept": TOP_N, "models": rerank_info},
        "hit_definitions": {
            "page_level": "a top-k chunk comes from the question's source_file",
            "chunk_level": "a top-k chunk from that page contains the evidence sentence (whitespace ignored)",
        },
        "n_questions": len(per_question),
        "n_with_evidence": len(with_ev),
        "page_level_only_ids": [r["id"] for r in page_only],
        "evidence_split_across_chunks": unreachable,
        "metrics": metrics,
        "disagreement_ids": [r["id"] for r in disagreements],
        "per_question": per_question,
    }, indent=2) + "\n")
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
