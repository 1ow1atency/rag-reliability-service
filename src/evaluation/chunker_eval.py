"""Compare chunkers (fixed_256, fixed_512, heading) on answerable dev questions with an evidence sentence.

For each chunker: dense, hybrid (RRF of dense + BM25) and hybrid+bge-reranker, each with its own
cached indexes. Chunk-level hit: a top-k chunk from the right page contains the whole evidence
sentence (whitespace ignored). Fixed windows can split a sentence across two chunks, so metrics are
reported on all evidence questions and on only those whose evidence is whole in that chunker's chunks.
"""

import json
import statistics
from pathlib import Path

from src.evaluation.retrieval_eval import (
    KS,
    TOP_N,
    clean_evidence,
    evidence_is_whole,
    first_evidence_rank,
    load_dev_answerable,
    load_evidence,
    summarize,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "metrics" / "chunker_comparison_dev.json"
CHUNKERS = ["fixed_256", "fixed_512", "heading"]
METHODS = ["dense", "hybrid", "hybrid+bge-reranker"]
RERANK_MODEL = "BAAI/bge-reranker-base"
TOP_TOKENS_K = 3


def collection_name(chunker: str) -> str:
    return f"chunks_{chunker}_bge_small"


def cells(s: dict) -> str:
    return "  ".join(f"{s[f'recall@{k}']['hits']:>2}/{s['n']:<2}" for k in KS) + f"  {s['mrr@10']:6.3f}"


def header() -> str:
    return "  ".join(f"{'R@' + str(k):>5}" for k in KS) + f"  {'MRR@10':>6}"


def main() -> None:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import MODEL_NAME, QDRANT_DIR, DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.rerank import CrossEncoderReranker, RerankRetriever

    evidence_raw = load_evidence()
    questions = [q for q in load_dev_answerable() if q["id"] in evidence_raw]
    evidence = {q["id"]: clean_evidence(evidence_raw[q["id"]]) for q in questions}
    n = len(questions)

    model = SentenceTransformer(MODEL_NAME)
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_DIR))
    reranker = CrossEncoderReranker(RERANK_MODEL)
    print(f"model.max_seq_length = {model.max_seq_length}; reranker {RERANK_MODEL} on {reranker.device}")

    results = {}
    for chunker in CHUNKERS:
        chunks_path = PROCESSED / f"chunks_{chunker}.jsonl"
        dense = DenseRetriever(chunks_path, collection_name(chunker), model=model, client=client)
        dense_built = dense.build()
        bm25 = BM25Retriever(chunks_path)
        bm25_built = bm25.build()
        hybrid = HybridRetriever(dense, bm25)
        reranked = RerankRetriever(hybrid, reranker)
        retrievers = {"dense": dense, "hybrid": hybrid, "hybrid+bge-reranker": reranked}
        print(f"\n[{chunker}] {len(bm25.chunks)} chunks; dense index {'built' if dense_built else 'cached'}, "
              f"bm25 index {'built' if bm25_built else 'cached'}")

        whole = {q["id"]: evidence_is_whole(evidence[q["id"]], q["source_file"], bm25.chunks) for q in questions}
        split_ids = [qid for qid, ok in whole.items() if not ok]
        print(f"  evidence whole in at least one chunk: {n - len(split_ids)}/{n}"
              + (f"; split across chunks: {', '.join(split_ids)}" if split_ids else ""))
        for qid in split_ids:
            print(f"    {qid}: {evidence[qid][:110]}{'...' if len(evidence[qid]) > 110 else ''}")

        per_question = {m: {} for m in METHODS}
        top_tokens = {m: [] for m in METHODS}
        for q in questions:
            for m, retriever in retrievers.items():
                retrieved = retriever.search(q["question"], k=TOP_N)
                per_question[m][q["id"]] = first_evidence_rank(q, evidence[q["id"]], retrieved)
                top_tokens[m].append(sum(c["n_tokens"] for c in retrieved[:TOP_TOKENS_K]))

        whole_ids = [q["id"] for q in questions if whole[q["id"]]]
        results[chunker] = {
            "n_chunks": len(bm25.chunks),
            "evidence_whole": len(whole_ids),
            "evidence_split_ids": split_ids,
            "rerank_timing": reranked.timing_summary(),
            "methods": {
                m: {
                    "chunk_level_all": summarize([per_question[m][q["id"]] for q in questions]),
                    "chunk_level_whole_only": summarize([per_question[m][qid] for qid in whole_ids]),
                    f"top{TOP_TOKENS_K}_tokens_per_question_mean": statistics.mean(top_tokens[m]),
                    f"top{TOP_TOKENS_K}_tokens_per_chunk_mean": statistics.mean(top_tokens[m]) / TOP_TOKENS_K,
                    "chunk_rank": per_question[m],
                }
                for m in METHODS
            },
        }
    client.close()
    reranker.save()

    w = max(len(f"{c} {m}") for c in CHUNKERS for m in METHODS)
    width = len(header())
    print(f"\nChunk-level comparison, {n} dev questions with evidence "
          f"(right: only questions whose evidence is whole in that chunker's chunks)")
    print(f"  {'':<{w}}  {'all ' + str(n):^{width}}  |  {'n':>2}  {'evidence whole only':^{width}}")
    print(f"  {'chunker / method':<{w}}  {header()}  |  {'':>2}  {header()}")
    for c in CHUNKERS:
        for m in METHODS:
            r = results[c]["methods"][m]
            print(f"  {c + ' ' + m:<{w}}  {cells(r['chunk_level_all'])}  |  "
                  f"{r['chunk_level_whole_only']['n']:>2}  {cells(r['chunk_level_whole_only'])}")

    print(f"\nText handed to the model: tokens in the top {TOP_TOKENS_K} chunks, mean per question")
    print(f"  {'chunker / method':<{w}}  {'total':>6}  {'per chunk':>9}")
    for c in CHUNKERS:
        for m in METHODS:
            r = results[c]["methods"][m]
            print(f"  {c + ' ' + m:<{w}}  {r[f'top{TOP_TOKENS_K}_tokens_per_question_mean']:6.0f}  "
                  f"{r[f'top{TOP_TOKENS_K}_tokens_per_chunk_mean']:9.0f}")

    print("\nReranking time per question (bge-reranker, top 20 hybrid chunks):")
    for c in CHUNKERS:
        t = results[c]["rerank_timing"]
        print(f"  {c:<10} mean {t['mean_s'] * 1000:6.0f} ms  median {t['median_s'] * 1000:6.0f} ms"
              f"  ({t['n_queries'] - t['from_cache']} computed this run, {t['from_cache']} from cache)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "split": "dev",
        "questions": "answerable dev questions with an evidence sentence",
        "n_questions": n,
        "question_ids": [q["id"] for q in questions],
        "dense_model": MODEL_NAME,
        "rerank_model": RERANK_MODEL,
        "hit_definition": "a top-k chunk from the question's page contains the whole evidence sentence (whitespace ignored)",
        "chunkers": results,
    }, indent=2) + "\n")
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
