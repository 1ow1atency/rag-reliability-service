"""Sample answers and score their uncertainty, for closed-book or RAG mode.

Usage:
  python -m src.reliability.sample_entropy --mode {closed_book,rag} (--ids q001 q002 ... | --split {dev,test,all})

For each question: N_SAMPLES answers from the answer model at temperature 1.0 with the locked
prompts (src/generation/answer.py), clustered by meaning with deberta-large-mnli bidirectional
entailment, then semantic entropy, naive entropy and 1 - agreement rate.
RAG mode reuses each question's saved top-3 chunks from results/generation/<split>_rag.jsonl,
so retrieval is not re-run. One JSONL row is written after every question; on restart, questions
already done with the same settings are skipped.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.generation_eval import load_done as load_generation_rows
from src.evaluation.generation_eval import results_path as generation_results_path
from src.evaluation.generation_eval import run_version as generation_version
from src.generation import answer as answer_mod
from src.generation.groq_chat import IncompleteResponse, RateLimitExhausted, chat, complete_text
from src.reliability.entropy import agreement_rate, naive_entropy, semantic_entropy
from src.retrieval.pipeline import load_config

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
OUT_DIR = ROOT / "results" / "reliability"
N_SAMPLES = 5
MIN_SAMPLES = 4  # fewer valid samples than this: record as failed and retry on the next run
TEMPERATURE = 1.0
MAX_ATTEMPTS = 2  # per sample: one initial try plus one retry

logger = logging.getLogger(__name__)


def settings_version() -> str:
    """Changes when anything that affects the samples or scores changes."""
    from src.reliability.entailment import NLI_MODEL

    text = "\x00".join([answer_mod.prompt_version(), str(N_SAMPLES), str(TEMPERATURE), NLI_MODEL])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def out_path(mode: str) -> Path:
    return OUT_DIR / f"{mode}_samples.jsonl"


def load_done(path: Path, version: str) -> dict[str, dict]:
    """Latest completed row per question id for this settings version."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("version") == version and row.get("status") == "ok":
                done[row["id"]] = row
    return done


def load_chunk_texts() -> dict[str, dict]:
    chunks_file = ROOT / load_config()["chunker"]["chunks_file"]
    with chunks_file.open(encoding="utf-8") as f:
        return {c["chunk_id"]: c for c in map(json.loads, f)}


def saved_rag_chunks(question: dict, chunk_texts: dict[str, dict], saved: dict[str, dict[str, dict]]) -> list[dict]:
    """The top-3 chunks saved for this question by the generation run (no new retrieval)."""
    split = question["split"]
    if split not in saved:
        saved[split] = load_generation_rows(generation_results_path(split, "rag"), generation_version())
    row = saved[split].get(question["id"])
    if row is None:
        raise KeyError(f"no saved RAG retrieval for {question['id']} ({split}); run generation_eval first")
    return [chunk_texts[c["chunk_id"]] for c in row["retrieved"]]


def sample(mode: str, question: str, chunks: list[dict] | None) -> tuple[list[dict], list[str], dict]:
    """N_SAMPLES answers with token usage per call; a sample that fails twice is skipped.

    Also returns usage over every call made, failed ones included: {"calls", "tokens"}.
    """
    system_prompt, user_message = answer_mod.build_prompt(mode, question, chunks)
    samples, failures = [], []
    usage = {"calls": 0, "tokens": 0}
    for i in range(N_SAMPLES):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = chat(answer_mod.MODEL, system_prompt, user_message, TEMPERATURE)
                usage["calls"] += 1
                usage["tokens"] += result.total_tokens or 0
                text = complete_text(result)
            except RateLimitExhausted:
                raise
            except Exception as e:
                if attempt == MAX_ATTEMPTS:
                    failures.append(f"sample {i + 1}: {type(e).__name__}: {e}")
                    if isinstance(e, IncompleteResponse):
                        logger.warning("Sample %d failed twice (%s); skipping.", i + 1, e)
                continue
            samples.append({"text": text, "finish_reason": result.finish_reason,
                            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                            "total_tokens": result.total_tokens, "attempt": attempt})
            break
    return samples, failures, usage


def score(question: str, answers: list[str]) -> dict:
    from src.reliability.entailment import cluster_answers

    ids = cluster_answers(question, answers)
    return {"cluster_ids": ids, "n_clusters": len(set(ids)),
            "semantic_entropy": semantic_entropy(ids), "naive_entropy": naive_entropy(answers),
            "one_minus_agreement": 1 - agreement_rate(ids)}


def select_questions(ids: list[str] | None, split: str | None) -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if ids:
        by_id = {r["id"]: r for r in rows}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"unknown question ids: {missing}")
        return [by_id[i] for i in ids]
    return rows if split == "all" else [r for r in rows if r["split"] == split]


def main() -> None:
    load_dotenv()  # eval scripts may read GROQ_API_KEY from .env; the service does not
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=answer_mod.MODES, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", nargs="+")
    group.add_argument("--split", choices=["dev", "test", "all"])
    args = parser.parse_args()

    questions = select_questions(args.ids, args.split)
    version = settings_version()
    path = out_path(args.mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(path, version)
    todo = [q for q in questions if q["id"] not in done]
    print(f"mode={args.mode} questions={len(questions)} done={len(questions) - len(todo)} to run={len(todo)} "
          f"samples={N_SAMPLES} temperature={TEMPERATURE} version={version}")

    chunk_texts = load_chunk_texts() if args.mode == "rag" and todo else {}
    saved: dict[str, dict[str, dict]] = {}
    calls = tokens = 0
    with path.open("a", encoding="utf-8") as f:
        for q in todo:
            chunks = saved_rag_chunks(q, chunk_texts, saved) if args.mode == "rag" else None
            samples, failures, usage = sample(args.mode, q["question"], chunks)
            calls += usage["calls"]
            tokens += usage["tokens"]
            row = {"id": q["id"], "split": q["split"], "mode": args.mode, "version": version,
                   "question": q["question"], "gold": q["answer"], "answerable": q["answerable"],
                   "samples": samples, "failures": failures,
                   "retrieved_chunk_ids": [c["chunk_id"] for c in chunks] if chunks else None,
                   "usage": usage}
            if len(samples) < MIN_SAMPLES:
                row["status"] = "failed"
                print(f"  {q['id']} FAILED: only {len(samples)}/{N_SAMPLES} samples")
            else:
                row.update(score(q["question"], [s["text"] for s in samples]), status="ok")
                print(f"  {q['id']} samples={len(samples)} clusters={row['n_clusters']} "
                      f"SE={row['semantic_entropy']:.3f} naive={row['naive_entropy']:.3f} "
                      f"1-agree={row['one_minus_agreement']:.2f}  "
                      f"tokens/call={usage['tokens'] / max(usage['calls'], 1):,.0f} ({usage['calls']} calls)")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    if calls:
        print(f"\nThis run: {tokens:,} answer-model tokens over {calls} calls "
              f"({tokens / calls:,.0f} tokens per call, failed calls included)")


if __name__ == "__main__":
    main()
