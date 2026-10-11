"""Closed-book vs RAG answers on a split, graded by the judge.

Usage: python -m src.evaluation.generation_eval --split {dev,test}

Each answered and judged question is appended to results/generation/<split>_<mode>.jsonl right
away. On restart, questions already done with the current prompt version are skipped; failed
ones are retried. The test split only runs when the prompt and config files have no uncommitted
changes, so test results always come from committed (locked) prompts.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from src.evaluation import judge as judge_mod
from src.evaluation.retrieval_eval import clean_evidence, first_evidence_rank, first_hit_rank, load_evidence
from src.generation import answer as answer_mod
from src.retrieval.pipeline import CONFIG, load_config

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "results" / "generation"
METRICS_OUT = ROOT / "results" / "metrics" / "generation_test.json"
LOCKED_FILES = ["src/generation/answer.py", "src/evaluation/judge.py", "configs/final_retrieval.json"]
ENTROPY_SAMPLES = 5
FAILED = "FAILED"


def run_version() -> str:
    """Changes whenever the answer prompts, judge prompts or retrieval config change."""
    text = "\x00".join([answer_mod.prompt_version(), judge_mod.prompt_version(), CONFIG.read_text()])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def results_path(split: str, mode: str) -> Path:
    return RESULTS_DIR / f"{split}_{mode}.jsonl"


def load_done(path: Path, version: str) -> dict[str, dict]:
    """Latest row per question id for this version, skipping rows whose answer or judging failed."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("version") == version and row.get("label") != FAILED:
                done[row["id"]] = row
    return done


def locked_files_dirty() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--", *LOCKED_FILES],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line[3:] for line in out.splitlines()]


def run_mode(split: str, mode: str, questions: list[dict], evidence: dict, version: str, retriever) -> list[dict]:
    path = results_path(split, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(path, version)
    todo = [q for q in questions if q["id"] not in done]
    print(f"[{split} {mode}] {len(done)} done, {len(todo)} to run")

    with path.open("a", encoding="utf-8") as f:
        for q in todo:
            row = {"id": q["id"], "split": split, "mode": mode, "version": version,
                   "question": q["question"], "gold": q["answer"], "answerable": q["answerable"],
                   "source_file": q["source_file"]}
            chunks = None
            if mode == "rag":
                chunks = retriever.search(q["question"], k=load_config()["top_k"])
                row["retrieved"] = [{"chunk_id": c["chunk_id"], "doc_path": c["doc_path"],
                                     "score": c["score"], "n_tokens": c["n_tokens"]} for c in chunks]
                if q["answerable"]:
                    row["page_in_top3"] = first_hit_rank(q, chunks) is not None
                    row["evidence_in_top3"] = (first_evidence_rank(q, evidence[q["id"]], chunks) is not None
                                               if q["id"] in evidence else None)
            ans = answer_mod.answer(mode, q["question"], chunks)
            row["answer"] = ans.text
            row["answer_usage"] = ans.to_dict()
            if ans.text is None:
                row["label"], row["judge"] = FAILED, None
            else:
                verdict = judge_mod.judge(q["question"], ans.text, q["answer"], q["answerable"])
                row["label"] = verdict.label or FAILED
                row["judge"] = verdict.to_dict()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[q["id"]] = row
            print(f"  {q['id']} {row['label']:<13} {(ans.text or '').replace(chr(10), ' ')[:90]}")
    return [done[q["id"]] for q in questions if q["id"] in done]


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "0/0"


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    out = {
        "answerable": {lab: sum(r["label"] == lab for r in ans) for lab in judge_mod.ANSWERABLE_LABELS},
        "unanswerable": {lab: sum(r["label"] == lab for r in una) for lab in judge_mod.UNANSWERABLE_LABELS},
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
    }
    if any("evidence_in_top3" in r for r in ans):
        groups = {"evidence_in_top3": [r for r in ans if r.get("evidence_in_top3") is True],
                  "evidence_not_in_top3": [r for r in ans if r.get("evidence_in_top3") is False],
                  "no_evidence_sentence": [r for r in ans if r.get("evidence_in_top3") is None]}
        out["answerable_by_retrieval"] = {
            g: {"n": len(rs), **{lab: sum(r["label"] == lab for r in rs) for lab in judge_mod.ANSWERABLE_LABELS}}
            for g, rs in groups.items()
        }
    return out


def token_stats(rows: list[dict]) -> dict:
    a_calls = sum(r["answer_usage"]["calls"] for r in rows)
    a_tokens = sum(r["answer_usage"]["total_tokens"] for r in rows)
    j_calls = sum(r["judge"]["calls"] for r in rows if r["judge"])
    j_tokens = sum(r["judge"]["total_tokens"] for r in rows if r["judge"])
    per_q = [r["answer_usage"]["total_tokens"] for r in rows]
    return {
        "answer_calls": a_calls, "answer_tokens": a_tokens,
        "answer_tokens_per_call": a_tokens / a_calls if a_calls else 0,
        "answer_prompt_tokens_per_question": statistics.mean(r["answer_usage"]["prompt_tokens"] for r in rows),
        "answer_completion_tokens_per_question": statistics.mean(r["answer_usage"]["completion_tokens"] for r in rows),
        "answer_tokens_per_question": statistics.mean(per_q),
        "judge_calls": j_calls, "judge_tokens": j_tokens,
        "judge_tokens_per_call": j_tokens / j_calls if j_calls else 0,
    }


def print_report(split: str, results: dict[str, list[dict]]) -> dict:
    report = {m: {"labels": summarize(rows), "tokens": token_stats(rows)} for m, rows in results.items()}
    print(f"\n=== {split}: closed-book vs RAG ===")
    for m, rep in report.items():
        s = rep["labels"]
        a, u = s["answerable"], s["unanswerable"]
        print(f"\n{m}")
        print(f"  answerable   ({s['n_answerable']}): correct {pct(a['CORRECT'], s['n_answerable'])}, "
              f"incorrect {pct(a['INCORRECT'], s['n_answerable'])}, not attempted {pct(a['NOT_ATTEMPTED'], s['n_answerable'])}")
        print(f"  unanswerable ({s['n_unanswerable']}): abstained {pct(u['ABSTAINED'], s['n_unanswerable'])}, "
              f"answered {pct(u['ANSWERED'], s['n_unanswerable'])}")
        failed = sum(r["label"] == FAILED for r in results[m])
        if failed:
            print(f"  FAILED (no answer or no verdict after retries): {failed}")
        for g, v in s.get("answerable_by_retrieval", {}).items():
            print(f"  answerable, {g:<21} n={v['n']:<2} correct {pct(v['CORRECT'], v['n'])}, "
                  f"incorrect {v['INCORRECT']}, not attempted {v['NOT_ATTEMPTED']}")

    n_all = sum(1 for _ in QUESTIONS.open())
    print("\nTokens (answer model; judge listed separately)")
    for m, rep in report.items():
        t = rep["tokens"]
        est = ENTROPY_SAMPLES * n_all * t["answer_tokens_per_question"]
        rep["entropy_estimate"] = {"samples_per_question": ENTROPY_SAMPLES, "questions": n_all,
                                   "answer_tokens": round(est)}
        print(f"  {m:<11} answer: {t['answer_tokens']:>7,} tokens over {t['answer_calls']} calls "
              f"({t['answer_tokens_per_call']:,.0f}/call; per question {t['answer_prompt_tokens_per_question']:,.0f} prompt "
              f"+ {t['answer_completion_tokens_per_question']:,.0f} completion)")
        print(f"  {'':<11} judge:  {t['judge_tokens']:>7,} tokens over {t['judge_calls']} calls "
              f"({t['judge_tokens_per_call']:,.0f}/call)")
        print(f"  {'':<11} entropy estimate: {ENTROPY_SAMPLES} samples x {n_all} questions x "
              f"{t['answer_tokens_per_question']:,.0f} = ~{est:,.0f} answer-model tokens")
    return report


def print_problems(results: dict[str, list[dict]]) -> None:
    bad = [r for rows in results.values() for r in rows
           if r["label"] in ("INCORRECT", FAILED) or (not r["answerable"] and r["label"] == "ANSWERED")]
    print(f"\nINCORRECT, FAILED, or ANSWERED-on-unanswerable: {len(bad)}")
    for r in bad:
        print(f"\n  [{r['mode']}] {r['id']} {r['label']}  {r['question']}")
        print(f"    gold:   {r['gold']}")
        print(f"    answer: {(r['answer'] or '').strip()[:300]}")
        if r["mode"] == "rag":
            print(f"    top3:   {[c['doc_path'] for c in r['retrieved']]}"
                  + (f"  evidence_in_top3={r.get('evidence_in_top3')}" if r["answerable"] else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    split = parser.parse_args().split

    if split == "test" and (dirty := locked_files_dirty()):
        raise SystemExit(f"Refusing to run on test: uncommitted changes in {dirty}")

    with QUESTIONS.open(encoding="utf-8") as f:
        questions = [q for q in map(json.loads, f) if q["split"] == split]
    evidence_raw = load_evidence()
    evidence = {q["id"]: clean_evidence(evidence_raw[q["id"]]) for q in questions if q["id"] in evidence_raw}
    version = run_version()
    print(f"split={split}  questions={len(questions)}  version={version}  "
          f"answer model={answer_mod.MODEL}  judge={judge_mod.JUDGE_MODEL}")

    retriever, res = None, None
    if any(q["id"] not in load_done(results_path(split, "rag"), version) for q in questions):
        from src.retrieval.pipeline import Resources, build_retrievers

        config = load_config()
        res = Resources(config)
        retriever = build_retrievers(config, res)["hybrid+bge-reranker"]
    try:
        results = {m: run_mode(split, m, questions, evidence, version, retriever) for m in answer_mod.MODES}
    finally:
        if res:
            res.close()

    print_problems(results)
    report = print_report(split, results)
    if split == "test":
        METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
        METRICS_OUT.write_text(json.dumps({
            "split": "test", "version": version, "n_questions": len(questions),
            "answer_model": answer_mod.MODEL, "judge_model": judge_mod.JUDGE_MODEL,
            "retrieval_config": str(CONFIG.relative_to(ROOT)),
            "results_files": {m: str(results_path(split, m).relative_to(ROOT)) for m in answer_mod.MODES},
            "modes": report,
        }, indent=2) + "\n")
        print(f"\nSaved {METRICS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
