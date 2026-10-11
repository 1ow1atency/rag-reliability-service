"""Paired RAG vs closed-book comparison on the answerable test questions, from saved results.

No model calls: reads results/generation/test_{closed_book,rag}.jsonl for the locked prompt
version. Reports the 2x2 agreement table, a 95% percentile bootstrap interval on the accuracy
difference (resampling questions, fixed seed), and accuracy with and without the hand-written
questions q002 and q004.
"""

import json
from pathlib import Path

import numpy as np

from src.evaluation.generation_eval import load_done, results_path, run_version

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "metrics" / "generation_test_paired.json"
SPLIT = "test"
HAND_WRITTEN = ["q002", "q004"]
N_BOOT = 10_000
SEED = 42


def paired_table(rag: dict[str, bool], closed: dict[str, bool], ids: list[str]) -> dict:
    return {
        "both_correct": sum(rag[i] and closed[i] for i in ids),
        "rag_only": sum(rag[i] and not closed[i] for i in ids),
        "closed_only": sum(closed[i] and not rag[i] for i in ids),
        "both_wrong": sum(not rag[i] and not closed[i] for i in ids),
    }


def bootstrap_diff(rag: np.ndarray, closed: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile bootstrap CI on mean(rag) - mean(closed), resampling questions (pairs kept together)."""
    rng = np.random.default_rng(seed)
    n = len(rag)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = rag[idx].mean(axis=1) - closed[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(rag.mean() - closed.mean()), "ci95": [float(lo), float(hi)],
            "n_boot": n_boot, "seed": seed}


def compare(rag: dict[str, bool], closed: dict[str, bool], ids: list[str]) -> dict:
    r = np.array([rag[i] for i in ids], dtype=float)
    c = np.array([closed[i] for i in ids], dtype=float)
    return {
        "n": len(ids),
        "rag_correct": int(r.sum()), "closed_correct": int(c.sum()),
        "rag_accuracy": float(r.mean()), "closed_accuracy": float(c.mean()),
        "paired": paired_table(rag, closed, ids),
        "bootstrap": bootstrap_diff(r, c),
    }


def main() -> None:
    version = run_version()
    rows = {m: load_done(results_path(SPLIT, m), version) for m in ("closed_book", "rag")}
    ids = sorted(i for i, r in rows["rag"].items() if r["answerable"])
    assert ids == sorted(i for i, r in rows["closed_book"].items() if r["answerable"]), "modes cover different questions"
    rag = {i: rows["rag"][i]["label"] == "CORRECT" for i in ids}
    closed = {i: rows["closed_book"][i]["label"] == "CORRECT" for i in ids}

    results = {
        "all_answerable": compare(rag, closed, ids),
        "without_hand_written": compare(rag, closed, [i for i in ids if i not in HAND_WRITTEN]),
    }
    for name, res in results.items():
        p, b = res["paired"], res["bootstrap"]
        label = "all answerable" if name == "all_answerable" else f"without {', '.join(HAND_WRITTEN)}"
        print(f"\n{label} (n={res['n']})")
        print(f"  RAG        {res['rag_correct']:>2}/{res['n']}  {res['rag_accuracy']:.1%}")
        print(f"  closed     {res['closed_correct']:>2}/{res['n']}  {res['closed_accuracy']:.1%}")
        print(f"  both correct {p['both_correct']}, RAG only {p['rag_only']}, "
              f"closed only {p['closed_only']}, both wrong {p['both_wrong']}")
        print(f"  accuracy difference (RAG - closed): {b['diff']:+.1%}, "
              f"95% bootstrap CI [{b['ci95'][0]:+.1%}, {b['ci95'][1]:+.1%}] "
              f"({b['n_boot']:,} resamples, seed {b['seed']})")
    print(f"\nHand-written questions: " + ", ".join(
        f"{i} RAG={rows['rag'][i]['label']} closed={rows['closed_book'][i]['label']}" for i in HAND_WRITTEN))

    OUT.write_text(json.dumps({"split": SPLIT, "version": version, "hand_written_ids": HAND_WRITTEN,
                               "correct_by_id": {"rag": rag, "closed_book": closed}, **results}, indent=2) + "\n")
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
