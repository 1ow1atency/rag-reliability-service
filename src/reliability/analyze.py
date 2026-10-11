"""Exploratory analysis: does sampling uncertainty flag bad answers? (5 samples per question)

Usage: python -m src.reliability.analyze [--mode closed_book]

Labels come from the saved generation rows (temperature-0 answers) for the locked prompt version
only. A "bad" answer is INCORRECT on an answerable question or ANSWERED on an unanswerable one;
CORRECT and ABSTAINED are good; NOT_ATTEMPTED is excluded and counted. Scores come from
results/reliability/<mode>_samples.jsonl. AUROC (higher score = more likely bad) with 95%
bootstrap intervals and paired bootstrap differences follow the semantic-entropy-hallucination
project: 1,000 question resamples, seed 0, resamples with one class skipped, entropies rounded
to 9 decimals so float noise counts as ties.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.generation_eval import load_done as load_generation_rows
from src.evaluation.generation_eval import results_path, run_version
from src.reliability.sample_entropy import load_done as load_sample_rows
from src.reliability.sample_entropy import out_path, settings_version

ROOT = Path(__file__).resolve().parents[2]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
ROUND_DECIMALS = 9
SCORES = {
    "semantic_entropy": "Semantic entropy",
    "naive_entropy": "Naive entropy",
    "one_minus_agreement": "1 - agreement rate",
}
BAD_LABELS = {"INCORRECT", "ANSWERED"}
GOOD_LABELS = {"CORRECT", "ABSTAINED"}


def load_records(mode: str) -> tuple[list[dict], dict]:
    """Questions with both a locked-version label and a scored sample row, plus exclusion counts."""
    version = run_version()
    labels = {}
    for split in ("dev", "test"):
        labels.update(load_generation_rows(results_path(split, mode), version))
    samples = load_sample_rows(out_path(mode), settings_version())

    records, counts = [], Counter()
    for qid, s in samples.items():
        row = labels.get(qid)
        if row is None:
            counts["no_label"] += 1
            continue
        if row["label"] == "NOT_ATTEMPTED":
            counts["not_attempted"] += 1
            continue
        if row["label"] not in BAD_LABELS | GOOD_LABELS:
            counts[f"other_label_{row['label']}"] += 1
            continue
        records.append({"id": qid, "split": s["split"], "answerable": s["answerable"], "label": row["label"],
                        "bad": int(row["label"] in BAD_LABELS),
                        **{k: round(s[k], ROUND_DECIMALS) for k in SCORES}})
    return sorted(records, key=lambda r: r["id"]), dict(counts)


def bootstrap_aurocs(bad: np.ndarray, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """AUROC of every score on the same question resamples; one-class resamples are skipped."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = {k: [] for k in scores}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(bad), len(bad))
        if len(set(bad[idx])) < 2:
            continue
        for k, s in scores.items():
            boot[k].append(roc_auc_score(bad[idx], s[idx]))
    return {k: np.array(v) for k, v in boot.items()}


def ci95(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def tie_table(score: np.ndarray, bad: np.ndarray) -> list[dict]:
    """Questions per distinct score value, with how many of them are bad."""
    return [{"value": float(v), "n": int((score == v).sum()), "bad": int(bad[score == v].sum())}
            for v in np.unique(score)]


def selective_curve(good: np.ndarray, score: np.ndarray) -> list[dict]:
    """Answer only questions with score <= threshold; thresholds sweep unique values, so ties move together."""
    return [{"threshold": float(t), "answered": int((score <= t).sum()), "coverage": float(np.mean(score <= t)),
             "accuracy": float(np.mean(good[score <= t]))} for t in np.unique(score)]


def analyze(records: list[dict]) -> dict:
    bad = np.array([r["bad"] for r in records])
    good = 1 - bad
    scores = {k: np.array([r[k] for r in records], dtype=float) for k in SCORES}
    out = {"n": len(records), "positives_bad": int(bad.sum()), "negatives_good": int(good.sum()),
           "accuracy_answer_all": float(good.mean()) if len(good) else None,
           "ties": {k: tie_table(s, bad) for k, s in scores.items()}}
    if not 0 < bad.sum() < len(bad):
        out["auroc"] = None
        return out
    boot = bootstrap_aurocs(bad, scores)
    out["bootstrap_resamples_used"] = len(boot["semantic_entropy"])
    out["auroc"] = {k: {"auroc": float(roc_auc_score(bad, s)), "ci95": ci95(boot[k])} for k, s in scores.items()}
    out["semantic_entropy_minus"] = {}
    for k in SCORES:
        if k == "semantic_entropy":
            continue
        diffs = boot["semantic_entropy"] - boot[k]
        out["semantic_entropy_minus"][k] = {
            "observed": out["auroc"]["semantic_entropy"]["auroc"] - out["auroc"][k]["auroc"],
            "mean": float(diffs.mean()), "ci95": ci95(diffs)}
    out["selective_semantic_entropy"] = selective_curve(good, scores["semantic_entropy"])
    return out


def print_report(name: str, res: dict) -> None:
    print(f"\n=== {name}: n={res['n']}, bad (positives)={res['positives_bad']}, good (negatives)={res['negatives_good']}, "
          f"accuracy answering all={res['accuracy_answer_all']:.1%} ===")
    if res["auroc"] is None:
        print("  AUROC undefined: only one class")
        return
    print(f"  {'score':<20} {'AUROC':>6}   95% CI   ({res['bootstrap_resamples_used']} resamples used)")
    for k, label in SCORES.items():
        a = res["auroc"][k]
        print(f"  {label:<20} {a['auroc']:6.3f}   [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]")
    print("  AUROC difference, semantic entropy minus baseline (paired bootstrap)")
    for k, d in res["semantic_entropy_minus"].items():
        print(f"    - {SCORES[k]:<18} observed {d['observed']:+.3f}  mean {d['mean']:+.3f}  "
              f"95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    print("  Questions per score value (ties):")
    for k, label in SCORES.items():
        cells = ", ".join(f"{t['value']:.3f}: {t['n']} ({t['bad']} bad)" for t in res["ties"][k])
        print(f"    {label:<20} {len(res['ties'][k])} distinct values -> {cells}")
    curve = res["selective_semantic_entropy"]
    print("  Selective answering by semantic entropy (answer if SE <= threshold):")
    for p in curve:
        print(f"    SE <= {p['threshold']:.3f}: answer {p['answered']:>2}/{res['n']} "
              f"(coverage {p['coverage']:5.1%}), accuracy {p['accuracy']:5.1%}")
    floor = curve[0]
    print(f"  Coverage floor: {floor['coverage']:.1%} ({floor['answered']} questions tie at SE = {floor['threshold']:.3f}); "
          f"no threshold answers fewer without breaking that tie.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["closed_book", "rag"], default="closed_book")
    mode = parser.parse_args().mode

    records, excluded = load_records(mode)
    print(f"mode={mode}: {len(records)} questions analyzed; excluded: {excluded or 'none'}")
    results = {
        "test": analyze([r for r in records if r["split"] == "test"]),
        "dev_test_pooled": analyze(records),
    }
    print_report("test only", results["test"])
    print_report("dev + test pooled", results["dev_test_pooled"])

    out = ROOT / "results" / "metrics" / f"reliability_{mode}.json"
    out.write_text(json.dumps({
        "exploratory": True, "mode": mode, "samples_per_question": 5,
        "label_source": f"results/generation/<split>_{mode}.jsonl, prompt version {run_version()} only",
        "bad_definition": "INCORRECT on answerable or ANSWERED on unanswerable; NOT_ATTEMPTED excluded",
        "excluded_counts": excluded, "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
        "results": results, "records": records,
    }, indent=2) + "\n")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
