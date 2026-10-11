"""Post-hoc, exploratory: recluster the saved closed-book samples using only each sample's first sentence.

Usage: python -m src.reliability.recluster

One variant only, "first_sentence": each saved sample is cut at the first '. ', '? ', '! ' or
newline (the whole text if none occurs), then clustered with the same NLI code and question prefix
as the original run (src/reliability/entailment.py). No model API calls; labels and the original
scores are read, not changed. Defined after the manual check of the original clusters, so this
is post-hoc. Bootstrap settings are the same as in analyze.py.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.reliability.analyze import ROUND_DECIMALS, bootstrap_aurocs, ci95, load_records, tie_table
from src.reliability.entropy import semantic_entropy
from src.reliability.sample_entropy import load_done, out_path, settings_version

ROOT = Path(__file__).resolve().parents[2]
MODE = "closed_book"
VARIANT = "first_sentence"
SAMPLES_OUT = ROOT / "results" / "reliability" / f"{MODE}_recluster_{VARIANT}.jsonl"
METRICS_OUT = ROOT / "results" / "metrics" / f"reliability_{MODE}_recluster_{VARIANT}.json"
SEPARATORS = (". ", "? ", "! ", "\n")

# From the manual check of the original clusters (README), for the before/after printout.
FORMER_GROUP_SE = 1.33217904
TRULY_DISAGREE = ["q006", "q010", "q030", "q047"]


def first_sentence(text: str) -> str:
    """Text up to the first '. ', '? ', '! ' (punctuation kept) or newline; the whole text if none."""
    cuts = []
    for sep in SEPARATORS:
        i = text.find(sep)
        if i != -1:
            cuts.append(i if sep == "\n" else i + 1)  # keep the '.', '?' or '!'
    return (text[:min(cuts)] if cuts else text).strip()


def recluster(rows: dict[str, dict]) -> dict[str, dict]:
    from src.reliability.entailment import cluster_answers

    out = {}
    for qid, row in sorted(rows.items()):
        firsts = [first_sentence(s["text"]) for s in row["samples"]]
        ids = cluster_answers(row["question"], firsts)
        out[qid] = {"id": qid, "variant": VARIANT, "first_sentences": firsts, "cluster_ids": ids,
                    "n_clusters": len(set(ids)), "semantic_entropy": round(semantic_entropy(ids), ROUND_DECIMALS)}
    return out


def auroc_report(records: list[dict], new: dict[str, dict]) -> dict:
    bad = np.array([r["bad"] for r in records])
    scores = {"original": np.array([r["semantic_entropy"] for r in records]),
              VARIANT: np.array([new[r["id"]]["semantic_entropy"] for r in records])}
    boot = bootstrap_aurocs(bad, scores)
    diffs = boot[VARIANT] - boot["original"]
    return {
        "n": len(records), "positives_bad": int(bad.sum()), "negatives_good": int((1 - bad).sum()),
        "bootstrap_resamples_used": len(diffs),
        "auroc": {k: {"auroc": float(roc_auc_score(bad, s)), "ci95": ci95(boot[k])} for k, s in scores.items()},
        "first_sentence_minus_original": {
            "observed": float(roc_auc_score(bad, scores[VARIANT]) - roc_auc_score(bad, scores["original"])),
            "mean": float(diffs.mean()), "ci95": ci95(diffs)},
        "ties": tie_table(scores[VARIANT], bad),
    }


def print_clusters(qid: str, row: dict, new: dict, records: dict) -> None:
    r = records[qid]
    print(f"\n  {qid} [{'bad' if r['bad'] else 'good'}] SE {r['semantic_entropy']:.3f} -> {new['semantic_entropy']:.3f}"
          f"  {row['question']}")
    for cid, text in sorted(zip(new["cluster_ids"], new["first_sentences"]), key=lambda p: p[0]):
        print(f"    [{cid}] {' '.join(text.split())}")


def main() -> None:
    rows = load_done(out_path(MODE), settings_version())
    records, excluded = load_records(MODE)
    by_id = {r["id"]: r for r in records}
    new = recluster({qid: rows[qid] for qid in by_id})
    with SAMPLES_OUT.open("w", encoding="utf-8") as f:
        for qid in sorted(new):
            f.write(json.dumps(new[qid], ensure_ascii=False) + "\n")

    results = {"test": auroc_report([r for r in records if r["split"] == "test"], new),
               "dev_test_pooled": auroc_report(records, new)}
    print(f"variant={VARIANT} (post-hoc, exploratory): {len(records)} questions; excluded: {excluded or 'none'}")
    for name, res in results.items():
        print(f"\n=== {name}: n={res['n']}, bad={res['positives_bad']}, good={res['negatives_good']} "
              f"({res['bootstrap_resamples_used']} resamples used) ===")
        for k, a in res["auroc"].items():
            print(f"  {k + ' semantic entropy':<32} AUROC {a['auroc']:.3f}  95% CI [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]")
        d = res["first_sentence_minus_original"]
        print(f"  first_sentence minus original (paired): observed {d['observed']:+.3f}  mean {d['mean']:+.3f}  "
              f"95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
        print("  first_sentence SE per value (ties): "
              + ", ".join(f"{t['value']:.3f}: {t['n']} ({t['bad']} bad)" for t in res["ties"]))

    former = [qid for qid, r in by_id.items() if abs(r["semantic_entropy"] - FORMER_GROUP_SE) < 1e-6]
    moves = {qid: new[qid]["semantic_entropy"] - by_id[qid]["semantic_entropy"] for qid in former}
    down = sorted(q for q, d in moves.items() if d < -1e-9)
    up = sorted(q for q, d in moves.items() if d > 1e-9)
    same = sorted(q for q, d in moves.items() if abs(d) <= 1e-9)
    print(f"\nFormer SE=1.332 questions ({len(former)}): {len(down)} lower, {len(up)} higher, {len(same)} unchanged")
    for label, ids in [("lower", down), ("higher", up), ("unchanged", same)]:
        print(f"  {label:<9}: " + ", ".join(f"{q} {by_id[q]['semantic_entropy']:.3f}->{new[q]['semantic_entropy']:.3f}"
                                            f" ({'bad' if by_id[q]['bad'] else 'good'})" for q in ids))

    zero = sorted(qid for qid, r in by_id.items() if r["semantic_entropy"] == 0)
    print("\nNew clusters for the 4 'truly disagree' questions (check that different answers stay apart):")
    for qid in TRULY_DISAGREE:
        print_clusters(qid, rows[qid], new[qid], by_id)
    print(f"\nNew clusters for the {len(zero)} original entropy-0 questions:")
    for qid in zero:
        print_clusters(qid, rows[qid], new[qid], by_id)

    METRICS_OUT.write_text(json.dumps({
        "post_hoc": True, "exploratory": True, "mode": MODE, "variant": VARIANT,
        "definition": "first sentence of each saved sample (cut at '. ', '? ', '! ' or newline), same NLI clustering",
        "results": results,
        "former_1332_moves": {"lower": down, "higher": up, "unchanged": same},
        "per_question": {q: {"original": by_id[q]["semantic_entropy"], VARIANT: new[q]["semantic_entropy"],
                             "bad": by_id[q]["bad"], "split": by_id[q]["split"]} for q in sorted(by_id)},
    }, indent=2) + "\n")
    print(f"\nSaved {SAMPLES_OUT.relative_to(ROOT)} and {METRICS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
