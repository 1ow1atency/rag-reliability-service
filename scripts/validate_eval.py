"""Validate data/eval/questions.jsonl.

Failures: invalid JSON, missing fields, duplicate ids, bad split, or an answerable
question whose source_file does not exist under the docs folder.
Warnings: an answerable question whose answer string does not appear in its source file.
Exits non-zero if there are any failures.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
DOCS_ROOT = ROOT / "data" / "raw" / "k8s-website" / "content" / "en" / "docs"
REQUIRED_FIELDS = ["id", "question", "answer", "source_file", "answerable", "split"]
SPLITS = {"dev", "test"}


def main() -> int:
    failures, warnings, rows = [], [], []
    seen_ids = Counter()

    for n, line in enumerate(QUESTIONS.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            failures.append(f"line {n}: invalid JSON ({e})")
            continue
        if not isinstance(row, dict):
            failures.append(f"line {n}: not a JSON object")
            continue
        rows.append(row)
        label = f"line {n} ({row.get('id', '?')})"

        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            failures.append(f"{label}: missing fields {missing}")
        if "id" in row:
            seen_ids[row["id"]] += 1
        if "split" in row and row["split"] not in SPLITS:
            failures.append(f"{label}: split is {row['split']!r}, expected 'dev' or 'test'")

        if row.get("answerable") is True and "source_file" in row:
            path = DOCS_ROOT / row["source_file"]
            if not row["source_file"] or not path.is_file():
                failures.append(f"{label}: source_file not found: {row['source_file']!r}")
            elif str(row.get("answer", "")).lower() not in path.read_text(encoding="utf-8").lower():
                warnings.append(f"{label}: answer {row.get('answer')!r} not found verbatim in {row['source_file']}")

    for id_, count in seen_ids.items():
        if count > 1:
            failures.append(f"id {id_!r} appears {count} times")

    splits = Counter(r.get("split") for r in rows)
    answerable = Counter(r.get("answerable") is True for r in rows)
    print(f"Questions: {len(rows)}")
    print(f"  dev: {splits['dev']}  test: {splits['test']}")
    print(f"  answerable: {answerable[True]}  unanswerable: {answerable[False]}")

    print(f"\nFailures: {len(failures)}")
    for f in failures:
        print(f"  FAIL {f}")
    print(f"\nWarnings: {len(warnings)}")
    for w in warnings:
        print(f"  WARN {w}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
