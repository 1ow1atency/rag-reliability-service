"""Corpus statistics for the fetched Kubernetes docs: document count and token lengths.

Tokens are counted with a simple regex tokenizer (words and individual punctuation
marks), so counts approximate, but will not exactly match, a subword tokenizer.
"""

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "data" / "raw" / "k8s-website" / "content" / "en" / "docs"
SECTIONS = ["concepts", "tasks"]

FRONT_MATTER_RE = re.compile(r"\A(?:---\s*\n.*?\n---|\+\+\+\s*\n.*?\n\+\+\+)\s*(?:\n|\Z)", re.DOTALL)
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def strip_front_matter(text: str) -> str:
    return FRONT_MATTER_RE.sub("", text, count=1)


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def load_documents() -> dict[Path, str]:
    docs = {}
    for section in SECTIONS:
        for path in sorted((DOCS_ROOT / section).rglob("*.md")):
            docs[path] = strip_front_matter(path.read_text(encoding="utf-8"))
    return docs


def main() -> None:
    if not DOCS_ROOT.exists():
        raise SystemExit(f"{DOCS_ROOT} not found; run scripts/fetch_docs.py first")

    docs = load_documents()
    lengths = np.array([count_tokens(text) for text in docs.values()])
    empty = int((lengths == 0).sum())

    print(f"Documents: {len(docs)}")
    for section in SECTIONS:
        n = sum(1 for p in docs if p.is_relative_to(DOCS_ROOT / section))
        print(f"  {section}: {n}")
    print(f"Empty after front matter strip: {empty}")
    print(f"Total tokens: {lengths.sum():,}")
    print("Token length per document:")
    print(f"  min:    {lengths.min():,}")
    print(f"  median: {int(np.median(lengths)):,}")
    print(f"  p95:    {int(np.percentile(lengths, 95)):,}")
    print(f"  max:    {lengths.max():,}")

    longest = max(docs, key=lambda p: count_tokens(docs[p]))
    print(f"Longest document: {longest.relative_to(DOCS_ROOT)}")


if __name__ == "__main__":
    main()
