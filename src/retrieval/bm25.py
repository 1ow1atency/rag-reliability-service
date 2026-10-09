"""BM25 (rank-bm25 Okapi) retrieval over a chunk file.

Tokenization is lowercase. Identifiers are kept whole and also split into parts, so
`terminationGracePeriodSeconds` yields terminationgraceperiodseconds, termination, grace,
period, seconds, and `--node-eviction-rate` yields node-eviction-rate, node, eviction, rate
(leading dashes are dropped). The index is cached next to the chunks and rebuilt only when
the chunks file or the tokenizer version changes.
"""

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.dense import file_sha256, load_chunks

TOKENIZER_VERSION = 1

# A word or identifier: alphanumerics joined by . _ - / (e.g. kube-apiserver, v1.32, a/b).
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._\-/][A-Za-z0-9]+)*")
SEPARATOR_RE = re.compile(r"[._\-/]")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in WORD_RE.findall(text):
        tokens.append(word.lower())
        parts = [p.lower() for piece in SEPARATOR_RE.split(word) for p in CAMEL_RE.findall(piece)]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class BM25Retriever:
    def __init__(self, chunks_path: Path):
        self.chunks_path = Path(chunks_path)
        self.cache_path = self.chunks_path.with_suffix(".bm25.pkl")
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def build(self, force: bool = False) -> bool:
        """Load the cached index if it matches the chunks file; otherwise build it. True if built."""
        key = {"chunks_sha256": file_sha256(self.chunks_path), "tokenizer": TOKENIZER_VERSION}
        if not force and self.cache_path.exists():
            with self.cache_path.open("rb") as f:
                cached = pickle.load(f)
            if cached["key"] == key:
                self.chunks, self.bm25 = cached["chunks"], cached["bm25"]
                return False

        self.chunks = load_chunks(self.chunks_path)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        with self.cache_path.open("wb") as f:
            pickle.dump({"key": key, "chunks": self.chunks, "bm25": self.bm25}, f)
        return True

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Top-k chunks for a query: each chunk's fields plus its BM25 "score"."""
        if self.bm25 is None:
            self.build()
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]
