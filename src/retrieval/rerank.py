"""Cross-encoder reranking of the hybrid retriever's top candidates.

Scores are cached per model in data/processed/rerank_cache/, keyed by (query, chunk text),
so re-running does not recompute them. The cache also keeps the time it took to score
each query, so timings can still be reported when everything comes from the cache.
"""

import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed" / "rerank_cache"
DEPTH = 20  # hybrid candidates passed to the reranker
BATCH_SIZE = 32
MAX_LENGTH = 512

ScoreFn = Callable[[str, list[str]], list[float]]


def rerank(query: str, candidates: list[dict], score_fn: ScoreFn) -> list[dict]:
    """Reorder candidates by score_fn(query, texts), highest first; ties keep the original order.

    Each returned chunk gets "score" (the reranker score) and "pre_rerank_rank" (its 1-based input rank).
    """
    if not candidates:
        return []
    scores = score_fn(query, [c["text"] for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [{**candidates[i], "score": float(scores[i]), "pre_rerank_rank": i + 1} for i in order]


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


class CrossEncoderReranker:
    """A sentence-transformers CrossEncoder with a persistent score cache."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=MAX_LENGTH)
        self.model.predict([("warm up", "warm up")])  # keep one-off setup out of per-query timings
        self.cache_path = CACHE_DIR / (re.sub(r"[^A-Za-z0-9._-]+", "_", model_name) + ".json")
        try:
            cache = json.loads(self.cache_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}
        self.scores: dict[str, float] = cache.get("scores", {})
        self.timings: dict[str, float] = cache.get("timings", {})
        self.last_seconds: float | None = None
        self.last_cached = False

    @property
    def device(self) -> str:
        return str(getattr(self.model, "device", "unknown"))

    def score(self, query: str, texts: list[str]) -> list[float]:
        keys = [_key(query, t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.scores]
        if missing:
            start = time.perf_counter()
            new = self.model.predict([(query, texts[i]) for i in missing], batch_size=BATCH_SIZE)
            self.last_seconds = time.perf_counter() - start
            self.last_cached = False
            for i, s in zip(missing, new):
                self.scores[keys[i]] = float(s)
            self.timings[_key(query)] = self.last_seconds
        else:
            self.last_seconds = self.timings.get(_key(query))
            self.last_cached = True
        return [self.scores[k] for k in keys]

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(
            {"model": self.model_name, "scores": self.scores, "timings": self.timings}))


class RerankRetriever:
    """Take the base retriever's top `depth` chunks and rerank them with a cross-encoder."""

    def __init__(self, base, reranker: CrossEncoderReranker, depth: int = DEPTH):
        self.base, self.reranker, self.depth = base, reranker, depth
        self.timings: list[float] = []
        self.n_cached = 0

    def search(self, query: str, k: int = 10) -> list[dict]:
        ranked = rerank(query, self.base.search(query, self.depth), self.reranker.score)
        if self.reranker.last_seconds is not None:
            self.timings.append(self.reranker.last_seconds)
        self.n_cached += self.reranker.last_cached
        return ranked[:k]

    def timing_summary(self) -> dict:
        t = self.timings
        return {
            "n_queries": len(t),
            "from_cache": self.n_cached,
            "mean_s": statistics.mean(t) if t else None,
            "median_s": statistics.median(t) if t else None,
            "max_s": max(t) if t else None,
        }
