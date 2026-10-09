"""Hybrid retrieval: reciprocal rank fusion (RRF) of dense and BM25 results."""

RRF_K = 60
DEPTH = 50  # candidates taken from each retriever before fusion


def rrf_fuse(rankings: list[list[dict]], k: int = RRF_K, key: str = "chunk_id") -> list[dict]:
    """Fuse ranked lists: each item scores sum(1 / (k + rank)) over the lists it appears in.

    Ties are broken by best single-list rank, then by key, so the order is deterministic.
    Returned items are the first-seen dicts with "score" replaced by the RRF score.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    items: dict[str, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            item_id = item[key]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            best_rank[item_id] = min(best_rank.get(item_id, rank), rank)
            items.setdefault(item_id, item)
    order = sorted(scores, key=lambda i: (-scores[i], best_rank[i], i))
    return [{**items[i], "score": scores[i]} for i in order]


class HybridRetriever:
    def __init__(self, dense, bm25, depth: int = DEPTH, k: int = RRF_K):
        self.dense, self.bm25, self.depth, self.k = dense, bm25, depth, k

    def search(self, query: str, k: int = 10) -> list[dict]:
        fused = rrf_fuse([self.dense.search(query, self.depth), self.bm25.search(query, self.depth)], self.k)
        return fused[:k]
