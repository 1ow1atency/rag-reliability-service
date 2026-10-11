from src.retrieval.rerank import RerankRetriever, rerank


def chunks(*texts):
    return [{"chunk_id": f"c{i}", "text": t, "score": 0.0} for i, t in enumerate(texts)]


def overlap_score(query, texts):
    """Fake cross-encoder: number of query words found in the text."""
    words = set(query.lower().split())
    return [float(len(words & set(t.lower().split()))) for t in texts]


class FakeBase:
    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, query, k):
        self.calls.append(k)
        return self.results[:k]


class FakeReranker:
    last_seconds, last_cached = 0.01, False

    def score(self, query, texts):
        return overlap_score(query, texts)


def test_rerank_orders_by_score_and_keeps_ties_stable():
    candidates = chunks("pods and nodes", "join token expires", "token", "unrelated text", "token")
    ranked = rerank("join token expires", candidates, overlap_score)

    assert [c["chunk_id"] for c in ranked] == ["c1", "c2", "c4", "c0", "c3"]  # c2 before c4: tie keeps order
    assert [c["pre_rerank_rank"] for c in ranked] == [2, 3, 5, 1, 4]
    assert ranked[0]["score"] == 3.0
    assert rerank("anything", [], overlap_score) == []

    # The retriever asks the base for `depth` candidates and returns the reranked top k.
    base = FakeBase(chunks(*[f"filler {i}" for i in range(19)], "join token expires"))
    retriever = RerankRetriever(base, FakeReranker(), depth=20)
    top = retriever.search("join token expires", k=3)
    assert base.calls == [20]
    assert len(top) == 3 and top[0]["chunk_id"] == "c19" and top[0]["pre_rerank_rank"] == 20
    assert retriever.timing_summary()["n_queries"] == 1
