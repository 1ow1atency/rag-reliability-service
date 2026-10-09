import pytest

from src.retrieval.hybrid import rrf_fuse


def ranked(*ids):
    return [{"chunk_id": i, "score": 0.0} for i in ids]


def test_rrf_fuse_scores_order_and_ties():
    fused = rrf_fuse([ranked("a", "b", "c"), ranked("c", "a", "d")], k=60)

    scores = {item["chunk_id"]: item["score"] for item in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)  # only in one list
    assert [item["chunk_id"] for item in fused] == ["a", "c", "b", "d"]

    # Equal RRF scores: tie broken by best single-list rank, then id.
    tied = rrf_fuse([ranked("x", "y"), ranked("y", "x")], k=60)
    assert tied[0]["score"] == pytest.approx(tied[1]["score"])
    assert [item["chunk_id"] for item in tied] == ["x", "y"]
