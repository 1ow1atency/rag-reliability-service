import json
import math

import pytest

import src.reliability.sample_entropy as se_mod
from src.generation.groq_chat import ChatResult
from src.reliability.entropy import agreement_rate, naive_entropy, semantic_entropy


def test_entropy_scores():
    assert semantic_entropy([0, 0, 0, 0, 0]) == 0.0
    assert semantic_entropy([0, 1, 2, 3, 4]) == pytest.approx(math.log(5))
    assert semantic_entropy([0, 0, 0, 1, 1]) == pytest.approx(-(0.6 * math.log(0.6) + 0.4 * math.log(0.4)))
    # Naive entropy treats different strings as different answers even when they mean the same.
    assert naive_entropy(["30 seconds", "30 seconds.", "30s", "30 seconds", "30 seconds"]) > 0
    assert 1 - agreement_rate([0, 0, 0, 1, 2]) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        semantic_entropy([])


def test_rag_reuses_saved_top3_chunks(monkeypatch):
    saved_rows = {"q007": {"retrieved": [{"chunk_id": "b#1"}, {"chunk_id": "a#0"}, {"chunk_id": "c#2"}]}}
    monkeypatch.setattr(se_mod, "load_generation_rows", lambda path, version: saved_rows)
    monkeypatch.setattr(se_mod, "generation_version", lambda: "v")
    texts = {cid: {"chunk_id": cid, "doc_path": cid.split("#")[0], "text": f"text {cid}"}
             for cid in ["a#0", "b#1", "c#2", "d#3"]}
    cache = {}

    chunks = se_mod.saved_rag_chunks({"id": "q007", "split": "dev"}, texts, cache)
    assert [c["chunk_id"] for c in chunks] == ["b#1", "a#0", "c#2"]  # saved order, no new retrieval
    with pytest.raises(KeyError):
        se_mod.saved_rag_chunks({"id": "q999", "split": "dev"}, texts, cache)


def test_sampling_retries_failed_calls_and_counts_their_tokens(monkeypatch, tmp_path):
    replies = iter([
        ChatResult("", "stop", 5, 100, 105),            # empty: retried
        ChatResult("30 seconds", "stop", 8, 100, 108),
        ChatResult("cut off", "length", 4096, 100, 4196),  # hit the token limit: retried
        ChatResult("30s", "stop", 6, 100, 106),
        *[ChatResult("30 seconds.", "stop", 7, 100, 107) for _ in range(3)],
    ])
    monkeypatch.setattr(se_mod, "chat", lambda *args: next(replies))

    samples, failures, usage = se_mod.sample("closed_book", "How long?", None)
    assert [s["text"] for s in samples] == ["30 seconds", "30s", "30 seconds.", "30 seconds.", "30 seconds."]
    assert failures == []
    assert usage == {"calls": 7, "tokens": 105 + 108 + 4196 + 106 + 3 * 107}

    # Resume: only rows that finished with the current settings version count as done.
    path = tmp_path / "rag_samples.jsonl"
    rows = [{"id": "q001", "version": "v1", "status": "ok"},
            {"id": "q002", "version": "v1", "status": "failed"},
            {"id": "q003", "version": "v0", "status": "ok"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert list(se_mod.load_done(path, "v1")) == ["q001"]


def test_ties_and_selective_curve_keep_tied_questions_together():
    import numpy as np

    from src.reliability.analyze import selective_curve, tie_table

    score = np.array([0.0, 0.0, 0.0, 0.5, 1.332, 1.332])
    bad = np.array([1, 0, 0, 0, 1, 1])
    assert tie_table(score, bad) == [{"value": 0.0, "n": 3, "bad": 1},
                                     {"value": 0.5, "n": 1, "bad": 0},
                                     {"value": 1.332, "n": 2, "bad": 2}]
    curve = selective_curve(1 - bad, score)
    # The lowest threshold already answers all three tied questions: the coverage floor is 3/6.
    assert [(p["answered"], round(p["accuracy"], 3)) for p in curve] == [(3, 0.667), (4, 0.75), (6, 0.5)]
