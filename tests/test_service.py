from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from src.service.app import create_app


class FakePipeline:
    def __init__(self):
        self.chunks = [
            {"chunk_id": f"concepts/p{i}.md#0", "doc_path": f"concepts/p{i}.md",
             "text": f"Chunk {i} text.\nMore text.", "score": 0.0}
            for i in range(5)
        ]

    def retrieve(self, question):
        return self.chunks

    def rerank(self, question, candidates):
        scores = [0.2, 0.9, 0.5, 0.1, 0.3]
        ranked = sorted(({**c, "score": s} for c, s in zip(candidates, scores)), key=lambda c: -c["score"])
        return ranked[:3]


@dataclass
class FakeAnswer:
    text: str | None
    errors: list[str] = field(default_factory=list)


def make_client(reply):
    calls = []

    def llm(question, chunks):
        calls.append((question, [c["chunk_id"] for c in chunks]))
        if isinstance(reply, Exception):
            raise reply
        return reply

    return TestClient(create_app(pipeline_factory=FakePipeline, llm=llm)), calls


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def test_normal_answer(key):
    client, calls = make_client(FakeAnswer("30 seconds"))
    with client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/ask", json={"question": "  How long is the grace period?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "30 seconds" and body["abstained"] is False
    assert [s["page"] for s in body["sources"]] == ["concepts/p1.md", "concepts/p2.md", "concepts/p4.md"]
    assert body["sources"][0]["snippet"] == "Chunk 1 text. More text."
    assert body["top_rerank_score"] == 0.9
    # Response shape: unchanged apart from the added queue_wait_ms in latency_ms.
    assert set(body) == {"answer", "abstained", "sources", "top_rerank_score", "latency_ms"}
    assert set(body["sources"][0]) == {"page", "chunk_id", "rerank_score", "snippet"}
    assert set(body["latency_ms"]) == {"retrieve", "queue_wait_ms", "rerank", "generate", "total"}
    assert body["latency_ms"]["queue_wait_ms"] < 1000
    # The LLM gets the stripped question and exactly the top 3 reranked chunks.
    assert calls == [("How long is the grace period?", ["concepts/p1.md#0", "concepts/p2.md#0", "concepts/p4.md#0"])]


def test_not_in_docs_is_abstained(key):
    client, _ = make_client(FakeAnswer("NOT_IN_DOCS"))
    with client:
        body = client.post("/ask", json={"question": "What does the CKA exam cost?"}).json()
    assert body["answer"] == "NOT_IN_DOCS" and body["abstained"] is True


def test_missing_key_returns_503_without_calling_the_llm(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client, calls = make_client(FakeAnswer("30 seconds"))
    with client:
        r = client.post("/ask", json={"question": "How long?"})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]
    assert calls == []


def test_empty_question_returns_422(key):
    client, calls = make_client(FakeAnswer("30 seconds"))
    with client:
        for payload in [{"question": ""}, {"question": "   "}, {}]:
            assert client.post("/ask", json=payload).status_code == 422
    assert calls == []


def test_groq_failure_returns_502_without_leaking_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value")
    for reply in [FakeAnswer(None, ["AuthenticationError: invalid key gsk_secret_value"]),
                  RuntimeError("boom gsk_secret_value")]:
        client, _ = make_client(reply)
        with client:
            r = client.post("/ask", json={"question": "How long?"})
        assert r.status_code == 502
        assert "gsk_secret_value" not in r.text


def test_service_index_check_fails_fast_instead_of_building(tmp_path):
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.pipeline import IndexMissing, build_retrievers, load_config

    # Missing chunk file: raises before touching any model or index.
    with pytest.raises(IndexMissing, match="build_index.py"):
        build_retrievers(load_config(), res=None, chunker="does_not_exist", build=False)

    # BM25 load-only: missing cache, then a cache that no longer matches the chunks file.
    chunks = tmp_path / "chunks_x.jsonl"
    chunks.write_text('{"chunk_id": "a#0", "doc_path": "a", "text": "pods and nodes", "n_tokens": 3}\n')
    with pytest.raises(FileNotFoundError, match="not found"):
        BM25Retriever(chunks).load()
    BM25Retriever(chunks).build()
    loaded = BM25Retriever(chunks)
    loaded.load()
    assert loaded.search("pods", k=1)[0]["chunk_id"] == "a#0"
    chunks.write_text(chunks.read_text() + '{"chunk_id": "b#0", "doc_path": "b", "text": "more", "n_tokens": 1}\n')
    with pytest.raises(FileNotFoundError, match="stale"):
        BM25Retriever(chunks).load()


def test_health_answers_while_a_slow_ask_is_running(key):
    """/ask runs in the threadpool, so a blocked /ask must not block /health."""
    import threading
    import time

    entered, release = threading.Event(), threading.Event()

    class SlowPipeline(FakePipeline):
        def retrieve(self, question):
            entered.set()
            release.wait(timeout=10)  # held until /health has been checked
            return self.chunks

    client = TestClient(create_app(pipeline_factory=SlowPipeline, llm=lambda q, c: FakeAnswer("30 seconds")))
    with client:
        result = {}
        ask = threading.Thread(target=lambda: result.update(r=client.post("/ask", json={"question": "q"})))
        ask.start()
        assert entered.wait(timeout=5), "/ask never reached the pipeline"

        health = {}
        probe = threading.Thread(target=lambda: health.update(r=client.get("/health"), t=time.perf_counter()))
        t0 = time.perf_counter()
        probe.start()
        probe.join(timeout=2)
        blocked = probe.is_alive()
        release.set()
        ask.join(timeout=10)
        probe.join(timeout=10)

    assert not blocked, "/health was blocked by the running /ask"
    assert health["r"].status_code == 200 and health["t"] - t0 < 2
    assert result["r"].status_code == 200


def test_llm_stub_is_off_by_default_and_skips_groq_when_on(monkeypatch):
    import src.service.app as app_mod

    groq_calls = []
    monkeypatch.setattr(app_mod, "groq_answer", lambda q, c: groq_calls.append(q) or FakeAnswer("24 hours"))

    # Default (LLM_STUB unset, or anything other than "1"): the real Groq path, key required.
    for value in [None, "0", "true"]:
        if value is None:
            monkeypatch.delenv("LLM_STUB", raising=False)
        else:
            monkeypatch.setenv("LLM_STUB", value)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        with TestClient(create_app(pipeline_factory=FakePipeline)) as client:
            assert client.get("/health").json() == {"status": "ok"}
            assert client.post("/ask", json={"question": "q"}).json()["answer"] == "24 hours"
        monkeypatch.delenv("GROQ_API_KEY")
        with TestClient(create_app(pipeline_factory=FakePipeline)) as client:
            assert client.post("/ask", json={"question": "q"}).status_code == 503
    assert len(groq_calls) == 3

    # LLM_STUB=1: fixed reply, Groq never called, no key needed; retrieval and rerank still run.
    monkeypatch.setenv("LLM_STUB", "1")
    with TestClient(create_app(pipeline_factory=FakePipeline)) as client:
        assert client.get("/health").json() == {"status": "ok", "llm_stub": True}
        body = client.post("/ask", json={"question": "q"}).json()
    assert body["answer"] == app_mod.STUB_REPLY and body["abstained"] is False
    assert [s["page"] for s in body["sources"]] == ["concepts/p1.md", "concepts/p2.md", "concepts/p4.md"]
    assert len(groq_calls) == 3


# ---------- one rerank at a time ----------

import threading  # noqa: E402
import time  # noqa: E402


class GatedPipeline(FakePipeline):
    """rerank() blocks until released, and records how many reranks overlap."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.entered = threading.Semaphore(0)
        self.active = self.max_active = 0
        self.lock = threading.Lock()

    def rerank(self, question, candidates):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.release()
        self.release.wait(timeout=10)
        with self.lock:
            self.active -= 1
        return super().rerank(question, candidates)


def gated_client(**kwargs):
    pipeline = GatedPipeline()
    app = create_app(pipeline_factory=lambda: pipeline, llm=lambda q, c: FakeAnswer("30 seconds"), **kwargs)
    return TestClient(app), app, pipeline


def post_in_thread(client, results, key):
    t = threading.Thread(target=lambda: results.__setitem__(key, client.post("/ask", json={"question": key})))
    t.start()
    return t


def wait_until(cond, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_two_concurrent_requests_are_serialised(key):
    client, app, pipeline = gated_client(max_waiting=8, wait_timeout_s=10)
    with client:
        results = {}
        threads = [post_in_thread(client, results, "a"), post_in_thread(client, results, "b")]
        assert pipeline.entered.acquire(timeout=5)                 # one request is reranking ...
        assert wait_until(lambda: app.state.gate.waiting == 1)     # ... and the other is queued
        assert pipeline.max_active == 1
        pipeline.release.set()
        for t in threads:
            t.join(timeout=10)
    assert pipeline.max_active == 1
    assert [results[k].status_code for k in "ab"] == [200, 200]
    waits = sorted(results[k].json()["latency_ms"]["queue_wait_ms"] for k in "ab")
    assert waits[0] < 1000 and waits[1] > 0


def test_queue_full_returns_503_with_retry_after(key):
    client, app, pipeline = gated_client(max_waiting=1, wait_timeout_s=10)
    with client:
        results = {}
        threads = [post_in_thread(client, results, "running")]
        assert pipeline.entered.acquire(timeout=5)
        threads.append(post_in_thread(client, results, "waiting"))
        assert wait_until(lambda: app.state.gate.waiting == 1)
        r = client.post("/ask", json={"question": "rejected"})
        pipeline.release.set()
        for t in threads:
            t.join(timeout=10)
    assert r.status_code == 503 and r.headers["Retry-After"] == "10"
    assert r.json()["detail"]["error"] == "queue_full" and r.json()["detail"]["max_waiting"] == 1
    assert results["running"].status_code == 200 and results["waiting"].status_code == 200


def test_wait_timeout_returns_503_with_retry_after(key):
    client, app, pipeline = gated_client(max_waiting=8, wait_timeout_s=0.2)
    with client:
        results = {}
        t = post_in_thread(client, results, "running")
        assert pipeline.entered.acquire(timeout=5)
        r = client.post("/ask", json={"question": "times out"})
        pipeline.release.set()
        t.join(timeout=10)
    assert r.status_code == 503 and "Retry-After" in r.headers
    assert r.json()["detail"]["error"] == "queue_timeout"
    assert results["running"].status_code == 200
    assert app.state.gate.waiting == 0 and app.state.gate.active == 0


def test_health_is_fast_while_one_reranks_and_others_wait(key):
    client, app, pipeline = gated_client(max_waiting=8, wait_timeout_s=10)
    with client:
        results = {}
        threads = [post_in_thread(client, results, k) for k in ("a", "b", "c")]
        assert pipeline.entered.acquire(timeout=5)
        assert wait_until(lambda: app.state.gate.waiting == 2)
        t0 = time.perf_counter()
        health = client.get("/health")
        elapsed = time.perf_counter() - t0
        pipeline.release.set()
        for t in threads:
            t.join(timeout=10)
    assert health.status_code == 200 and elapsed < 1.0
    assert all(results[k].status_code == 200 for k in "abc") and pipeline.max_active == 1
