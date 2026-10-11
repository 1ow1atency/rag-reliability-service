"""FastAPI service for the locked RAG pipeline.

POST /ask  {"question": "..."} -> answer, abstained, sources, top_rerank_score, latency_ms
GET  /health                   -> ok once indexes and models are loaded

Pipeline (configs/final_retrieval.json): hybrid RRF over dense + BM25 on fixed_512 chunks, the top
20 reranked by bge-reranker-base, the top 3 sent with the locked RAG prompt to gpt-oss-20b at
temperature 0 (src/generation/answer.py). Models and indexes load once at startup.
GROQ_API_KEY is read from the environment when a request needs it and is never logged or returned.

LLM_STUB=1 (load testing only; default off): /ask returns a fixed reply instead of calling Groq,
and no GROQ_API_KEY is needed. Retrieval and reranking still run, and /health reports llm_stub.

Run: uvicorn src.service.app:app --port 8000
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from src.generation.answer import NOT_IN_DOCS
from src.retrieval.rerank import rerank

SNIPPET_CHARS = 300
STUB_REPLY = "LLM_STUB: fixed reply for load testing; Groq was not called."


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class Source(BaseModel):
    page: str
    chunk_id: str
    rerank_score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[Source]
    top_rerank_score: float | None
    latency_ms: dict[str, float]


class Pipeline:
    """The locked retrieval pipeline, split into a retrieve and a rerank stage."""

    def __init__(self, config: dict | None = None):
        from src.retrieval.pipeline import Resources, build_retrievers, load_config

        self.config = config or load_config()
        self.resources = Resources(self.config)
        try:  # never build indexes at startup: fail fast if scripts/build_index.py has not been run
            self.hybrid = build_retrievers(self.config, self.resources, build=False)["hybrid"]
        except Exception:
            self.resources.client.close()
            raise
        self.rerank_depth = self.config["rerank"]["depth"]
        self.top_k = self.config["top_k"]
        model = self.resources.reranker.model
        self._score = lambda query, texts: [float(s) for s in model.predict([(query, t) for t in texts])]

    def retrieve(self, question: str) -> list[dict]:
        return self.hybrid.search(question, k=self.rerank_depth)

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        return rerank(question, candidates, self._score)[: self.top_k]

    def close(self) -> None:
        self.resources.client.close()  # the rerank score cache is not saved by the service


def groq_answer(question: str, chunks: list[dict]):
    from src.generation.answer import answer

    return answer("rag", question, chunks)


def stub_answer(question: str, chunks: list[dict]):
    from src.generation.answer import Answer

    return Answer(text=STUB_REPLY, finish_reason="stub")


def llm_stub_enabled() -> bool:
    return os.environ.get("LLM_STUB") == "1"


def is_abstention(text: str) -> bool:
    return text.strip().rstrip(".").strip() == NOT_IN_DOCS


def create_app(pipeline_factory: Callable[[], object] = Pipeline,
               llm: Callable[[str, list[dict]], object] | None = None) -> FastAPI:
    """llm=None picks the real Groq call, or the fixed stub reply when LLM_STUB=1 at app creation."""
    stub = llm is None and llm_stub_enabled()
    if llm is None:
        llm = stub_answer if stub else groq_answer

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        app.state.pipeline = pipeline_factory()
        app.state.ready = True
        yield
        app.state.ready = False
        if hasattr(app.state.pipeline, "close"):
            app.state.pipeline.close()

    app = FastAPI(title="RAG Reliability Service", lifespan=lifespan)

    @app.get("/health")
    def health():
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="loading indexes and models")
        return {"status": "ok", "llm_stub": True} if stub else {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="loading indexes and models")
        if not stub and not os.environ.get("GROQ_API_KEY"):
            raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set in the server environment")

        pipeline = app.state.pipeline
        t0 = time.perf_counter()
        candidates = pipeline.retrieve(req.question)
        t1 = time.perf_counter()
        top = pipeline.rerank(req.question, candidates)
        t2 = time.perf_counter()
        try:
            result = llm(req.question, top)
        except Exception as e:  # includes RateLimitExhausted
            raise HTTPException(status_code=502, detail=f"Groq request failed ({type(e).__name__})") from None
        t3 = time.perf_counter()
        if result.text is None:
            # Error kinds only (exception class, or "incomplete reply" for empty/cut-off text), no messages.
            kinds = sorted({"incomplete reply" if err.startswith("attempt") else err.split(":", 1)[0]
                            for err in getattr(result, "errors", [])}) or ["unknown error"]
            raise HTTPException(status_code=502, detail=f"Groq request failed ({', '.join(kinds)})")

        return AskResponse(
            answer=result.text,
            abstained=is_abstention(result.text),
            sources=[Source(page=c["doc_path"], chunk_id=c["chunk_id"], rerank_score=c["score"],
                            snippet=" ".join(c["text"].split())[:SNIPPET_CHARS]) for c in top],
            top_rerank_score=top[0]["score"] if top else None,
            latency_ms={"retrieve": round((t1 - t0) * 1000, 1), "rerank": round((t2 - t1) * 1000, 1),
                        "generate": round((t3 - t2) * 1000, 1), "total": round((t3 - t0) * 1000, 1)},
        )

    return app


app = create_app()
