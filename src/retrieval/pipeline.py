"""Build retrievers from configs/final_retrieval.json (or the same settings for another chunker)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "final_retrieval.json"


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(Path(path).read_text())


class Resources:
    """Models and the Qdrant client, loaded once and shared by every retriever."""

    def __init__(self, config: dict):
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        from src.retrieval.dense import QDRANT_DIR
        from src.retrieval.rerank import CrossEncoderReranker

        self.embedder = SentenceTransformer(config["dense"]["model"])
        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(QDRANT_DIR))
        self.reranker = CrossEncoderReranker(config["rerank"]["model"])

    def close(self) -> None:
        self.reranker.save()
        self.client.close()


def build_retrievers(config: dict, res: Resources, chunker: str | None = None) -> dict:
    """Dense, BM25, hybrid and reranked retrievers for a chunker (default: the config's)."""
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.rerank import RerankRetriever

    chunker = chunker or config["chunker"]["name"]
    chunks_path = ROOT / "data" / "processed" / f"chunks_{chunker}.jsonl"
    dense = DenseRetriever(chunks_path, f"chunks_{chunker}_bge_small",
                           model_name=config["dense"]["model"], model=res.embedder, client=res.client)
    dense.build()
    bm25 = BM25Retriever(chunks_path)
    bm25.build()
    hybrid = HybridRetriever(dense, bm25, depth=config["hybrid"]["depth_per_retriever"], k=config["hybrid"]["rrf_k"])
    reranked = RerankRetriever(hybrid, res.reranker, depth=config["rerank"]["depth"])
    return {"dense": dense, "bm25": bm25, "hybrid": hybrid, "hybrid+bge-reranker": reranked}
