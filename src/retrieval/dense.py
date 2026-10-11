"""Dense retrieval over chunk files with BAAI/bge-small-en-v1.5 and a local-mode Qdrant store.

Chunks are embedded as-is; queries get the bge retrieval instruction prefix.
Embeddings are normalized and stored with cosine distance. An index is rebuilt only
when the chunks file or the model changes (tracked in data/qdrant/index_meta.json).
"""

import hashlib
import json
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
QDRANT_DIR = ROOT / "data" / "qdrant"
META_PATH = QDRANT_DIR / "index_meta.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32
UPSERT_BATCH = 256


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class DenseRetriever:
    def __init__(
        self,
        chunks_path: Path,
        collection: str,
        model_name: str = MODEL_NAME,
        model: SentenceTransformer | None = None,
        client: QdrantClient | None = None,
    ):
        """Pass `model` and `client` to share them across collections (local-mode Qdrant
        allows only one client per storage folder in a process)."""
        self.chunks_path = Path(chunks_path)
        self.collection = collection
        self.model_name = model_name
        self.model = model or SentenceTransformer(model_name)
        if client is None:
            QDRANT_DIR.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(QDRANT_DIR))
        self.client = client

    # ---------- indexing ----------

    def _fingerprint(self) -> dict:
        return {"chunks_sha256": file_sha256(self.chunks_path), "model": self.model_name}

    def _read_meta(self) -> dict:
        try:
            return json.loads(META_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def is_cached(self) -> bool:
        meta = self._read_meta().get(self.collection)
        if not meta or not self.client.collection_exists(self.collection):
            return False
        count = self.client.count(self.collection, exact=True).count
        return meta["fingerprint"] == self._fingerprint() and count == meta["n_chunks"]

    def build(self, force: bool = False) -> bool:
        """Embed and index the chunks unless a matching index exists. Returns True if (re)built."""
        if not force and self.is_cached():
            return False

        chunks = load_chunks(self.chunks_path)
        vectors = self.model.encode(
            [c["text"] for c in chunks],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
        )
        for start in range(0, len(chunks), UPSERT_BATCH):
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
                    for i in range(start, min(start + UPSERT_BATCH, len(chunks)))
                ],
            )

        meta = self._read_meta()
        meta[self.collection] = {
            "fingerprint": self._fingerprint(),
            "n_chunks": len(chunks),
            "chunks_path": str(self.chunks_path.relative_to(ROOT)),
        }
        META_PATH.write_text(json.dumps(meta, indent=2))
        return True

    # ---------- search ----------

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Top-k chunks for a query: each chunk's payload plus its cosine "score"."""
        vector = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        hits = self.client.query_points(
            self.collection, query=vector.tolist(), limit=k, with_payload=True
        ).points
        return [{**hit.payload, "score": hit.score} for hit in hits]

    def close(self) -> None:
        self.client.close()
