"""Build everything the service needs from data/raw: the chunk file, the dense index and the BM25 index.

Usage: python scripts/build_index.py

Uses the chunker and models in configs/final_retrieval.json (fixed_512). Chunks are written in the
same format as src/ingest/chunking.py, so an unchanged corpus gives a byte-identical chunk file and
the existing indexes are reused rather than rebuilt. Run scripts/fetch_docs.py first.
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingest.chunking import CHUNKERS, OUT_DIR, Tokenizer, load_docs  # noqa: E402
from src.ingest.stats import DOCS_ROOT  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.pipeline import load_config  # noqa: E402


def main() -> None:
    config = load_config()
    chunker = config["chunker"]["name"]
    chunks_path = ROOT / config["chunker"]["chunks_file"]
    if not DOCS_ROOT.exists():
        raise SystemExit(f"{DOCS_ROOT.relative_to(ROOT)} not found; run scripts/fetch_docs.py first")

    start = time.perf_counter()
    tok = Tokenizer(config["chunker"]["tokenizer"])
    docs = load_docs()
    chunks = [c for doc in docs for c in CHUNKERS[chunker](doc, tok)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_text = "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks)
    unchanged = chunks_path.exists() and chunks_path.read_text(encoding="utf-8") == new_text
    if not unchanged:
        chunks_path.write_text(new_text, encoding="utf-8")
    print(f"chunks: {len(chunks)} {chunker} chunks from {len(docs)} docs -> "
          f"{chunks_path.relative_to(ROOT)} ({'unchanged' if unchanged else 'written'})")

    dense = DenseRetriever(chunks_path, f"chunks_{chunker}_bge_small", model_name=config["dense"]["model"])
    print(f"dense: collection {dense.collection} {'built' if dense.build() else 'already up to date'}")
    dense.close()
    bm25 = BM25Retriever(chunks_path)
    print(f"bm25:  {bm25.cache_path.relative_to(ROOT)} {'built' if bm25.build() else 'already up to date'}")
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
