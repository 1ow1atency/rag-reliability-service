# syntax=docker/dockerfile:1
# RAG Reliability Service: CPU-only image with models and indexes baked in at build time.
#
# Build stages:
#   base     - Python 3.11 slim, CPU-only torch, pinned requirements
#   builder  - fetches the pinned docs commit, builds the fixed_512 chunks + dense and BM25
#              indexes (scripts/build_index.py), and downloads the reranker into the model cache
#   runtime  - copies only the code, model cache and indexes; runs as a non-root user, offline
#
# GROQ_API_KEY is never part of the image: pass it at run time (see docker-compose.yml).

ARG PYTHON_IMAGE=python:3.11-slim
ARG TORCH_VERSION=2.14.1

FROM ${PYTHON_IMAGE} AS base
ARG TORCH_VERSION
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf
WORKDIR /app
COPY requirements-service.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
 && pip install -r requirements-service.txt

FROM base AS builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY configs configs
COPY src src
COPY scripts scripts
RUN python scripts/fetch_docs.py \
 && python scripts/build_index.py \
 && python -c "import json; from sentence_transformers import CrossEncoder; \
CrossEncoder(json.load(open('configs/final_retrieval.json'))['rerank']['model'])"

FROM base AS runtime
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/data/processed \
 && chown -R app:app /app/data
COPY --from=builder /opt/hf /opt/hf
COPY configs configs
COPY src src
COPY --from=builder --chown=app:app /app/data/processed/chunks_fixed_512.jsonl \
                                    /app/data/processed/chunks_fixed_512.bm25.pkl /app/data/processed/
# Qdrant local mode writes a lock file in this folder, so it must be owned by the app user.
COPY --from=builder --chown=app:app /app/data/qdrant /app/data/qdrant
# Startup must not download anything.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
# One worker: local-mode Qdrant allows a single process per storage folder.
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
