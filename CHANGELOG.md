## 7.1.0 (2026-07-24)

### Breaking Changes

- **Embedding chuyển sang remote HTTP**: Repo không còn chạy local BGE-m3 (FlagEmbedding + torch). `EMBEDDING_PROVIDER` default đổi từ `bge` → `remote` (giá trị `bge` đã bỏ hoàn toàn). Thêm `EMBEDDING_BASE_URL` — base URL của server BGE-m3 HTTP (Colab ngrok hoặc server GPU tự host), endpoint `POST /embed` → `{"dense_vecs": [...]}`.
- **Sparse embedding đã bỏ**: Retrieval dense-only (`RetrievalMode.DENSE`). Không còn `bge-m3-sparse` named vector, không còn hybrid search, không còn RRF/DBSF/fusion.
- **Reranker chuyển sang remote HTTP**: `RERANK_PROVIDER` hỗ trợ `none` (default) và `remote` (giá trị `bge` đã bỏ). Thêm `RERANK_BASE_URL` — endpoint `POST /rerank` → `{"results": [{"index","document","score"}]}`.
- **Env vars đã bỏ**: `EMBEDDING_MODEL`, `EMBEDDING_SIZE`, `SPARSE_MODEL`, `FUSION_ALGORITHM`/`hybrid_fusion`, `RERANK_MODEL`.

### Added

- **Env mới**: `EMBEDDING_BASE_URL`, `RERANK_BASE_URL`, `EMBEDDING_TIMEOUT` (default 60s), `RERANK_TIMEOUT` (default 60s).
- **Docker image CUDA-free & model-cache-free**: Xoá `FlagEmbedding`, torch, transformers, sentence-transformers, scikit-learn, scipy, pandas, nvidia-*, triton khỏi `uv.lock`. `Dockerfile` bỏ `HF_HOME`/`UV_HTTP_TIMEOUT`. `docker-compose.yml` bỏ `hf_cache` volume + `extra_hosts`.
- **Makefile `docker-clean`**: `docker compose down --remove-orphans -v && docker system prune -a --volumes -f` — dọn image/volume khi build fail.
- **httpx>=0.28.1** thay `FlagEmbedding` trong `pyproject.toml`.
- **Source**: `BGEM3RemoteEmbeddings` + `get_embedding_model(cfg)` (`src/agent/utils/embeddings.py`), `rerank_with_remote(...)` + `get_reranker(cfg, *, top_k=...)` (`src/agent/utils/reranker.py`), `get_retriever(k, *, cfg=None)` dense-only (`src/agent/utils/retriever.py`).
- **Reference server**: notebook `rag_test_bge_m3_reranker_ngrok.ipynb` (chạy trên Colab T4, ngrok URL) ngoài repo.

## 7.0.0 (2026-07-23)

### Breaking Changes

- **Ingestion đã chuyển hệ thống ngoài**: Repo này không còn endpoint `/collection/create`, `/embeddings/documents`, `/embeddings/string`, `/embeddings/delete`. Không còn `python -m agent.scripts.migrate_dump_to_qdrant`. Qdrant collection phải được dựng bởi hệ thống ingestion ngoài trước khi API phục vụ.
- **Frontend (Streamlit) tách repo**: `frontend/`, `Dockerfile.frontend` đã được loại bỏ khỏi repo này. Liên hệ team frontend để biết repo Streamlit mới.
- **Cohere + FlashRank rerankers đã bỏ**: Chỉ hỗ trợ provider `none` (passthrough) và `bge` (BGE-reranker v2-m3). Default đổi từ `bge` → `none`.
- **OpenAI embedding provider đã bỏ**: Chỉ hỗ trợ `bge` provider.

### Added

- **Health endpoints**: `GET /healthz` (liveness) + `GET /readyz` (Qdrant + collection readiness).
- **Docker Compose healthcheck**: dùng `/healthz` qua Python `urllib`.
- **Makefile cleanup**: xoá target `start_frontend`.
- **Configuration**: xoá legacy env vars (COHERE_API_KEY, OPENAI_API_KEY, QWEN_EVAL_THINKING, EMBEDDING_BASE_URL, RRF_K, DBSF_WINDOW...).

### Tests

- 61/61 unit + integration tests pass.
- DeepEval Qwen / NVIDIA NIM chạy qua `TestClient.post('/rag/', ...)`.

## 6.0.0 (2026-07-22)

### Breaking Changes

- **Retrieval-Only**: Backend RAG API `/rag` now returns `RetrievalResponse` (relevant documents and query) instead of generating answers via LLM.
- **BGE-m3 Embeddings**: Switched default dense and sparse embeddings to `BAAI/bge-m3` (1024-dim dense + `bge-m3-sparse` named vector).
- **BGE Reranker**: Integrated `BAAI/bge-reranker-v2-m3` cross-encoder for reranking retrieved candidates.
- **Qwen DeepEval**: Replaced legacy Gemini DeepEval test suite with `test_rag_deepeval_qwen.py` for context precision/recall evaluation.
- **Legacy Cleanup**: Removed generation, grading, rewrite nodes, prompt files, and legacy VCR tests.

## 5.5.0 (2025-12-05)

### Feat

- **Cohere**: Reverting to correct Cohere impl for better grounded generation

## 5.4.0 (2025-12-04)

### Feat

- **Reranking**: Adding reranking with flashrank or cohere (#133)

### Perf

- **Graph**: Performance Optimization for Graph (#134)

## 5.3.0 (2025-11-30)

### Feat

- **Graph**: Retry logik if question can not be answered (#131)

### Refactor

- **Warnings**: Supress warnings that can not be fixed
- **Frontend**: Refactor to httpx
- **Backend**: Fixing minor stuff
- **Frontend**: Refactoring the Frontend into client and assistant

## 5.2.0 (2025-11-30)

### Feat

- **Frontend**: Moving frontend to own uv project

## 5.1.0 (2025-11-30)

### Feat

- **Graph**: Defaults to gemini

## 5.0.0 (2025-11-30)

### Feat

- **Upload**: Adding upload sidebar in frontend

## 4.0.0 (2025-11-30)

### Feat

- **Langchain-1.0**: Migration to Langchain & Langgraph 1.0, fixing docker problems

### Refactor

- **Embeddings**: Removing unnessary calls and cleanup
- **Config**: Converting to Pydantic Settings and removing yml config files (#128)

## 3.2.1 (2025-11-22)

### Fix

- **LiteLLM**: Fixing deprectation warning.s

## 3.2.0 (2025-11-22)

### Feat

- **Frontend**: Reworking frontend to support streaming

## 3.1.0 (2025-11-22)

### Feat

- **Updates**: Updating Frontend and Backend

## 3.0.0 (2025-06-15)

## 2.5.0 (2023-12-26)

## 2.4.2 (2023-12-07)

## 2.4.1 (2023-11-21)

## 2.4.0 (2023-11-10)

## 2.3.0 (2023-10-13)

## 2.2.7 (2023-09-16)

## 2.2.6 (2023-09-16)

## 2.2.5 (2023-09-15)

## 2.2.4 (2023-09-13)

## 2.2.3 (2023-09-12)

## 2.2.2 (2023-09-10)

## 2.2.1 (2023-09-04)

## 2.2.0 (2023-09-04)

## 2.1.0 (2023-08-04)

## 2.0.1 (2023-07-17)

## 2.0.0 (2023-07-15)

## 1.0.2 (2023-06-06)

## 1.0.1 (2023-05-30)

## 1.0.0 (2023-05-19)
