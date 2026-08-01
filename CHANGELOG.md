## Unreleased (v7.2.0)

### Breaking Changes

- **DeepEval backend**: Xoá `tests/test_rag_deepeval_qwen.py` (Qwen + NIM song song) → thay bằng **`tests/test_rag_deepeval_nim.py`** chỉ dùng **NVIDIA NIM** (`meta/llama-3.3-70b-instruct`). Không còn `QWEN_EVAL_*`, `TEST_EVAL_BACKEND` env vars.
- **Test file rename**: Xoá `test_rag_deepeval_qwen.py` — toàn bộ eval chuyển sang file NIM-only.

### Added

- **Query Transformation (optional)**: Node `query_transform` mới trong LangGraph (bật qua `QUERY_TRANSFORM_ENABLED=true`):
  - **Rewrite**: paraphrase câu hỏi để cụ thể hơn.
  - **Step-back**: sinh câu hỏi tổng quát hơn cho background context.
  - **Decompose**: chia câu hỏi phức tạp thành 2-4 sub-queries.
  - Chạy song song 3 LLM calls qua `RunnableParallel` (Qwen self-host, OpenAI-compatible endpoint). Fallback về query gốc nếu LLM lỗi.
  - Env mới: `QUERY_TRANSFORM_ENABLED`, `QWEN_BASE_URL`, `QWEN_API_KEY`, `QWEN_MODEL`.
- **Hybrid Retrieval (từ v7.1 nhưng bật mặc định)**: BGE-m3 trả cả dense + sparse → Qdrant `RetrievalMode.HYBRID` (kế thừa từ v7.1 commit `3d78643`).
- **DeepEval 5 metrics** (theo reference notebook `example/evaluation_deep_eval.ipynb`):
  - `GEval` (Correctness) — fact-correctness giữa actual vs expected.
  - `FaithfulnessMetric` — hallucination check (actual output có grounded trong context không).
  - `ContextualRelevancyMetric` — relevance của retrieval context với query.
  - `ContextualPrecisionMetric` + `ContextualRecallMetric` (giữ từ v7.0/v7.1).
  - Helper `create_deep_eval_test_cases()` + batch `evaluate()` pattern.
- **Answer Generation trong test**: NIM tự sinh `actual_output` từ retrieved context → mới đo được Correctness/Faithfulness (route `/rag/` chỉ trả documents).
- **Env tuning cho eval**: `TEST_SKIP_ANSWER_GEN`, `TEST_DEEPEVAL_TOP_K`, `TEST_MIN_PASS_RATIO`, `TEST_LOCATOR_STRICT`, `TEST_SKIP_DEEPEVAL`.

### Changed

- **Reranker**: v7.1 đã chuyển `RERANK_PROVIDER` default từ `none` → **`bge`** (local FlagReranker, cần GPU). `RERANK_MODEL=BAAI/bge-reranker-v2-m3`. Legacy `remote` vẫn hỗ trợ.
- **Port & Network**: API port `8001` → **`8005`**, Docker network `test_network` → **`ami-network`** (external).
- **Dockerfile**: Base image `uv:python3.13-bookworm-slim` → **`pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`** (GPU cho local reranker). Quản lý deps bằng `requirements.txt` + `pip install` thay `uv sync`.
- **Dependencies**: Thêm `FlagEmbedding==1.4.0`, `transformers==4.57.1`, `langchain-openai>=0.3.0` (cho Qwen LLM).
- **Volume**: Thêm `hf-cache:/app/.cache/huggingface` cache model reranker (~1.1GB).
- **GPU reservation**: `docker-compose.yml` thêm `deploy.resources.reservations.devices` (NVIDIA all GPUs).
- **API Response**: `SearchResponse.page` + `source` đổi từ required → **optional** (`None` nếu metadata thiếu).
- **Unit tests**: Thêm `tests/unit_tests/test_query_transform_node.py` (8 tests: disabled fallback, LLM success, failure fallback, parse sub-queries, multi-query retrieve dedupe, rerank dùng original query, legacy behaviour).
- **Docs**: Cập nhật toàn bộ 14 file trong `docs/` (CONFIGURATION, SETUP, DEPLOYMENT, ARCHITECTURE, OPERATIONS, TROUBLESHOOTING, EVALUATION, TESTING, DEVELOPMENT, GLOSSARY, HANDOVER_CHECKLIST, API_REFERENCE, README.md root, docs/README.md) cho v7.1/v7.2.

### Tests

- 70+ unit + integration tests pass (bao gồm query_transform + retrieval multi-query).
- DeepEval NIM-only: `ALLOW_NETWORK_TESTS=1 NVIDIA_API_KEY=xxx pytest tests/test_rag_deepeval_nim.py -m qwen -vv`.


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
