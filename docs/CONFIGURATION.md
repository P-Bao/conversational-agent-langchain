# Cấu Hình (Environment Variables) — v7.1.0

> Tất cả biến đọc từ `.env` qua Pydantic Settings (`src/agent/utils/config.py`).
> Mỗi biến có alias để backward-compat. Default an toàn trong code.
>
> **Repo v7.1 chỉ retrieval & search** — tất cả biến ingest / chunking đã
> bị loại bỏ. Embedding gọi HTTP endpoint ngoài (embedding-server),
> reranker chạy local qua FlagEmbedding.

## 1. Embedding (Dense + Sparse — remote BGE-m3)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `EMBEDDING_PROVIDER` | — | `remote` | Chỉ `remote` từ v7. Local `bge` (FlagEmbedding) đã loại bỏ |
| `EMBEDDING_BASE_URL` | `AU_EMBED_BASE_URL` | `""` | Base URL của remote BGE-m3 server. Server expose `POST /embed` trả `{"dense_vecs": [[float,...],...], "sparse_vecs": [{"indices": [...], "values": [...]}, ...]}`. **Bắt buộc** khi provider = `remote` |
| `EMBEDDING_API_KEY` | `AU_EMBED_API_KEY` | `none` | API key cho remote server (nếu bật auth). Server verify header `Authorization: Bearer <key>` |
| `EMBEDDING_RETURN_SPARSE` | — | `true` | Trả về cả dense + sparse vectors từ remote server |
| `EMBEDDING_TIMEOUT` | — | `60` | Timeout (giây) khi gọi remote `/embed` endpoint |

> BGE-m3 trả cả dense (1024-dim) **và** sparse vectors (BM25-like). Repo dùng
> **hybrid retrieval** (Qdrant `RetrievalMode.HYBRID`) — v7.1 không còn dense-only.

## 2. Retrieval

| Biến | Default | Mô tả |
|---|---|---|
| `RETRIEVAL_K` | `40` | Số document lấy từ Qdrant (trước rerank) |
| `RETRIEVAL_K_RETRY` | `100` | Số document lấy khi `retry_count > 0` |

> Không còn `FUSION_ALGORITHM` / `hybrid_fusion` — fusion không còn dùng.
> Qdrant tự quản qua `RetrievalMode.HYBRID`.

## 3. Query Transformation (Qwen self-host — rewrite + step-back + decompose)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `QUERY_TRANSFORM_ENABLED` | — | `false` | Bật = 1 (`true`) → thêm node `query_transform` trước `retriever` trong LangGraph. Tắt = giữ nguyên pipeline cũ |
| `QWEN_BASE_URL` | — | `http://localhost:8000/v1` | OpenAI-compatible endpoint của Qwen self-host |
| `QWEN_API_KEY` | — | `dummy` | API key cho Qwen (self-host thường để `dummy`) |
| `QWEN_MODEL` | — | `qwen` | Tên model trong Qwen server |

> **Impact:** Khi bật, mỗi query gốc số sinh thêm **3 biến thể** (rewritten,
> step-back, decompose 2-4 sub-queries) rồi retrieve tuần tự tất cả 5-7 query.
> Embedding server bị gọi ~5-7 lần thay vì 1 lần. Rerank vẫn dùng câu hỏi gốc.

## 4. Reranker (Local BGE-reranker-v2-m3 via FlagEmbedding)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `RERANK_PROVIDER` | — | `bge` | `bge` = local FlagReranker (default); `remote` = legacy HTTP endpoint; `none` = passthrough |
| `RERANK_MODEL` | `AU_RERANK_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Model HuggingFace cho local FlagReranker. Tải lần đầu ~1.1GB vào cache |
| `RERANK_TOP_K` | — | `5` | Số document giữ lại sau rerank |
| `RERANK_BASE_URL` | `AU_RERANK_BASE_URL` | — | **Legacy only** — chỉ dùng khi `RERANK_PROVIDER=remote` |
| `RERANK_TIMEOUT` | — | `60` | Timeout (giây) cho remote `/rerank` (legacy) |

> Các provider `cohere` / `flashrank` đã bị loại bỏ hoàn toàn.

## 5. Qdrant

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `QDRANT_URL` | — | `http://localhost` | URL Qdrant (trong container: `http://qdrant` hoặc `http://host.docker.internal`) |
| `QDRANT_PORT` | — | `6333` | REST API port |
| `QDRANT_API_KEY` | `qdrant_cloud_api_key` | `(none)` | API key Qdrant Cloud |
| `QDRANT_COLLECTION_NAME` | `QDRANT_COLLECTION`, `qdrant_collection` | `documents` | Tên collection dùng cho `/readyz` và default route |

> **Collection do hệ thống ingestion, bên ngoài quản lý** — repo này không
> có endpoint tạo collection.

## 6. DeepEval — NVIDIA NIM (test only)

| Biến | Default | Mô tả |
|---|---|---|
| `NVIDIA_API_KEY` | `""` | API key cho NVIDIA NIM |
| `NVIDIA_EVAL_MODEL` | `meta/llama-3.3-70b-instruct` | Model eval trên NVIDIA NIM |
| `NVIDIA_EVAL_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Endpoint NVIDIA NIM |
| `NVIDIA_EVAL_RPS` | `30` | Giới hạn requests/sec cho rate limiter |
| `TEST_EVAL_BACKEND` | `""` | Từ v7.1, chỉ hỗ trợ `nvidia`. Không set → auto-detect NVIDIA |
| `RAG_API_URL` | `http://localhost:8005` | Base URL API Docker container. **Đôi khi phải là `8005` (port mới)** |
| `TEST_SKIP_ANSWER_GEN` | `0` | `1` = bỏ qua answering-generation step (dùng top-1 chunk làm actual_output) |
| `TEST_LOCATOR_STRICT` | `0` | `1` = fail test khi locator/context mismatch |
| `TEST_SKIP_DEEPEVAL` | `0` | `1` = chỉ kiểm tra retrieval/locator, bỏ qua DeepEval metrics |
| `TEST_MIN_PASS_RATIO` | `0.7` | Tỷ lệ câu hỏi tối thiểu phải pass để assert |
| `TEST_DEEPEVAL_TOP_K` | `5` | Số context doc đưa vào `LLMTestCase.retrieval_context` |

> **QWEN_EVAL_* biến đã bỏ** — eval chỉ dùng NVIDIA NIM (chia sẻ cùng judge).

## 7. Tổng hợp default `RERANK_PROVIDER` theo use case

| Use case | `RERANK_PROVIDER` | Trade-off |
|---|---|---|
| Production (default) | `bge` | Local FlagReranker — precision tốt nhất, cần GPU (~1.4GB VRAM) |
| Smoke test nhanh | `none` | Passthrough, không load model — container nhẹ ~512MB-1GB RAM |
| Legacy remote server | `remote` | HTTP endpoint ngoài — chỉ nếu không có GPU local |

## 8. Migration / Chunking (LOẠI BỎ từ v7.0)

Cac biến `INPUT_DIR`, `MIGRATE_*`, `CHUNK_*`, `ENABLE_LLM_ENRICH`, `LLM_*`,
`BACKEND_HOST`, `BACKEND_PORT`, `EVAL_RPM`, `EVAL_TPM`, `REC` đã bị loại bỏ.
Migration scripts thuộc hệ thống ingestion bên ngoài quản lý Qdrant.

## 9. Backward-compat Aliases

| Alias mới | Alias cũ (backward-compat) |
|---|---|
| `EMBEDDING_BASE_URL` | `AU_EMBED_BASE_URL` |
| `EMBEDDING_API_KEY` | `AU_EMBED_API_KEY` |
| `QDRANT_COLLECTION_NAME` | `QDRANT_COLLECTION`, `qdrant_collection` |
| `QDRANT_API_KEY` | `qdrant_cloud_api_key` |
| `RERANK_BASE_URL` | `AU_RERANK_BASE_URL` (legacy only) |
| `RERANK_MODEL` | `AU_RERANK_MODEL_NAME` |

> Nếu `.env` còn chứa `AU_*` → vẫn chạy OK. Nhưng khuyến nghị trim về tên
> mới để đồng bộ với `template.env`.

Các biến đã **xoá hoàn toàn**:
`EMBEDDING_PROVIDER=bge`, `EMBEDDING_MODEL`, `EMBEDDING_SIZE`, `SPARSE_MODEL`,
`FUSION_ALGORITHM` / `hybrid_fusion`, `RERANK_PROVIDER=cohere`,
`COHERE_API_KEY`, `OPENAI_API_KEY`, `RRF_K`, `DBSF_WINDOW`,
`QWEN_EVAL_BASE_URL`, `QWEN_EVAL_API_KEY`, `QWEN_EVAL_MODEL`,
`QWEN_EVAL_THINKING`, `EVAL_RPM`, `EVAL_TPM`, `BACKEND_*`,
`RECREATE_COLLECTION`.

## 10. Model Config Reference

| Model | HuggingFace ID | Loại | Chạy ở đâu |
|---|---|---|---|
| BGE-m3 | `BAAI/bge-m3` | Dense 1024-dim + Sparse | Remote server (embedding-server container, port 8008) — chỉ gọi HTTP |
| BGE-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | Cross-attention reranker | **Local** (FlagEmbedding) — model load khi gọi rerank lần đầu |
| Qwen (self host) | — | Query transformation LLM | Self-host Qwen server (OpenAI-compatible, port 8000) |
| Llama 3.3 70B | NVIDIA NIM | Eval LLM (test only) | NVIDIA NIM API |
