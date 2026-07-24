# Cấu Hình (Environment Variables) — v7.0.0

> Tất cả env vars đều đọc từ `.env` qua `python-dotenv` + Pydantic Settings.
> Mỗi biến có alias để backward-compat. Default an toàn trong code.
>
> **Repo v7 chỉ retrieval & search** — biến ingest / chunking / migration đã
> được loại bỏ. Embedding + rerank delegate tới HTTP endpoint ngoài (Colab ngrok
> / server GPU riêng). Xem [API_REFERENCE.md](API_REFERENCE.md) §8.

## 1. Embedding (Dense — remote BGE-m3)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `EMBEDDING_PROVIDER` | — | `remote` | Provider: chỉ `remote` ở v7. Local `bge` (FlagEmbedding) đã loại bỏ |
| `EMBEDDING_BASE_URL` | `AU_EMBED_BASE_URL` | `""` | Base URL của remote BGE-m3 server (Colab ngrok / self-hosted). Server expose `POST /embed` trả `{"dense_vecs": [[float,...],...]}`. **Bắt buộc** khi provider = `remote` |
| `EMBEDDING_TIMEOUT` | — | `60` | Timeout (giây) khi gọi remote `/embed` endpoint |

> Sparse embedding đã loại bỏ (remote endpoint không trả sparse). Retrieval
> dùng dense-only (`RetrievalMode.DENSE`). Không có hybrid search / RRF / DBSF.

## 2. Retrieval

| Biến | Default | Mô tả |
|---|---|---|
| `RETRIEVAL_K` | `40` | Số document lấy từ dense search (trước rerank) |
| `RETRIEVAL_K_RETRY` | `100` | Số document lấy khi `retry_count > 0` (retry logic trong graph) |

> Không còn `FUSION_ALGORITHM` / `hybrid_fusion` — fusion không còn dùng (dense-only).

## 3. Reranker (remote BGE-reranker-v2-m3, optional)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `RERANK_PROVIDER` | — | `none` | **`none` = passthrough (không gọi reranker)**; `remote` = HTTP tới `RERANK_BASE_URL`. Local `bge` (FlagEmbedding) đã loại bỏ |
| `RERANK_BASE_URL` | `AU_RERANK_BASE_URL` | `""` | Base URL của remote reranker server (cùng Colab server với embed). Server expose `POST /rerank` trả `{"results": [{"index": int, "document": str, "score": float},...]}` |
| `RERANK_TOP_K` | — | `5` | Số document giữ lại sau rerank |
| `RERANK_TIMEOUT` | — | `60` | Timeout (giây) khi gọi remote `/rerank` endpoint |

> Các provider reranker `cohere` / `flashrank` / `bge` (local) đã bị loại bỏ ở
> v7 — không có dependency nào cho chúng.

## 4. Qdrant

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `QDRANT_URL` | — | `http://localhost` | URL Qdrant (⚠️ Trong Docker container, `localhost` trỏ về chính container đó — dùng `http://qdrant` hoặc `http://host.docker.internal`) |
| `QDRANT_PORT` | — | `6333` | REST API port |
| `QDRANT_API_KEY` | `qdrant_cloud_api_key` | `(none)` | API key Qdrant Cloud |
| `QDRANT_COLLECTION_NAME` | `QDRANT_COLLECTION`, `qdrant_collection` | `documents` | Tên collection dùng cho `/readyz` và default route |

> **Collection do hệ thống ngoài dựng.** Repo này không có endpoint create
> collection. Phải chắc chắn collection `QDRANT_COLLECTION_NAME` đã tồn tại
> trên Qdrant trước khi `/readyz` trả về 200.

## 5. DeepEval — NVIDIA NIM (test only)

| Biến | Default | Mô tả |
|---|---|---|
| `NVIDIA_API_KEY` | `""` | API key cho NVIDIA NIM |
| `NVIDIA_EVAL_MODEL` | `meta/llama-3.3-70b-instruct` | Model eval trên NVIDIA NIM |
| `NVIDIA_EVAL_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Endpoint NVIDIA NIM |
| `NVIDIA_EVAL_RPS` | `30` | Giới hạn requests/sec cho rate limiter |

## 6. DeepEval — Qwen self-host (test only)

| Biến | Default | Mô tả |
|---|---|---|
| `QWEN_EVAL_BASE_URL` | `http://localhost:8000/v1` | Endpoint OpenAI-compatible của Qwen |
| `QWEN_EVAL_API_KEY` | `""` | API key (nếu cần) |
| `QWEN_EVAL_MODEL` | `qwen` | Tên model Qwen |

> `QWEN_EVAL_THINKING` đã bỏ ở v7 — `QwenEvalLLM` hardcode `thinking=False`
> trong request body. Nếu cần bật/tắt runtime, sửa trực tiếp
> `tests/test_rag_deepeval_qwen.py::QwenEvalLLM.generate`.

## 7. Tổng hợp default `RERANK_PROVIDER` theo use case

| Use case | `RERANK_PROVIDER` | Trade-off |
|---|---|---|
| Smoke test nhanh, không cần precision cao | `none` | Passthrough, không gọi remote reranker -> container API nhẹ, ~512MB-1GB RAM |
| Production cần score chính xác | `remote` | Gọi BGE-reranker-v2-m3 trên Colab/GPU server, tăng precision + ~100ms/request (RTT tới remote) |

## 8. Migration / Chunking (LOẠI BỎ ở v7)

Các biến `INPUT_DIR`, `MIGRATE_*`, `CHUNK_*`, `ENABLE_LLM_ENRICH`, `LLM_*`,
`BACKEND_HOST`, `BACKEND_PORT`, `EVAL_RPM`, `EVAL_TPM`, `REC` đã bị loại bỏ
trong `template.env`. Migration scripts và ingestion logic thuộc về repo hệ
thống ngoài quản lý Qdrant.

## 9. Backward-compat Aliases (giữ backward-compat cho `.env` cũ)

`Config` vẫn nhận các alias cũ để tránh break `.env` legacy trong qúa trình
upgrade:

| Alias mới | Alias cũ (backward-compat) |
|---|---|
| `EMBEDDING_BASE_URL` | `AU_EMBED_BASE_URL` |
| `RERANK_BASE_URL` | `AU_RERANK_BASE_URL` |
| `QDRANT_COLLECTION_NAME` | `QDRANT_COLLECTION`, `qdrant_collection` |
| `QDRANT_API_KEY` | `qdrant_cloud_api_key` |

> Nếu `.env` của bạn còn chứa `AU_EMBED_BASE_URL`, `AU_RERANK_BASE_URL` → vẫn
> chạy OK. Nhưng khuyến nghị trim về tên mới để đồng bộ với `template.env`.

Các biến đã bị **xoá hoàn toàn** ở v7 (không còn backward-compat):
`EMBEDDING_PROVIDER=bge` (local FlagEmbedding), `EMBEDDING_MODEL`,
`EMBEDDING_SIZE`, `SPARSE_MODEL`, `FUSION_ALGORITHM` / `hybrid_fusion`,
`RERANK_PROVIDER=bge` (local), `RERANK_MODEL`, `RERANK_API_KEY`,
`COHERE_API_KEY`, `OPENAI_API_KEY`, `RRF_K`, `DBSF_WINDOW`,
`QWEN_EVAL_THINKING`, `EVAL_RPM`, `EVAL_TPM`, `BACKEND_*`,
`RECREATE_COLLECTION`, `EMBEDDING_API_KEY`.

## 10. Model Config Reference

| Model | HuggingFace ID | Loại | dim | Chạy ở đâu |
|---|---|---|---|---|
| BGE-m3 | `BAAI/bge-m3` | Dense (qua `/embed`) | 1024 | Remote server (Colab ngrok / GPU server) — repo chỉ gọi HTTP |
| BGE-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | Reranker (qua `/rerank`) | — | Remote server (cùng Colab) |
| Qwen 2.5 (16B) | local serve | Eval LLM (test only) | — | Self-host |
| Llama 3.3 70B | NVIDIA NIM | Eval LLM (test only) | — | NVIDIA NIM |
