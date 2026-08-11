# Cấu Hình (Environment Variables) — v8.x

> Tất cả env vars đều đọc từ `.env` qua `python-dotenv` + Pydantic Settings.
> Mỗi biến có alias để backward-compat. Default an toàn trong code.
>
> **Repo retrieval-only** — biến ingest / chunking / migration đã được
> loại bỏ. Embedding + rerank delegate tới HTTP endpoint ngoài (embedding-server
> `:8008` cho BGE-m3 dense+sparse, rerank-server `:8010` cho BGE-reranker-v2-m3).
> Xem [API_REFERENCE.md](API_REFERENCE.md) §8.

## 1. Embedding (Dense + Sparse — remote BGE-m3 qua embedding-server)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `EMBEDDING_PROVIDER` | — | `remote` | Provider: chỉ `remote`. Local `bge` (FlagEmbedding) đã loại bỏ. |
| `EMBEDDING_BASE_URL` | `AU_EMBED_BASE_URL` | `""` | Base URL của embedding-server (vd `http://bge-m3-embed:8008`). Expose `POST /embed` trả `{"dense_vecs", "sparse_vecs"}`. **Bắt buộc** khi `EMBEDDING_PROVIDER=remote`. |
| `EMBEDDING_API_KEY` | `AU_EMBED_API_KEY` | `""` | Bearer token gửi qua header `Authorization: Bearer <key>` (khớp với `BGE_API_KEY` trong embedding-server). Để trống nếu server không bật auth. |
| `EMBEDDING_TIMEOUT` | — | `60` | Timeout (giây) khi gọi remote `/embed` endpoint. |

> Retrieval dùng **hybrid dense+sparse** (`RetrievalMode.HYBRID`) qua Qdrant
> named vectors `dense` + `sparse`. Collection phải được dựng bởi hệ thống
> ingestion ngoài với cùng schema.

## 2. Retrieval

| Biến | Default | Mô tả |
|---|---|---|
| `RETRIEVAL_K` | `40` | Số document lấy từ dense search (trước rerank) |
| `RETRIEVAL_K_RETRY` | `100` | Số document lấy khi `retry_count > 0` (retry logic trong graph) |

> Fusion dense+sparse xử lý bởi Qdrant (`RetrievalMode.HYBRID`) — không cần
> `FUSION_ALGORITHM` / `hybrid_fusion` trong config repo này.

## 3. Reranker (3 provider, mặc định `remote` qua rerank-server :8010)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `RERANK_PROVIDER` | — | `remote` | **`remote`** = HTTP tới `RERANK_BASE_URL` (rerank-server :8010); **`bge`** = local FlagEmbedding fallback (cần GPU); **`none`** = passthrough `docs[:top_k]`. |
| `RERANK_BASE_URL` | `AU_RERANK_BASE_URL` | `""` | Base URL của rerank server (vd `http://127.0.0.1:8010`). **Bắt buộc** khi `RERANK_PROVIDER=remote`. |
| `RERANK_MIN_SCORE` | — | `0.0` | Ngưỡng lọc doc theo score (gửi qua payload `min_score` **chỉ khi > 0**). `0.0` = giữ tất cả (không gửi — tránh loại doc có raw logit âm). |
| `RERANK_TOP_K` | — | `5` | Số document giữ lại sau rerank (fallback khi client không gửi `top_k`). |
| `RERANK_MODEL` | `AU_RERANK_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Model cho fallback `bge` local. |
| `RERANK_TIMEOUT` | — | `60` | Timeout (giây) khi gọi remote `/rerank` endpoint. |

**Contract rerank-server (`POST {base_url}/rerank`):**

Request:
```json
{"query": "...", "documents": ["...", "..."], "top_k": 5, "min_score": 0.0}
```
Response:
```json
{"scores": [0.9978, 0.0018], "ranked_indices": [0]}
```
- `scores` = điểm (đã normalize 0-1) theo thứ tự input documents.
- `ranked_indices` = index đã sort giảm dần + áp `top_k` + `min_score`.
- Có fallback tương thích ngược với contract cũ `{"results": [{index, score}]}` (server Colab).

> **Fail-fast**: Khi `remote` server lỗi / timeout → API raise 5xx. Không tự
> động chuyển sang local `bge`. Nếu cần fallback bền vững, set cấu hình
> `RERANK_PROVIDER=none` trong thời gian server down.

> Provider `cohere` / `flashrank` đã bị loại bỏ — không có dependency nào.

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
| Production (rerank server `:8010` online) | `remote` (default) | Gọi BGE-reranker-v2-m3 qua HTTP; container API không tải model; ~100ms RTT tới server |
| Smoke test nhanh, không cần precision | `none` | Passthrough `docs[:top_k]`; container API nhẹ nhất |
| Self-host rerank trong cùng container (cần GPU) | `bge` | Local FlagEmbedding; tăng precision, nhưng tốn ~2.2GB RAM + GPU |

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
| BGE-m3 | `BAAI/bge-m3` | Dense + sparse (qua `/embed`) | 1024 | Embedding-server `:8008` (container GPU ngoài) |
| BGE-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | Reranker (qua `/rerank`) | — | Rerank-server `:8010` (container GPU ngoài) |
| Qwen 2.5 (16B) | local serve | Eval LLM (test only) | — | Self-host |
| Llama 3.3 70B | NVIDIA NIM | Eval LLM (test only) | — | NVIDIA NIM |
