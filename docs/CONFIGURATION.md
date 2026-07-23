# Cấu Hình (Environment Variables) — v7.0.0

> Tất cả env vars đều đọc từ `.env` qua `python-dotenv` + Pydantic Settings.
> Mỗi biến có alias để backward-compat. Default an toàn trong code.
>
> **Repo v7 chỉ retrieval & search** — biến ingest / chunking / migration đã
> được loại bỏ. Xem [API_REFERENCE.md](API_REFERENCE.md) §8.

## 1. Embedding (Dense)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `EMBEDDING_PROVIDER` | — | `bge` | Provider: chỉ `bge` (FlagEmbedding) ở v7 |
| `EMBEDDING_MODEL` | `AU_EMBED_MODEL_NAME` | `BAAI/bge-m3` | Tên model dense trên HuggingFace |
| `EMBEDDING_SIZE` | `AU_EMBED_DIMENSION` | `1024` | Dense vector dimension. BGE-m3 = 1024 |

## 2. Embedding (Sparse)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `SPARSE_MODEL` | `AU_SPARSE_MODEL_NAME` | `BAAI/bge-m3` | Model cho sparse lexical weights. Tách env với dense để swap linh hoạt. Mặc định cùng model với dense → 1 instance FlagEmbed chia sẻ. Nếu đặt khác, load model riêng (tốn RAM gấp đôi). |

## 3. Retrieval

| Biến | Default | Mô tả |
|---|---|---|
| `RETRIEVAL_K` | `40` | Số document lấy từ hybrid search (trước rerank) |
| `RETRIEVAL_K_RETRY` | `100` | Số document lấy khi `retry_count > 0` (retry logic trong graph) |
| `FUSION_ALGORITHM` (alias `hybrid_fusion`) | `rrf` | Thuật toán fusion: `rrf` (Reciprocal Rank Fusion) hoặc `dbsf` (Distribution-Based Score Fusion) |

> Thuật toán fusion do Qdrant xử lý server-side. Để tinh chỉnh `rrf_k` /
> `dbsf_window`, dùng Qdrant REST API trực tiếp — repo này không expose env riêng.

## 4. Reranker

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `RERANK_PROVIDER` | — | `none` | **`none` = passthrough (không load reranker)**; `bge` = local BGE-reranker-v2-m3 |
| `RERANK_MODEL` | `AU_RERANK_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Model reranker trên HuggingFace |
| `RERANK_TOP_K` | — | `5` | Số document giữ lại sau rerank |

> Các provider reranker `cohere` / `flashrank` đã bị loại bỏ ở v7 — không có
> dependency nào cho chúng.

## 5. Qdrant

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `QDRANT_URL` | — | `http://qdrant` | URL Qdrant (⚠️ Trong Docker container, `localhost` trỏ về chính container đó — dùng `http://qdrant` hoặc `http://host.docker.internal`) |
| `QDRANT_PORT` | — | `6333` | REST API port |
| `QDRANT_API_KEY` | `qdrant_cloud_api_key` | `(none)` | API key Qdrant Cloud |
| `QDRANT_COLLECTION_NAME` | `QDRANT_COLLECTION`, `qdrant_collection` | `default` | Tên collection dùng cho `/readyz` và default route |

> **Collection do hệ thống ngoài dựng.** Repo này không có endpoint create
> collection. Phải chắc chắn collection `QDRANT_COLLECTION_NAME` đã tồn tại
> trên Qdrant trước khi `/readyz` trả về 200.

## 6. DeepEval — NVIDIA NIM (test only)

| Biến | Default | Mô tả |
|---|---|---|
| `NVIDIA_API_KEY` | `""` | API key cho NVIDIA NIM |
| `NVIDIA_EVAL_MODEL` | `meta/llama-3.3-70b-instruct` | Model eval trên NVIDIA NIM |
| `NVIDIA_EVAL_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Endpoint NVIDIA NIM |
| `NVIDIA_EVAL_RPS` | `30` | Giới hạn requests/sec cho rate limiter |

## 7. DeepEval — Qwen self-host (test only)

| Biến | Default | Mô tả |
|---|---|---|
| `QWEN_EVAL_BASE_URL` | `http://localhost:8000/v1` | Endpoint OpenAI-compatible của Qwen |
| `QWEN_EVAL_API_KEY` | `""` | API key (nếu cần) |
| `QWEN_EVAL_MODEL` | `qwen` | Tên model Qwen |

> `QWEN_EVAL_THINKING` đã bỏ ở v7 — `QwenEvalLLM` hardcode `thinking=False`
> trong request body. Nếu cần bật/tắt runtime, sửa trực tiếp
> `tests/test_rag_deepeval_qwen.py::QwenEvalLLM.generate`.

## 8. Tổng hợp default `RERANK_PROVIDER` theo use case

| Use case | `RERANK_PROVIDER` | Trade-off |
|---|---|---|
| Smoke test nhanh, không cần precision cao | `none` | Passthrough, không load reranker model -> tiết kiệm ~2.2GB RAM |
| Production cần score chính xác | `bge` | Load BGE-reranker-v2-m3, tăng precision nhưng tốn RAM + ~100ms/request |

## 9. Migration / Chunking (LOẠI BỎ ở v7)

Các biến `INPUT_DIR`, `MIGRATE_*`, `CHUNK_*`, `ENABLE_LLM_ENRICH`, `LLM_*`,
`BACKEND_HOST`, `BACKEND_PORT`, `EVAL_RPM`, `EVAL_TPM`, `REC` đã bị loại bỏ
trong `template.env`. Migration scripts và ingestion logic thuộc về repo hệ
thống ngoài quản lý Qdrant.

## 10. Backward-compat Aliases (giữ backward-compat cho `.env` cũ)

`Config` vẫn nhận các alias cũ để tránh break `.env` legacy trong qúa trình
upgrade:

| Alias mới | Alias cũ (backward-compat) |
|---|---|
| `EMBEDDING_MODEL` | `AU_EMBED_MODEL_NAME`, `AU_EMBED_MODEL` |
| `EMBEDDING_SIZE` | `AU_EMBED_DIMENSION` |
| `SPARSE_MODEL` | `AU_SPARSE_MODEL_NAME`, `AU_SPARSE_MODEL` |
| `RERANK_MODEL` | `AU_RERANK_MODEL_NAME`, `AU_RERANK_MODEL` |
| `QDRANT_COLLECTION_NAME` | `QDRANT_COLLECTION`, `qdrant_collection` |
| `FUSION_ALGORITHM` | `hybrid_fusion` |
| `QDRANT_API_KEY` | `qdrant_cloud_api_key` |

> Nếu `.env` của bạn còn chứa `AU_EMBED_*`, `AU_SPARSE_*`, `AU_RERANK_*` → vẫn chạy OK ở
> v7. Nhưng khuyến nghị trim về tên mới để đồng bộ với `template.env`.

Các biến đã bị **xoá hoàn toàn** (không còn backward-compat): `COHERE_API_KEY`,
`OPENAI_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `RERANK_*` ngoài
`RERANK_MODEL/TOP_K`, `RRF_K`, `DBSF_WINDOW`, `QWEN_EVAL_THINKING`, `EVAL_RPM`,
`EVAL_TPM`, `BACKEND_*`, `RECREATE_COLLECTION`.

## 11. Model Config Reference

| Model | HuggingFace ID | Loại | dim |
|---|---|---|---|
| BGE-m3 | `BAAI/bge-m3` | Dense + Sparse | 1024 |
| BGE-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | Reranker | — |
| Qwen 2.5 (16B) | local serve | Eval LLM (test only) | — |
| Llama 3.3 70B | NVIDIA NIM | Eval LLM (test only) | — |
