# Cấu Hình (Environment Variables) — v6.0.0

> Tất cả env vars đều đọc từ `.env` qua `python-dotenv` + Pydantic Settings.
> Mỗi biến có alias để backward-compat. Default an toàn trong code.

## 1. Embedding (Dense)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `AU_EMBED_MODEL_NAME` | `EMBEDDING_MODEL` | `BAAI/bge-m3` | Tên model dense trên HuggingFace |
| `AU_EMBED_DIMENSION` | `EMBEDDING_SIZE` | `1024` | Dense vector dimension. BGE-m3 support 1024 (mặc định) |
| `EMBEDDING_PROVIDER` | — | `bge` | Provider: `bge`, `flagembedding`, `openai`, `openai-compatible`, `custom` |
| `EMBEDDING_BASE_URL` | — | `""` | Base URL nếu dùng OpenAI-compatible API cho embedding |
| `EMBEDDING_API_KEY` | — | `""` | API key cho remote embedding API |

Lưu ý:
- `bge`/`flagembedding` dùng model local qua FlagEmbedding.
- `openai-compatible` đòi hỏi `EMBEDDING_BASE_URL` hợp lệ (dùng cho BGE serve qua API).

## 2. Embedding (Sparse)

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `AU_SPARSE_MODEL_NAME` | `SPARSE_MODEL` | `BAAI/bge-m3` | Model cho sparse lexical weights. Tách env với dense để swap linh hoạt |

Mặc định cùng model với dense (BGE-m3) → 1 instance FlagEmbed chia sẻ.
Nếu đặt khác, load model riêng (tốn RAM gấp đôi).

## 3. Retrieval

| Biến | Default | Mô tả |
|---|---|---|
| `RETRIEVAL_K` | `40` | Số document lấy từ hybrid search (trước rerank) |
| `RETRIEVAL_K_RETRY` | `100` | Số document lấy khi `retry_count > 0` (retry handle chưa active trong graph v6) |

## 4. Reranker

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `RERANK_PROVIDER` | — | `bge` | Provider: `bge`, `cohere`, `flashrank`, `none` |
| `RERANK_TOP_K` | — | `5` | Số document giữ lại sau rerank |
| `AU_RERANK_MODEL_NAME` | `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Model reranker trên HuggingFace |
| `RERANK_BASE_URL` | — | `""` | Base URL nếu dùng remote rerank API |
| `RERANK_API_KEY` | — | `""` | API key cho remote rerank |

## 5. Fusion Algorithm

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `FUSION_ALGORITHM` | `hybrid_fusion` | `rrf` | `rrf` (Reciprocal Rank Fusion) hoặc `dbsf` (Distribution-Based) |
| `RRF_K` | — | `60` | Constant K trong RRF (càng cao, càng ưu tiên rank cao) |
| `DBSF_WINDOW` | — | `1000` | Window size cho DBSF |

## 6. Qdrant

| Biến | Alias | Default | Mô tả |
|---|---|---|---|
| `QDRANT_URL` | — | `http://localhost` | URL Qdrant (⚠️ Không dùng `localhost` nếu cả 2 đều trong container) |
| `QDRANT_PORT` | — | `6333` | REST API port |
| `QDRANT_API_KEY` | `qdrant_cloud_api_key` | `(none)` | API key Qdrant Cloud |
| `QDRANT_COLLECTION_NAME` | `QDRANT_COLLECTION`, `qdrant_collection` | `default` | Tên collection chính |

⚠️ **Important**: Khi API chạy Docker container, `localhost` trỏ về chính container đó.
Nếu Qdrant chạy ở host → dùng `http://host.docker.internal`.
Nếu Qdrant cùng Docker network → dùng service name (vd `http://qdrant`).
Xem [DEPLOYMENT.md](DEPLOYMENT.md) để biết cách setup đúng.

## 7. DeepEval — NVIDIA NIM

| Biến | Default | Mô tả |
|---|---|---|
| `NVIDIA_API_KEY` | `""` | API key cho NVIDIA NIM |
| `NVIDIA_EVAL_MODEL` | `meta/llama-3.3-70b-instruct` | Model eval trên NVIDIA NIM |
| `NVIDIA_EVAL_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Endpoint NVIDIA NIM |
| `NVIDIA_EVAL_RPS` | `30` | Giới hạn requests/sec cho rate limiter |

Eval LLM dùng rate limit (`NVIDIA_EVAL_RPS`). Có thể bật/tắt qua `EVAL_RPM`, `EVAL_TPM`.

## 8. DeepEval — Qwen (self-host)

| Biến | Default | Mô tả |
|---|---|---|
| `QWEN_EVAL_BASE_URL` | `http://localhost:8000/v1` | Endpoint OpenAI-compatible của Qwen |
| `QWEN_EVAL_API_KEY` | `""` | API key (nếu cần) |
| `QWEN_EVAL_MODEL` | `qwen` | Tên model Qwen |
| `QWEN_EVAL_THINKING` | `false` | Bật/tắt thinking mode của Qwen (extra_body) |

Qwen không có rate limit (chạy local).

## 9. Rate Limit (tổng quát — backup)

| Biến | Default | Mô tả |
|---|---|---|
| `EVAL_RPM` | `13` | Giới hạn requests/minute cho Eval LLM |
| `EVAL_TPM` | `40000` | Giới hạn tokens/minute |

Hiện tại các rate limit này được dùng chủ yếu trong DeepEval suite.

## 10. Migration

| Biến | Default | Mô tả |
|---|---|---|
| `INPUT_DIR` | `../input` | Thư mục chứa Mongo dump JSON |
| `MIGRATE_CHECKPOINT_FILE` | `./migration_checkpoint.jsonl` | File checkpoint resume migration |
| `MIGRATE_MAX_DOCUMENTS` | không set | Giới hạn số document (ưu tiên thấp hơn CLI `--limit`) |
| `MIGRATE_UPSERT_BATCH_SIZE` | `50` | Số record upsert 1 batch vào Qdrant |

## 11. Chunking

| Biến | Default | Mô tả |
|---|---|---|
| `CHUNK_SIZE` | `1500` | Kích thước chunk (ký tự) |
| `CHUNK_OVERLAP` | `100` | Overlap giữa các chunk |
| `MIN_CHUNK_TOKENS` | `100` | Gộp chunk nếu dưới ngưỡng token |
| `CHUNK_CHECKPOINT_FILE` | `./chunk_checkpoint.jsonl` | File checkpoint resume chunking |
| `ENABLE_LLM_ENRICH` | `false` | Bật enrich title/keywords bằng LLM (yêu cầu `LLM_BASE_URL` etc) |

## 12. Frontend (optional)

| Biến | Default | Mô tả |
|---|---|---|
| `BACKEND_HOST` | `localhost` | Host của backend API cho Streamlit |
| `BACKEND_PORT` | `8001` | Port backend API |

## 13. Backward Compatibility

| Biến | Dùng trong | Mô tả |
|---|---|---|
| `COHERE_API_KEY` | reranker (cohere provider) | Cohere Rerank API key |
| `OPENAI_API_KEY` | graph cũ / deep eval | Giữ để test backward-compat |

## 14. Model Config Reference

Nếu cần swap model embedding/reranker, liệt kê các model được test ổn định:

| Model | HuggingFace ID | Loại | dim |
|---|---|---|---|
| BGE-m3 | `BAAI/bge-m3` | Dense+Sparse | 1024 |
| BGE-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | Reranker | — |
| Qwen 2.5 (16B) | local serve | Eval LLM | — |
| Llama 3.3 70B | NVIDIA NIM | Eval LLM | — |
