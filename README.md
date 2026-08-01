# Retrieval & Search API (v7.1.0)

Backend RAG retrieval service: trả về context (documents) cho downstream
LLMs bằng **remote BGE-m3 (dense + sparse, qua HTTP)** và **local
BGE-reranker-v2-m3** (optional, default bật). Branch `feat/qwen-query-transform-nim-eval` — chỉ retrieval & search, ingestion do hệ thống ngoài quản lý. Docker image dùng PyTorch CUDA base (GPU cho local reranker).

## Features (v7.1.0)

- **Retrieval-Only API**: `/rag` và `/rag/stream` trả về danh sách các document chunks đã qua **hybrid retrieval** (remote BGE-m3 dense+sparse) + **optional rerank** (local BGE-reranker-v2-m3, mặc định bật). Không sinh answer ở backend.
- **Direct Semantic Search**: `/semantic/search` — hybrid search không qua graph pipeline.
- **Health checks**: `/healthz` (liveness) + `/readyz` (Qdrant + collection readiness) cho Docker / Kubernetes.
- **Remote BGE-m3 Embedding**: Gọi BGE-m3 qua `EMBEDDING_BASE_URL` (embedding-server container, port 8008). Trả dense 1024-dim + sparse vectors.
- **Local BGE Reranker v2-m3**: Mặc định `RERANK_PROVIDER=bge` (FlagEmbedding, chạy trong container, cần GPU). Tắt qua `RERANK_PROVIDER=none`. Legacy remote vẫn hỗ trợ.
- **Optional Query Transformation**: Bật `QUERY_TRANSFORM_ENABLED=true` + Qwen self-host → thêm node `query_transform` (rewrite + step-back + decompose) trước retrieve.
- **LangGraph Graph**: Conditional pipeline — `query_transform? -> retriever -> END`. Khi tắt query transform: giữ nguyên `retriever -> END` như v7.0.
- **DeepEval với NVIDIA NIM**: Suite đánh giá `test_rag_deepeval_nim.py` — 5 metrics (Correctness GEval, Faithfulness, ContextualRelevancy, Precision, Recall) chạy qua `evaluate()` batch. Judge: NVIDIA NIM `meta/llama-3.3-70b-instruct`.

## Endpoints

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Welcome |
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe (Qdrant OK + collection tồn tại) |
| POST | `/rag/` | Retrieval qua LangGraph (+ optional query transform + rerank) |
| POST | `/rag/stream` | NDJSON stream của `/rag/` |
| POST | `/semantic/search` | Direct hybrid search (no rerank) |

> **Endpoints đã bỏ (chuyển sang repo ingestion ngoài):** `POST /collection/create/{name}`,
> `POST /embeddings/documents`, `POST /embeddings/string/`, `DELETE /embeddings/delete/{source}`.
> Repo này chỉ đọc Qdrant.

## Documentation

Bộ tài liệu bàn giao đầy đủ tại [`docs/`](docs/README.md):

| Lĩnh vực | File |
|---|---|
| Kiến trúc | [ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Cài đặt | [SETUP.md](docs/SETUP.md) |
| Triển khai Docker | [DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| API Reference | [API_REFERENCE.md](docs/API_REFERENCE.md) |
| Cấu hình env | [CONFIGURATION.md](docs/CONFIGURATION.md) |
| Vận hành | [OPERATIONS.md](docs/OPERATIONS.md) |
| Troubleshooting | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| Bảo mật | [SECURITY.md](docs/SECURITY.md) |
| Testing | [TESTING.md](docs/TESTING.md) |
| Phát triển | [DEVELOPMENT.md](docs/DEVELOPMENT.md) |
| Đánh giá DeepEval | [EVALUATION.md](docs/EVALUATION.md) |
| Nhập dữ liệu | [DATA_INGESTION.md](docs/DATA_INGESTION.md) |
| Thuật ngữ | [GLOSSARY.md](docs/GLOSSARY.md) |
| User Guide | [USER_GUIDE.md](docs/USER_GUIDE.md) |
| Handover Checklist | [HANDOVER_CHECKLIST.md](docs/HANDOVER_CHECKLIST.md) |

## Quickstart

1. Sao chép `template.env` thành `.env` (cập nhật `EMBEDDING_BASE_URL`, `RERANK_PROVIDER`, `QDRANT_URL`, `QDRANT_COLLECTION_NAME`, tuỳ chọn `QUERY_TRANSFORM_ENABLED` + `QWEN_BASE_URL`):
   ```bash
   cp template.env .env
   ```
   > Embedding chạy trên embedding-server container (`http://bge-m3-embed:8008` trong Docker network). Reranker local bật mặc định (`RERANK_PROVIDER=bge`, cần GPU). Query transform tắt mặc định.

2. Tạo network + start các stack:
   ```bash
   docker network create ami-network
   cd ../qdrant_docker && docker compose up -d
   cd ../embedding-server && docker compose up -d
   cd ../conversational-agent-langchain && docker compose up --build -d
   ```

3. Verify health:
   ```bash
   curl http://localhost:8005/healthz
   # {"status":"ok"}

   curl http://localhost:8005/readyz
   # {"status":"ready","collection":"documents"}   (nếu Qdrant + collection OK)
   ```

4. Search thử:
   ```bash
   curl -X POST http://localhost:8005/semantic/search \
     -H "Content-Type: application/json" \
     -d '{"query":"test","k":3}'
   ```

## Architecture

```
Caller / Client
    |
    v
Retrieval & Search API (FastAPI :8005)
    |
    GET  /healthz            (liveness - always 200 if process up)
    GET  /readyz             (Qdrant connectivity + collection)
    POST /semantic/search    (direct hybrid search, no graph)
    POST /rag/               (LangGraph: query_transform? -> retriever + rerank)
    POST /rag/stream         (NDJSON stream)
    |
    +-- (optional) Qwen self-host LLM  (query rewrite + step-back + decompose)
    |
    +-- HTTP --> Embedding Server (bge-m3-embed:8008 — repo ngoài)
    |                POST /embed  ->  {"dense_vecs", "sparse_vecs"}
    |                                v
    |   Qdrant (Hybrid Search, RetrievalMode.HYBRID)  <-- collection do hệ ngoài quản lý
    |                                |
    +-- Local FlagReranker (BAAI/bge-reranker-v2-m3)  <-- rerank theo câu hỏi gốc
    |
    v
JSON: RetrievalResponse(query, documents[])
```

## Testing & Evaluation

- **Unit tests** (70+ pass):
  ```bash
  uv run pytest tests/unit_tests -q
  ```

- **Integration tests**:
  ```bash
  uv run pytest tests/test_integration.py
  ```

- **E2E live**:
  ```bash
  RUN_LIVE_E2E=1 uv run pytest tests/test_stream.py
  ```

- **NVIDIA NIM DeepEval**:
  ```bash
  ALLOW_NETWORK_TESTS=1 NVIDIA_API_KEY=nvapi-xxx \
    uv run pytest tests/test_rag_deepeval_nim.py -m qwen -vv
  ```
  - 5 metrics: Correctness (GEval), Faithfulness, ContextualRelevancy, ContextualPrecision, ContextualRecall
  - Auto generate answer từ retrieved context qua NIM

## Migration từ v7.0 → v7.1

| Thay đổi | Mô tả |
|---|---|
| Port | `8001` → `8005` |
| Network | `test_network` → `ami-network` (external) |
| Embedding | Dense-only → **Dense + Sparse** (hybrid retrieval) |
| Reranker | Remote HTTP → **Local FlagEmbedding** (default `bge`) |
| Docker base | `uv:python3.13-bookworm-slim` → `pytorch:2.7.1-cuda12.6` |
| Deps | `uv sync` → `pip install -r requirements.txt` (FlagEmbedding, transformers) |
| DeepEval | `test_rag_deepeval_qwen.py` (Qwen/NIM) → `test_rag_deepeval_nim.py` (NIM-only, 5 metrics) |
| Query Transform | — | Tuỳ chọn qua Qwen (`QUERY_TRANSFORM_ENABLED=true`) |

> Xem `CHANGELOG.md` chi tiết.