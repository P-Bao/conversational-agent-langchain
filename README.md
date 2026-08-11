# Retrieval & Search API (v8.1.0)

Backend RAG retrieval service: trả về context (documents) cho downstream LLMs bằng remote BGE-m3 (dense+sparse qua HTTP) và remote BGE-reranker v2-m3 (mặc định `remote` qua rerank-server `:8010`). Branch `feature/retrieval-search-only` — chỉ retrieval & search, ingestion do hệ thống ngoài quản lý. Docker image cần GPU (giữ `bge` local fallback); rerank-server chạy ngoài container.

## Features (v8.1.0)

- **Retrieval-Only API**: `/rag` và `/rag/stream` trả về danh sách các document chunks đã qua **hybrid dense+sparse retrieval** (remote BGE-m3, `:8008`) + **optional rerank** (mặc định `remote` qua rerank-server `:8010`).
- **Direct Semantic Search**: `/semantic/search` — hybrid search không qua graph pipeline.
- **Health checks**: `/healthz` (liveness) + `/readyz` (Qdrant + collection readiness) cho Docker / Kubernetes.
- **Remote BGE-m3 Embedding (dense+sparse)**: Gọi BGE-m3 qua `EMBEDDING_BASE_URL` (vd `http://bge-m3-embed:8008`). Bearer auth qua `EMBEDDING_API_KEY`.
- **Rerank mặc định remote qua `:8010`**: `RERANK_PROVIDER=remote` (contract `{scores, ranked_indices}`); fallback `bge` local (cần GPU) qua `RERANK_PROVIDER=bge`; `none` passthrough.
- **`top_k` request param** (`/rag/` + `/rag/stream`): override `RERANK_TOP_K` per-request (1-40).
- **LangGraph Graph (giữ nguyên)**: Pipeline retrieval 1-node `retriever` → END.
- **DeepEval với Qwen Self-host**: Suite đánh giá `ContextualPrecision` và `ContextualRecall` chạy qua TestClient `POST /rag/`.

## Endpoints

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Welcome |
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe (Qdrant OK + collection tồn tại) |
| POST | `/rag/` | Retrieval qua LangGraph (hybrid) + optional rerank (remote `:8010`) |
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

1. Sao chép `template.env` thành `.env` (cập nhật `EMBEDDING_BASE_URL`, `RERANK_BASE_URL`, `QDRANT_URL`, `QDRANT_COLLECTION_NAME`):
   ```bash
   cp template.env .env
   ```
   > Embedding/rerank chạy trên remote server (embed `:8008`, rerank `:8010`). Điền vào `EMBEDDING_BASE_URL` / `RERANK_BASE_URL` (bắt buộc với `RERANK_PROVIDER=remote`). Repo Docker image không tải model.

2. Sync dependencies và chạy:
   ```bash
   uv sync
   uv run uvicorn agent.api:app --reload --port 8005
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
     -d '{"query":"test","k":3,"collection_name":"documents"}'
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
    POST /rag/               (LangGraph: retriever + rerank)
    POST /rag/stream         (NDJSON stream)
    |
    +-- Remote BGE-m3 Embed (dense+sparse :8008)  via EMBEDDING_BASE_URL (/embed)
    +-- Remote BGE Reranker (:8010)              via RERANK_BASE_URL (/rerank)
    |
    v
Qdrant (Hybrid Search, dense + sparse fusion)  <-- collection do he ngoai quan ly
```

## Testing & Evaluation

- **Unit tests** (51/51 pass ở v8.1.0):
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

- **Qwen / NVIDIA NIM DeepEval**:
  ```bash
  ALLOW_NETWORK_TESTS=1 uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
  ```
