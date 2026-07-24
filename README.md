# Retrieval & Search API (v7.0.0)

Backend RAG retrieval service: trả về context (documents) cho downstream LLMs bằng remote BGE-m3 (dense, qua HTTP) và remote BGE-reranker v2-m3 (optional). Branch `feature/retrieval-search-only` — chỉ retrieval & search, ingestion do hệ thống ngoài quản lý. Docker image không load model (CUDA-free, model-cache-free).

## Features (v7.0.0)

- **Retrieval-Only API**: `/rag` và `/rag/stream` trả về danh sách các document chunks đã qua dense retrieval (remote BGE-m3) + optional rerank (không sinh answer ở backend).
- **Direct Semantic Search**: `/semantic/search` — dense search không qua graph pipeline.
- **Health checks**: `/healthz` (liveness) + `/readyz` (Qdrant + collection readiness) cho Docker / Kubernetes.
- **Remote BGE-m3 Embedding**: Gọi BGE-m3 qua `EMBEDDING_BASE_URL` (Colab ngrok / server GPU Docker-light). Dense-only 1024-dim, không chạy local model trong container.
- **Optional Remote BGE Reranker v2-m3**: Mặc định `RERANK_PROVIDER=none` (passthrough). Bật qua `RERANK_PROVIDER=remote` + `RERANK_BASE_URL`.
- **LangGraph Graph (giữ nguyên)**: Pipeline retrieval 1-node `retriever` → END, dùng cho `/rag/`.
- **DeepEval với Qwen Self-host**: Suite đánh giá `ContextualPrecision` và `ContextualRecall` chạy qua TestClient `POST /rag/`.

## Endpoints

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Welcome |
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe (Qdrant OK + collection tồn tại) |
| POST | `/rag/` | Retrieval qua LangGraph + optional rerank |
| POST | `/rag/stream` | NDJSON stream của `/rag/` |
| POST | `/semantic/search` | Direct dense search (no rerank) |

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
   > Embedding/rerank chạy trên remote server (notebook `rag_test_bge_m3_reranker_ngrok.ipynb` trên Colab T4). Lấy ngrok URL, điền vào `EMBEDDING_BASE_URL` / `RERANK_BASE_URL` (nếu bật rerank). Repo Docker image không tải model.

2. Sync dependencies và chạy:
   ```bash
   uv sync
   uv run uvicorn agent.api:app --reload --port 8001
   ```

3. Verify health:
   ```bash
   curl http://localhost:8001/healthz
   # {"status":"ok"}

   curl http://localhost:8001/readyz
   # {"status":"ready","collection":"documents"}   (nếu Qdrant + collection OK)
   ```

4. Search thử:
   ```bash
   curl -X POST http://localhost:8001/semantic/search \
     -H "Content-Type: application/json" \
     -d '{"query":"test","k":3,"collection_name":"documents"}'
   ```

## Architecture

```
Caller / Client
    |
    v
Retrieval & Search API (FastAPI :8001)
    |
    GET  /healthz            (liveness - always 200 if process up)
    GET  /readyz             (Qdrant connectivity + collection)
    POST /semantic/search    (direct dense search, no graph)
    POST /rag/               (LangGraph: retriever + optional rerank)
    POST /rag/stream         (NDJSON stream)
    |
    +-- Remote BGE-m3 Embed (1024 dense)  via EMBEDDING_BASE_URL (/embed)
    +-- Remote BGE Reranker (optional)   via RERANK_BASE_URL (/rerank)
    |
    v
Qdrant (Dense Search, COSINE)  <-- collection do he ngoai quan ly
```

## Testing & Evaluation

- **Unit tests** (61/61 pass ở v7.0.0):
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
