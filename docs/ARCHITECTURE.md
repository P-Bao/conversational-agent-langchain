# Kiến trúc Hệ thống — Retrieval & Search API v8.1.0

## 1. Tổng Quan

Repo này chỉ cung cấp **Retrieval & Search API**. Backend FastAPI nhận query,
truy xuất document chunks từ Qdrant (đã có sẵn do hệ thống ngoài dựng), và trả
về danh sách documents. Không sinh câu trả lời ở phía backend. Downstream LLM
(nếu có) tự dùng context này để trả lời.

Collection / ingest / delete trên Qdrant thuộc về hệ thống ngoài (không có
endpoint nào trong repo này để tạo collection, upload file, hay xoá document).

```
Caller / Client
    |
    v
Retrieval & Search API (FastAPI :8005)
    |
    +-- GET  /healthz               (liveness)
    +-- GET  /readyz                (Qdrant connectivity + collection exists)
    |
    +-- POST /semantic/search       (direct hybrid search, no rerank)
    +-- POST /rag/                  (LangGraph: hybrid retrieval + rerank)
    +-- POST /rag/stream            (NDJSON stream của /rag/)
    |
    +-- HTTP --> Remote BGE-m3 Embed Server (GPU server, default :8008)
    |                POST /embed  ->  {"dense_vecs": [[float,...],...],
    |                                  "sparse_vecs": [...]}
    |                                v
    |   Qdrant (Hybrid Search, dense + sparse fusion)  <-- collection do he ngoai dung
    |                                |
    +-- HTTP --> Remote Reranker Server (default = remote)
    |                POST /rerank ->  {"scores": [float,...],
    |                                  "ranked_indices": [int,...]}
    |
    v
JSON: RetrievalResponse(query, documents[])
```

## 2. Các Component Chính

### 2.1 API Layer (`src/agent/routes/`)

| Route Module | Endpoint | Mô tả |
|---|---|---|
| `rag.py` | `POST /rag/`, `POST /rag/stream` | Hybrid retrieval qua LangGraph + rerank |
| `search.py` | `POST /semantic/search` | Direct hybrid search, không rerank, không graph |
| `health.py` | `GET /healthz`, `GET /readyz` | Liveness + readiness (Qdrant connectivity) |

### 2.2 Data Models (`src/agent/data_model/`)

- `request_data_model.py`: `SearchParams`, `ChatMessages`, `RAGRequest`
- `response_data_model.py`: `SearchResponse`, `RetrievalResponse`, `RetrievedDoc`, `EmbeddingResponse` (giữ nhưng không dùng), `Status`

### 2.3 Backend Core (`src/agent/backend/`)

- `graph.py` — LangGraph `StateGraph` pipeline: `entry -> retriever -> END`. **Giữ nguyên kiến trúc graph gốc.**
- `state.py` — `AgentState(TypedDict)`: query, documents, messages, retry_count
- `nodes/retrieval.py` — `retrieve_documents()`: gọi `get_retriever` (dense) + `get_reranker`. Trả về documents + query + retry_count.

### 2.4 Utilities (`src/agent/utils/`)

- `config.py` — Pydantic Settings, đọc từ `.env`. Embedding provider mặc định `remote`; rerank provider mặc định `remote`.
- `embeddings.py` — `BGEM3RemoteEmbeddings` (dense + sparse), gọi HTTP `POST /embed` tới `EMBEDDING_BASE_URL`. Xem `embeddings.py:29`.
- `retriever.py` — Hybrid retriever wrapper (dense + sparse fusion), cache `QdrantVectorStore` theo `collection_name`.
- `reranker.py` — `get_reranker(cfg, *, top_k=...)` với 3 providers: `none` (passthrough), `remote` (HTTP `POST /rerank` tới `RERANK_BASE_URL`), `bge` (local FlagEmbedding, fallback). Default `remote`. Xem `reranker.py:62`.
- `vdb.py` — Chỉ `QdrantClient` + `AsyncQdrantClient` singleton. CRUD collection / embed / init vdb đã chuyển sang hệ ngoài.

## 3. Data Flow Chi Tiết

### Flow A: `/rag` (Retrieval qua LangGraph)

```
POST /rag/  body: { "messages": [...], "collection_name": "..." }
  1. RAGRequest -> graph.with_config({"metadata": {"collection_name": ...}}).ainvoke(...)
  2. LangGraph chạy "retriever" node
  3. nodes/retrieval.py :: retrieve_documents -> get_retriever(collection_name, k)
  4. retriever.invoke(query) -> Qdrant hybrid search (dense + sparse fusion, remote BGE-m3 embed)
  5. get_reranker(cfg, top_k=...) -> remote rerank (provider=remote), ghi score vào metadata
  6. Clamp top_k bởi min(top_k, k)
  7. Trả về RetrievalResponse(query, documents[])
```

### Flow B: `/semantic/search` (Direct search, không rerank)

```
POST /semantic/search  body: { "query": "...", "k": N, "collection_name": "..." }
  1. SearchParams -> get_retriever(collection_name, k)
  2. await retriever.ainvoke(query)
  3. Trả về SearchResponse[] (text, score, metadata)
```

### Flow C: Health check

```
GET /healthz -> 200 {"status": "ok"}  (luôn luôn — chỉ xác nhận process sống)

GET /readyz  -> 200 {"status": "ready", "collection": "default"} (Qdrant OK + collection tồn tại)
            -> 503 {"status": "fail", "reason": "collection_missing"}  (Qdrant OK nhưng collection không có)
            -> 503 {"status": "fail", "reason": "qdrant_unreachable"} (không connect được Qdrant)
            -> 503 {"status": "fail", "reason": "qdrant_error"}    (Qdrant trả về non-2xx)
```

## 4. Design Decisions

| Decision | Lý do |
|---|---|
| Retrieval-only (không LLM sinh answer) | Backend pure context provider; downstream LLM tự xử lý response. Giảm tài nguyên GPU local. |
| Tách retrieval khỏi collection management | Repo này chỉ đọc Qdrant đã có sẵn; ingestion / CRUD thuộc hệ ngoài tránh duplicate responsibility. |
| Remote BGE-m3 thay vì local | Docker image nhẹ, không tải model ~2.7GB, không phụ thuộc CUDA/torch wheels. Model chạy trên Colab T4 / server GPU riêng qua HTTP. |
| Hybrid search (dense + sparse fusion) | Remote embed server trả cả `dense_vecs` lẫn `sparse_vecs`; fusion nâng recall so với dense-only. |
| `EMBEDDING_BASE_URL` bắt buộc | Là base URL của remote server (GPU self-hosted). Không có default;.env phải điền. |
| Reranker default = `"remote"` | Default dùng remote BGE-reranker qua `RERANK_BASE_URL`; đặt `none` nếu muốn passthrough; `bge` giữ làm fallback local. |
| Fail-fast khi rerank remote lỗi | Không tự fallback sang local `bge` khi remote fail — tránh silent behavior change; lỗi hiện rõ trong log. |
| Không lọc theo min_score | Client không gửi `min_score` — BGE-reranker trả raw logit (có thể âm); chỉ giữ `top_k` do server sort. |
| LangGraph giữ nguyên | Đảm bảo tương thích với downstream consumers; chỉ rút gọn module xung quanh. |

## 5. Biểu Đồ Thành Phần

```
Docker (api:8005)
    |
    +-- /src/agent/ --> FastAPI app
    |       |-- routes/       endpoint handlers (rag, search, health)
    |       |-- backend/      LangGraph pipeline (giữ nguyên)
    |       |-- utils/        config, vdb (only client), retriever, embed, rerank
    |       |-- data_model/   request/response pydantic models
    |
    +-- pyproject.toml    dependencies (uv sync)
    +-- .env              secrets + config
    +-- Dockerfile        build image
    +-- docker-compose.yml
    |
    +-- tests/            pytest suite (unit + integration + e2e + contract)
```

## 6. File Map

| Area | Path |
|---|---|
| API app entry | `src/agent/api.py` |
| Config | `src/agent/utils/config.py` |
| Qdrant client (only) | `src/agent/utils/vdb.py` |
| Embeddings | `src/agent/utils/embeddings.py` |
| Retriever | `src/agent/utils/retriever.py` |
| Reranker | `src/agent/utils/reranker.py` |
| LangGraph | `src/agent/backend/graph.py` |
| Retrieval node | `src/agent/backend/nodes/retrieval.py` |
| Agent state | `src/agent/backend/state.py` |
| Health routes | `src/agent/routes/health.py` |
| RAG routes | `src/agent/routes/rag.py` |
| Search routes | `src/agent/routes/search.py` |
| Test suite | `tests/` |

> **Không còn trong repo (chuyển sang hệ ngoài):** `backend/services/embedding_management.py`,
> `scripts/{dump_reader,chunking,migrate_dump_to_qdrant,load_dummy_data}.py`,
> `utils/utility.py`, `data_model/internal_model.py`,
> `routes/{collection,delete,embeddings}.py`.
