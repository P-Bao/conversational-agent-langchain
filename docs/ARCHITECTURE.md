# Kiến trúc Hệ thống — Retrieval & Search API v7.1.0

> **Khác v7.0.0:**
> - LangGraph thêm node **`query_transform`** (tuỳ chọn, bật qua `QUERY_TRANSFORM_ENABLED`).
> - Retrieval đổi từ dense-only sang **hybrid dense + sparse** (Qdrant `RetrievalMode.HYBRID`).
> - Reranker đổi từ remote HTTP sang **local FlagEmbedding** (default `bge`).
> - Port API `8001` → **`8005`**, network `test_network` → **`ami-network`**.

## 1. Tổng Quan

Repo này chỉ cung cấp **Retrieval & Search API**. Backend FastAPI nhận query,
(tuỳ chọn) biến đổi query qua Qwen LLM, rồi truy xuất document chunks từ
Qdrant (đã có sẵn do hệ thống ingestion ngoài dựng), và trả về danh sách
documents. Không sinh câu trả lời ở phía backend.

```
Caller / Client
    |
    v
Retrieval & Search API (FastAPI :8005)
    |
    +-- GET  /healthz               (liveness — process sống)
    +-- GET  /readyz                (Qdrant + collection exists)
    |
    +-- POST /semantic/search       (direct hybrid search, no rerank, no graph)
    +-- POST /rag/                  (LangGraph: query_transform? -> retriever -> END)
    +-- POST /rag/stream            (NDJSON stream của /rag/)
    |
    +-- (optional) Qwen self-host LLM  (query rewrite + step-back + decompose)
    |
    +-- HTTP --> Embedding Server (bge-m3-embed:8008 — repo ngoài)
    |                POST /embed  ->  {"dense_vecs", "sparse_vecs"}
    |                                v
    |   Qdrant (Hybrid Search, RetrievalMode.HYBRID)  <-- collection do hệ ngoài dựng
    |                                |
    +-- Local FlagReranker (BAAI/bge-reranker-v2-m3)  <-- rerank theo câu hỏi gốc
    |
    v
JSON: RetrievalResponse(query, documents[])
```

## 2. Các Component Chính

### 2.1 API Layer (`src/agent/routes/`)

| Route Module | Endpoint | Mô tả |
|---|---|---|
| `rag.py` | `POST /rag/`, `POST /rag/stream` | Hybrid retrieval qua LangGraph + optional query transform + local rerank |
| `search.py` | `POST /semantic/search` | Direct hybrid search, không rerank, không graph |
| `health.py` | `GET /healthz`, `GET /readyz` | Liveness + readiness (Qdrant connectivity) |

### 2.2 Data Models (`src/agent/data_model/`)

- `request_data_model.py`: `SearchParams`, `ChatMessages`, `RAGRequest`
- `response_data_model.py`: `SearchResponse` (`page`/`source` optional từ v7.1),
  `RetrievalResponse`, `RetrievedDoc`, `Status`

### 2.3 Backend Core (`src/agent/backend/`)

- **`graph.py`** — LangGraph `StateGraph` pipeline. Hai mode:
  - `QUERY_TRANSFORM_ENABLED=false` (default): `entry -> retriever -> END` (giữ nguyên v7.0)
  - `QUERY_TRANSFORM_ENABLED=true`: `entry -> query_transform -> retriever -> END`
- **`state.py`** — `AgentState(TypedDict, total=False)`:
  `query`, `original_query`, `rewritten_query`, `step_back_query`,
  `sub_queries`, `documents`, `messages`, `retry_count`
- **`nodes/query_transform.py`** — `transform_query()` node:
  - Gọi Qwen qua `RunnableParallel` (3 prompts song song: rewrite, step-back, decompose).
  - Khi LLM lỗi hoặc feature tắt → fallback về query gốc, pipeline không bao giờ break.
- **`nodes/retrieval.py`** — `retrieve_documents()`:
  - Build danh sách queries: `[original, rewritten, step_back, *sub_queries]` (dedupe).
  - Retrieve tuần tự từng query qua Qdrant hybrid search.
  - Dedupe kết quả bằng `metadata.global_id` (hoặc hash `page_content`).
  - Rerank toàn bộ bằng **câu hỏi gốc** (không phải query biến đổi).

### 2.4 Utilities (`src/agent/utils/`)

- `config.py` — Pydantic Settings. Default: `embedding_provider=remote`,
  `rerank_provider=bge`, `query_transform_enabled=false`.
- `embeddings.py` — `BGEM3RemoteEmbeddings`: wrapper gọi HTTP `POST /embed`
  tới `EMBEDDING_BASE_URL`, trả dense + sparse. Có hỗ trợ auth header
  (`EMBEDDING_API_KEY`).
- `retriever.py` — Hybrid retriever (`RetrievalMode.HYBRID`), dùng dense
  embedding + sparse embedding (`BGE_M3SparseEmbeddings` wrapper). Cache
  `QdrantVectorStore` theo `collection_name`.
- `reranker.py` — `get_reranker(cfg, *, top_k)`:
  - `bge` (default): local `FlagReranker` (BAAI/bge-reranker-v2-m3), cache model.
  - `remote` (legacy): HTTP `POST /rerank` tới remote server (Colab ngrok,...).
  - `none`: passthrough (truncate top_k).
- `vdb.py` — `QdrantClient` + `AsyncQdrantClient` singleton. Không CRUD.

## 3. Data Flow Chi Tiết

### Flow A: `/rag` (Retrieval qua LangGraph, có query transform)

```
POST /rag/  body: { "messages": [...] }
  1. RAGRequest -> graph.ainvoke({"messages": messages})
  2. LangGraph chạy:
     [query_transform] (nếu QUERY_TRANSFORM_ENABLED=true)
       -> transform_query(state, cfg)
       -> Gọi Qwen 3 lần song song (rewrite + step-back + decompose)
       -> state.original_query = query gốc
       -> state.rewritten_query, .step_back_query, .sub_queries = biến thể
       -> Fallback nếu LLM lỗi
     [retriever]
       -> _build_queries(state) -> [original, rewritten, step_back, *sub_queries]
       -> for q in queries: retriever.invoke(q)  # Qdrant HYBRID search
       -> dedupe theo metadata.global_id
       -> rerank(docs, query=original_query)  # neo theo câu gốc
  3. Trả về RetrievalResponse(query, documents[])
```

### Flow A': `/rag` (Retrieval, không query transform — default)

```
POST /rag/  body: { "messages": [...] }
  1. RAGRequest -> graph.ainvoke({"messages": messages})
  2. LangGraph chạy [retriever]:
       -> _build_queries: chỉ [original_query]  (legacy behaviour)
       -> retriever.invoke(query)  # Qdrant HYBRID search
       -> rerank(docs, query=original_query)
  3. Trả về RetrievalResponse(query, documents[])
```

### Flow B: `/semantic/search` (Direct search, không graph, không rerank)

```
POST /semantic/search  body: { "query", "k" }
  1. SearchParams -> get_retriever(k)
  2. await retriever.ainvoke(query)
  3. Trả về SearchResponse[] (text, page?, source?)
```

### Flow C: Health check

```
GET /healthz -> 200 {"status": "ok"}                      (process sống)
GET /readyz  -> 200 {"status": "ready", "collection": ...}  (Qdrant OK + collection)
            -> 503 {"reason": "collection_missing"}
            -> 503 {"reason": "qdrant_unreachable"}
            -> 503 {"reason": "qdrant_error"}
```

## 4. Design Decisions

| Decision | Lý do |
|---|---|
| Retrieval-only (không sinh answer) | Backend là context provider; downstream LLM tự xử lý. Giảm tài nguyên GPU local |
| Hybrid dense + sparse retrieval | BGE-m3 hỗ trợ cả 2; Qdrant HYBRID cải thiện recall so với dense-only |
| Reranker local (bge default) | Loại bỏ latency mạng tới remote server (~100ms); tận dụng GPU container |
| Query transformation optional | Khi bật tăng recall cho câu phức tạp; khi tắt giữ latency thấp như v7.0 |
| Qwen cho query transform | Self-host, không phí API, OpenAI-compatible, dễ bật/tắt |
| Rerank theo câu gốc (không query biến đổi) | Đảm bảo reranker đo relevance đúng ý người dùng, không đo relevance với paraphrase |
| Multi-query dedupe theo `global_id` | Tránh trả về trùng lặp khi nhiều biến thể query trùng chunk |
| `QUERY_TRANSFORM_ENABLED=false` default | Backward-compat: pipeline v7.1 mặc định giống v7.0; bật khi cần |
| Port 8005 + network `ami-network` | Tách biệt với các stack khác trong ami infrastructure |

## 5. Biểu Đồ Thành Phần

```
Docker (api:8005) [GPU]
    |
    +-- /src/agent/ --> FastAPI app
    |       |-- routes/       endpoint handlers (rag, search, health)
    |       |-- backend/      LangGraph pipeline
    |       |     |-- graph.py           conditional: query_transform? -> retriever -> END
    |       |     |-- state.py            AgentState (query, variants, documents, ...)
    |       |     |-- nodes/query_transform.py   Qwen LLM rewrite+step-back+decompose
    |       |     +-- nodes/retrieval.py         multi-query retrieve + dedupe + rerank
    |       |-- utils/        config, vdb, retriever (HYBRID), embeddings (dense+sparse), reranker (local bge)
    |       +-- data_model/   request/response pydantic models
    |
    +-- requirements.txt   dependencies (pip)
    +-- .env              secrets + config
    +-- Dockerfile        build image (PyTorch CUDA base)
    +-- docker-compose.yml
    +-- hf-cache/         volume mount — cache BGE-reranker model
    +-- tests/            pytest suite (unit + integration + e2e + contract + deepeval)
```

## 6. File Map

| Area | Path |
|---|---|
| API app entry | `src/agent/api.py` |
| Config | `src/agent/utils/config.py` |
| Qdrant client | `src/agent/utils/vdb.py` |
| Embeddings (remote dense+sparse) | `src/agent/utils/embeddings.py` |
| Retriever (hybrid) | `src/agent/utils/retriever.py` |
| Reranker (local bge) | `src/agent/utils/reranker.py` |
| LangGraph | `src/agent/backend/graph.py` |
| Query transform node | `src/agent/backend/nodes/query_transform.py` |
| Retrieval node | `src/agent/backend/nodes/retrieval.py` |
| Agent state | `src/agent/backend/state.py` |
| Health routes | `src/agent/routes/health.py` |
| RAG routes | `src/agent/routes/rag.py` |
| Search routes | `src/agent/routes/search.py` |
| DeepEval NIM suite | `tests/test_rag_deepeval_nim.py` |
| Unit tests | `tests/unit_tests/` |

## 7. Đã loại bỏ (chuyển sang repo ngoài)

- `backend/services/embedding_management.py` — embedding server tách riêng
- `scripts/{dump_reader,chunking,migrate_dump_to_qdrant,load_dummy_data}.py` — ingestion ngoài
- `utils/utility.py`, `data_model/internal_model.py`
- `routes/{collection,delete,embeddings}.py` — CRUD endpoints
- Local BGE-m3 embedding (`EMBEDDING_PROVIDER=bge`) — chuyển sang embedding-server container
- Dense-only retrieval (`RetrievalMode.DENSE`) — chuyển sang HYBRID
- DeepEval `test_rag_deepeval_qwen.py` — thay bằng `test_rag_deepeval_nim.py` (NIM-only)
