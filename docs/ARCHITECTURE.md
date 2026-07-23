# Kiến trúc Hệ thống — Conversational Agent LangChain v6.0.0

## 1. Tổng Quan

Hệ thống là **Retrieval-Only RAG API** — backend FastAPI trả về các document chunks
chứa thông tin liên quan tới câu query của người dùng, không sinh câu trả lời bằng LLM
ở phía backend. Downstream LLM (nếu có) tự dùng context này để trả lời.

```
User / Client
    |
    v
RAG Backend (FastAPI :8001)
    |
    +-> BGE-m3 Dense Embed (1024-dim) ──┐
    +-> BGE-m3 Sparse Embed (lexical) ──┤
    |                                    v
    |          Qdrant (Hybrid Search) <--+
    |                    |
    |                    v
    +-> BGE Reranker v2-m3 (cross-encoder rerank top-K)
    |
    v
JSON: RetrievalResponse(query, documents[])
```

## 2. Các Component Chính

### 2.1 API Layer (`src/agent/routes/`)

| Route Module | File | Mô tả |
|---|---|---|
| `rag.py` | `src/agent/routes/rag.py` | POST `/rag/` (sync) + `/rag/stream` (NDJSON) |
| `search.py` | `src/agent/routes/search.py` | POST `/semantic/search` — direct search |
| `embeddings.py` | `src/agent/routes/embeddings.py` | POST `/embeddings/documents` + `/embeddings/string/` |
| `collection.py` | `src/agent/routes/collection.py` | POST `/collection/create/{name}` |
| `delete.py` | `src/agent/routes/delete.py` | DELETE `/embeddings/delete/{source}` |

### 2.2 Data Models (`src/agent/data_model/`)

- `request_data_model.py`: `SearchParams`, `RAGRequest`, `EmbeddTextRequest`
- `response_data_model.py`: `SearchResponse`, `RetrievalResponse`, `RetrievedDoc`, `EmbeddingResponse`
- `internal_model.py`: `RetrievalResults` (dùng nội bộ)

### 2.3 Backend Core (`src/agent/backend/`)

- `graph.py` — LangGraph StateGraph pipeline: `entry -> retriever -> END` (v6 retrieval-only)
- `state.py` — `AgentState(TypedDict)`: query, documents, messages, retry_count
- `nodes/retrieval.py` — `retrieve_documents()`: gọi retriever hybrid + rerank, lưu ý
  thứ tự documents là thứ tự rerank score giảm dần
- `services/embedding_management.py` — `EmbeddingManagement`: load PDF/txt → split → embed → upsert

### 2.4 Utilities (`src/agent/utils/`)

- `config.py` — Pydantic Settings, đọc từ .env
- `embeddings.py` — BGE3Embeddings (dense) + BGE3SparseEmbeddings (sparse), share 1 BGEM3FlagModel
- `retriever.py` — Hybrid retriever wrapper (RRF/DBSF fusion)
- `reranker.py` — 3 providers: bge (default), cohere, flashrank
- `vdb.py` — QdrantClient singleton, generate collection, init vdb
- `utility.py` — helpers: create_tmp_folder, format_docs_for_citations

### 2.5 Scripts (`src/agent/scripts/`)

- `chunking.py` — Markdown-aware text chunking + LLM enrich (optional)
- `dump_reader.py` — Read MongoDB Extended JSON dump from `input/`
- `migrate_dump_to_qdrant.py` — E2E migration pipeline: read → chunk → embed → upsert
- `load_dummy_data.py` — Quick upload từ `resources/`

## 3. Data Flow Chi Tiết

### Flow A: `/rag` (Retrieval)

```
POST /rag/  body: { "messages": [{"role":"user","content":"..."}], ... }
  1. RAGRequest → graph.ainvoke({messages:[...]})
  2. LangGraph runs "retriever" node
  3. get_retriever() returns QdrantVectorStore.as_retriever(hybrid_fusion=RRF)
  4. retriever.invoke(query) → Qdrant hybrid search (dense + sparse)
  5. rerank_with_bge(docs, query, top_k) → sorted documents (score descending)
  6. return RetrievalResponse(query, documents[])
```

### Flow B: Embedding upload

```
POST /embeddings/documents?collection_name=...
  1. Save uploaded files to tmp_dir
  2. EmbeddingManagement.embed_documents(directory, file_ending)
  3. DirectoryLoader (PyPDFium2Loader / TextLoader) → split (750/200)
  4. vector_db.add_texts(texts, metadatas)
  5. return EmbeddingResponse(status="success", filenames=...)
```

### Flow C: Migration (Mongo dump → Qdrant)

```
uv run python -m agent.scripts.migrate_dump_to_qdrant [--limit N] [--recreate]
  1. Read collection from `input/organization_db.documents.json`
  2. Chunk via chunking.py (Markdown-aware + merge short)
  3. BGE-m3 dense + sparse embed (singleton from embedding.py)
  4. Upsert batch (50/turn) → Qdrant collection
  5. Checkpoint / resume via migration_checkpoint.jsonl
```

## 4. Design Decisions

| Decision | Lý do |
|---|---|
| Retrieval-only (không LLM sinh answer) | Backend pure context provider; downstream LLM tự xử lý response. Giảm tài nguyên GPU local. |
| BGE-m3 cho cả dense + sparse | Một forward pass ra 3 mode (dense/sparse/colbert) — tiết kiệm RAM+CPU. Singleton instance chia sẻ. |
| FlagEmbedding thay vì FastEmbed | FastEmbed chưa support BGE-m3 đầy đủ (thiếu sparse). FlagEmbedding cung cấp `BGEM3FlagModel`. |
| Qdrant hybrid search (RRF/DBSF) | Kết hợp dense (ngữ nghĩa) + sparse (từ vựng) cho search đa ngữ, đặc biệt tiếng Việt cần exact match từ khoá. |
| `use_fp16 = torch.cuda.is_available()` | Tránh CPU hang khi forced FP16. Tự động tắt FP16 trên CPU. |

## 5. Biểu Đồ Thành Phần

```
Docker (api:8001)
    |
    +-- /src/agent/ --> FastAPI app
    |       |-- routes/       endpoint handlers
    |       |-- backend/      LangGraph pipeline + services
    |       |-- utils/        config, vdb, retriever, embed, rerank
    |       |-- data_model/   request/response pydantic models
    |
    +-- pyproject.toml    dependencies (uv sync)
    +-- .env              secrets + config
    +-- Dockerfile        build image (uv:python3.13-bookworm-slim)
    +-- docker-compose.yml
    |
    +-- frontend/         Streamlit GUI (optional)
    +-- tests/            pytest suite
```

## 6. File Map

| Area | Path |
|---|---|
| API app entry | `src/agent/api.py` |
| Config | `src/agent/utils/config.py` |
| Qdrant VDB init | `src/agent/utils/vdb.py` |
| Embeddings | `src/agent/utils/embeddings.py` |
| Retriever | `src/agent/utils/retriever.py` |
| Reranker | `src/agent/utils/reranker.py` |
| LangGraph | `src/agent/backend/graph.py` |
| Retrieval node | `src/agent/backend/nodes/retrieval.py` |
| Embedding service | `src/agent/backend/services/embedding_management.py` |
| Migration script | `src/agent/scripts/migrate_dump_to_qdrant.py` |
| Test suite | `tests/` |
