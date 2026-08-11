# API Reference — Retrieval & Search API v8.1.0

> Base URL: `http://localhost:8005`
> OpenAPI: `http://localhost:8005/docs` (Swagger UI)
>
> Repo chỉ retrieval & search. Collection / embedding / delete thuộc hệ thống
> ngoài (xem [DATA_INGESTION.md](DATA_INGESTION.md)).

## 1. GET `/` — Welcome

**Response** (200):
```
Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!
```

## 2. GET `/healthz` — Liveness probe

Trả về 200 ngay khi process phục vụ HTTP. Không gọi Qdrant.

**Response** (200):
```json
{ "status": "ok" }
```

Dùng cho Docker `healthcheck` (process còn sống).

## 3. GET `/readyz` — Readiness probe

Kiểm tra Qdrant có sẵn sàng + collection đang khai báo tồn tại.

**Response** (200):
```json
{ "status": "ready", "collection": "default" }
```

**Response** (503 — collection missing):
```json
{ "status": "fail", "reason": "collection_missing", "collection": "default" }
```

**Response** (503 — Qdrant unreachable):
```json
{ "status": "fail", "reason": "qdrant_unreachable", "details": "Connection refused" }
```

**Response** (503 — Qdrant non-2xx):
```json
{ "status": "fail", "reason": "qdrant_error", "details": "..." }
```

## 4. POST `/rag/` — Retrieval Query

Lấy danh sách document chunks liên quan đến query. Không sinh câu trả lời.
Pipeline: LangGraph (`retriever` node) -> hybrid dense+sparse retrieval (remote BGE-m3) -> optional rerank.

**Request Body** (`RAGRequest`):

```json
{
  "messages": [
    { "role": "user", "content": "Thế nào là attention trong Transformer?" }
  ],
  "collection_name": "documents",
  "top_k": 5
}
```

- `top_k` (optional, `ge=1`, `le=40`): số doc trả về sau rerank. Falls back to `RERANK_TOP_K` config khi omitted; clamp thêm về `min(top_k, k)` (số doc retrieve được).

**Response** (200, `RetrievalResponse`):

```json
{
  "query": "Thế nào là attention trong Transformer?",
  "documents": [
    {
      "text": "The attention mechanism allows the model to focus on relevant parts...",
      "score": 0.89,
      "metadata": {
        "page": 2,
        "source": "1706.03762v5.pdf",
        "document_id": "abc123",
        "chunk_index": 5,
        "global_id": "550e8400-e29b-41d4-a716-446655440000"
      }
    }
  ]
}
```

> `page`/`source` đã bỏ khỏi top-level response (v8.0). Vẫn có trong `metadata`.

Khi `RERANK_PROVIDER=none` thì `score` có thể `null` (chỉ retrieval thuần).
Khi `RERANK_PROVIDER=remote` thì `score` là rerank score đã chuẩn hoá về [0, 1] từ rerank-server `:8010`.
Khi `RERANK_PROVIDER=remote` mà server lỗi/timeout → fail-fast (HTTP 500, không tự động fallback).

**Curl example**:

```bash
curl -X POST http://localhost:8005/rag/ \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is attention?"}], "top_k": 5}'
```

**Python example**:

```python
import httpx

resp = httpx.post("http://localhost:8005/rag/", json={
    "messages": [{"role": "user", "content": "What is attention?"}],
    "top_k": 5,
})
data = resp.json()
for doc in data["documents"]:
    print(f"[{doc['score']:.2f}] {doc['text'][:100]}...")
```

## 5. POST `/rag/stream` — Streaming Retrieval

Stream kết quả retrieval dạng NDJSON. Client đọc từng dòng.

**Request Body**: giống `/rag/`.

**Response** (200, `application/x-ndjson`):

```
{"type": "status", "data": "Starting request..."}
{"type": "status", "data": "Searching documents..."}
{"type": "status", "data": "Found 5 documents."}
{"type": "documents", "data": [{"text": "...", "metadata": {...}}, ...]}
{"type": "status", "data": "Done."}
```

**Python async example**:

```python
import httpx, json

async with httpx.AsyncClient() as client:
    async with client.stream("POST", "http://localhost:8005/rag/stream",
        json={"messages": [{"role": "user", "content": "test"}]}) as resp:
        async for line in resp.aiter_lines():
            event = json.loads(line)
            if event["type"] == "documents":
                print(f"Received {len(event['data'])} documents")
```

## 6. POST `/semantic/search` — Direct Semantic Search

Tìm kiếm trực tiếp hybrid dense+sparse search, không qua graph pipeline (không rerank,
không graph). Phù hợp khi cần kết quả nhanh.

**Request Body** (`SearchParams`):

```json
{
  "query": "attention mechanism",
  "k": 3,
  "collection_name": "documents"
}
```

**Response** (200, `SearchResponse[]`):

```json
[
  {
    "text": "The attention mechanism..."
  }
]
```

Khi không có document nào, trả về:
```json
{ "message": "No documents found." }
```

## 7. Summary Table

| Method | Path | Input | Output | Mô tả |
|---|---|---|---|---|
| GET | `/` | — | plain text | Welcome |
| GET | `/healthz` | — | `{status: ok}` | Liveness probe |
| GET | `/readyz` | — | `{status, collection}` hoặc `{status, reason}` | Readiness probe (Qdrant + collection) |
| POST | `/rag/` | `RAGRequest` | `RetrievalResponse` | Hybrid dense+sparse retrieval (remote BGE-m3) + optional rerank (LangGraph) |
| POST | `/rag/stream` | `RAGRequest` | NDJSON stream | Streaming retrieval |
| POST | `/semantic/search` | `SearchParams` | `SearchResponse[]` | Direct dense search (no rerank) |

## 8. Endpoints Đã Loại Bỏ (v7)

Các endpoint sau **đã được loại bỏ** khỏi repo này — chúng thuộc về hệ thống
ngoài quản lý Qdrant. Tất cả trả về `404 Not Found`:

- `POST /collection/create/{name}`
- `POST /embeddings/documents`
- `POST /embeddings/string/`
- `DELETE /embeddings/delete/{source}`

Migration script `python -m agent.scripts.migrate_dump_to_qdrant` cũng đã
chuyển sang repo ingestion ngoài.

## 9. Error Handling

Tất cả endpoints đều có global exception handler (`agent/api.py`).

**Response** (500):

```json
{
  "error": "Internal Server Error",
  "details": "Error message string"
}
```

Common HTTP status codes:

| Code | Meaning |
|---|---|
| 200 | Success |
| 404 | Not found — endpoint đã bị loại bỏ ở v7 hoặc sai path |
| 422 | Validation error (request body sai schema) |
| 500 | Internal error (xem `details` + container logs) |
| 503 | Readiness failure (chỉ `/readyz`) |
