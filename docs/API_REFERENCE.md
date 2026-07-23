# API Reference — Conversational Agent LangChain v6.0.0

> Base URL: `http://localhost:8001`
> OpenAPI: `http://localhost:8001/docs` (Swagger UI)

## 1. GET `/` — Health Check

**Response** (200):
```
Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!
```

## 2. POST `/rag/` — Retrieval Query

Lấy danh sách document chunks liên quan đến query của người dùng.
Không sinh câu trả lời — chỉ trả về context.

**Request Body** (`RAGRequest`):

```json
{
  "messages": [
    { "role": "user", "content": "Thế nào là attention trong Transformer?" }
  ],
  "collection_name": "documents"
}
```

**Response** (200, `RetrievalResponse`):

```json
{
  "query": "Thế nào là attention trong Transformer?",
  "documents": [
    {
      "text": "The attention mechanism allows the model to focus on relevant parts...",
      "page": 2,
      "source": "1706.03762v5.pdf",
      "score": 0.89,
      "metadata": {
        "page": 2,
        "source": "1706.03762v5.pdf",
        "document_id": "abc123",
        "chunk_index": 5,
        "global_id": "550e8400-e29b-41d4-a716-446655400000"
      }
    }
  ]
}
```

**Curl example**:

```powershell
$body = @{
    messages = @(@{ role = "user"; content = "What is attention?" })
    collection_name = "documents"
} | ConvertTo-Json -Compress

curl -X POST http://localhost:8001/rag/ `
  -H "Content-Type: application/json" `
  -d $body
```

**Python example**:

```python
import httpx

resp = httpx.post("http://localhost:8001/rag/", json={
    "messages": [{"role": "user", "content": "What is attention?"}],
    "collection_name": "documents",
})
data = resp.json()
for doc in data["documents"]:
    print(f"[{doc['score']:.2f}] {doc['text'][:100]}...")
```

## 3. POST `/rag/stream` — Streaming Retrieval

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
    async with client.stream("POST", "http://localhost:8001/rag/stream",
        json={"messages": [{"role": "user", "content": "test"}]}) as resp:
        async for line in resp.aiter_lines():
            event = json.loads(line)
            if event["type"] == "documents":
                print(f"Received {len(event['data'])} documents")
```

## 4. POST `/semantic/search` — Direct Semantic Search

Tìm kiếm trực tiếp hybrid search, không qua graph pipeline (không rerank, không full metadata).

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
    "text": "The attention mechanism...",
    "page": 2,
    "source": "1706.03762v5.pdf"
  }
]
```

## 5. POST `/collection/create/{name}` — Create Collection

Tạo collection mới trong Qdrant với dense + sparse vector config.

**Path params**: `collection_name` (string), `embeddings_size` (query, int, 1-5000).

**Curl**:

```powershell
curl -X POST "http://localhost:8001/collection/create/my_collection?embeddings_size=1024"
```

**Response** (200):

```json
{ "message": "Collection my_collection created." }
```

## 6. POST `/embeddings/documents` — Upload & Embed Files

Upload PDF/txt files → chunk → embed → upsert vào Qdrant collection.

**Query params**: `collection_name` (string, required), `file_ending` (default `.pdf`).

**Request**: `multipart/form-data` với field `files` (multiple files).

**Curl**:

```powershell
curl -X POST "http://localhost:8001/embeddings/documents?collection_name=default&file_ending=.pdf" `
  -F "files=@test.pdf" `
  -F "files=@paper.pdf"
```

**Python**:

```python
import httpx

with open("test.pdf", "rb") as f:
    files = {"files": ("test.pdf", f, "application/pdf")}
    resp = httpx.post(
        "http://localhost:8001/embeddings/documents",
        params={"collection_name": "default", "file_ending": ".pdf"},
        files=files,
    )
print(resp.json())  # {"status": "success", "files": ["test.pdf"]}
```

**Response** (200, `EmbeddingResponse`):

```json
{
  "status": "success",
  "files": ["attention.pdf", "transformer.pdf"]
}
```

## 7. POST `/embeddings/string/` — Embed Text String

Embed raw text string (viết thành file .txt tạm rồi xử lý).

**Query params**: `collection_name` (string, required).

**Request Body** (`EmbeddTextRequest`):

```json
{
  "text": "Attention is all you need. The transformer architecture...",
  "file_name": "transformer_intro",
  "separator": "###"
}
```

**Response** (200, `EmbeddingResponse`):

```json
{ "status": "success", "files": ["transformer_intro"] }
```

## 8. DELETE `/embeddings/delete/{source}` — Delete by Source

Xóa tất cả points trong Qdrant có `metadata.source == source`.

**Query params**: `collection_name` (string, required).

**Path params**: `source` — giá trị metadata.source cần xóa.

**Curl**:

```powershell
curl -X DELETE "http://localhost:8001/embeddings/delete/old_document.pdf?collection_name=documents"
```

**Response** (200, `UpdateResult`):

```json
{
  "result": "acknowledged",
  "status": "completed",
  "operation_id": 0
}
```

## 9. Summary Table

| Method | Path | Input | Output | Mô tả |
|---|---|---|---|---|
| GET | `/` | — | plain text | Health check |
| POST | `/rag/` | `RAGRequest` | `RetrievalResponse` | Hybrid retrieval + rerank |
| POST | `/rag/stream` | `RAGRequest` | NDJSON stream | Streaming retrieval |
| POST | `/semantic/search` | `SearchParams` | `SearchResponse[]` | Direct search (no rerank) |
| POST | `/collection/create/{name}` | path + query | JSON | Create Qdrant collection |
| POST | `/embeddings/documents` | multipart | `EmbeddingResponse` | Upload file → embed |
| POST | `/embeddings/string/` | `EmbeddTextRequest` | `EmbeddingResponse` | Embed raw text string |
| DELETE | `/embeddings/delete/{source}` | path + query | `UpdateResult` | Delete by source metadata |

## 10. Error Handling

Tất cả endpoints đều có global exception handler (`agent/api.py:42-49`).

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
| 400 | Bad request (missing params, validation error) |
| 404 | Not found (trả về mặc định FastAPI) |
| 500 | Internal error (xem details + container logs) |
