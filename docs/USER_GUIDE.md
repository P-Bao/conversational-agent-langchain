# Hướng Dẫn Sử Dụng API — User Guide (v8.1.0)

> Tài liệu dành cho end-user / caller muốn gọi API retrieval & search.

## 1. API Là Gì?

API này là một **Retrieval Service** (dịch vụ truy xuất tài liệu). Nó nhận câu hỏi
của bạn, tìm kiếm trong cơ sở dữ liệu Qdrant các đoạn tài liệu liên quan nhất (chunks),
sắp xếp theo độ relevance, và trả về cho bạn.

**API KHÔNG tự sinh câu trả lời.** Nó chỉ trả về các đoạn tài liệu gốc để bạn
(hoặc LLM downstream) tự tổng hợp câu trả lời.

> **Lưu ý v7.0.0:** Các endpoint upload file / tạo collection / xoá document đã
> được chuyển sang hệ thống ngoài. API chỉ cung cấp retrieval & search.

> **Lưu ý v8.1.0:** Rerank mặc định là **remote** (qua HTTP server) — `score` luôn
> có nếu remote hoạt động. `top_k` giới hạn 1–40. Bỏ `page`/`source` khỏi response
> top-level (chỉ còn trong `metadata`).

## 2. Kết Nối Nhanh

### Yêu cầu:

- API endpoint URL (ví dụ: `http://rag-api.example.com:8005`)
- API đã được deploy và chạy
- Qdrant đã có sẵn collection (do hệ thống ingestion ngoài quản lý)

### Health check trước:

```bash
# Process còn sống?
curl http://localhost:8005/healthz
# {"status":"ok"}

# Qdrant + collection ready?
curl http://localhost:8005/readyz
# {"status":"ready","collection":"documents"}
```

### Ví dụ Gọi API (cURL):

```bash
curl -X POST http://localhost:8005/rag/ \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Attention trong Transformer hoạt động thế nào?"}
    ],
    "collection_name": "documents"
  }'
```

### Response:

```json
{
  "query": "Attention trong Transformer hoạt động thế nào?",
  "documents": [
    {
      "text": "The attention mechanism allows the model...",
      "score": 0.89,
      "metadata": {
        "source": "1706.03762v5.pdf",
        "page": 2,
        "document_id": "abc123",
        "global_id": "550e8400-e29b-41d4-a716-446655400000"
      }
    }
  ]
}
```

## 3. Xử Lý Dữ Liệu Trả Về

### Python:

```python
import httpx

def get_relevant_docs(query: str, api_url: str = "http://localhost:8005") -> list[dict]:
    resp = httpx.post(
        f"{api_url}/rag/",
        json={
            "messages": [{"role": "user", "content": query}],
            "collection_name": "documents",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["documents"]

docs = get_relevant_docs("Cách tính điểm GPA?")
for doc in docs:
    print(f"[{doc['score']:.2f}] {doc['metadata'].get('source')} (p.{doc['metadata'].get('page')})")
    print(f"  {doc['text'][:150]}...")
    print()
```

### JavaScript / Node.js:

```javascript
const response = await fetch("http://localhost:8005/rag/", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "What is attention?" }],
    collection_name: "documents",
  }),
});
const data = await response.json();
console.log(data.documents);
```

## 4. Các Loại Dữ Liệu Trong Response

Mỗi document trong mảng `documents`:

| Field | Luôn có? | Mô tả |
|---|---|---|
| `text` | Yes | Nội dung chunk |
| `score` | Maybe (null nếu no rerank) | Relevance score (0-1) từ reranker remote; `null` nếu `RERANK_PROVIDER=none` hoặc rerank lỗi |
| `metadata` | Yes | Object chứa thông tin bổ sung (document_id, global_id, chunk_index, page, source...). Dùng cho traceability. |

> `page`/`source` chỉ còn trong `metadata` (bỏ khỏi top-level từ v8.1.0).

## 5. Collection

Dữ liệu được tổ chức thành các **collections** trên Qdrant. Mỗi collection
chứa documents của một chủ đề / tổ chức / khoá học.

- Mặc định: `collection_name = "default"` (cấu hình qua `QDRANT_COLLECTION_NAME`)
- API chấp nhận `collection_name` khác nhau cho mỗi request.
- **Collections do hệ thống ngoài dựng.** API không có endpoint tạo / xoá collection.

Để xem danh sách collection: `http://<qdrant>:6333/collections` (Qdrant
dashboard) hoặc hỏi team ingestion.

## 6. Streaming (Đọc Từng Dòng)

Nếu muốn hiển thị tiến trình khi retrieval:

```python
import httpx, json

with httpx.stream(
    "POST", "http://localhost:8005/rag/stream",
    json={"messages": [{"role": "user", "content": query}], "collection_name": "documents"},
    timeout=30,
) as resp:
    for line in resp.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        if event["type"] == "status":
            print(f"Status: {event['data']}")
        elif event["type"] == "documents":
            print(f"Received {len(event['data'])} documents")
            for doc in event["data"]:
                print(f"  - {doc['text'][:100]}...")
```

## 7. Direct Search (Không Rerank, Không Graph)

Endpoint `/semantic/search` trả về kết quả nhanh hơn (bỏ qua graph pipeline + rerank):

```bash
curl -X POST http://localhost:8005/semantic/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention is all you need", "k": 3, "collection_name": "documents"}'
```

Phù hợp khi cần kết quả nhanh, không cần rerank.

## 8. Health Endpoint Cho Monitoring

Nếu bạn dùng API làm backend cho app, có thể monitor bằng:

```python
def is_api_ready() -> bool:
    try:
        r = httpx.get("http://localhost:8005/readyz", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ready"
    except httpx.RequestError:
        return False

assert is_api_ready(), "RAG API chưa ready — kiểm tra Qdrant + collection"
```

## 9. Error Handling

| HTTP Status | Nghĩa | Xử lý |
|---|---|---|
| 200 | Success | Parse response JSON |
| 404 | Endpoint đã bỏ ở v7 hoặc sai path | Xem [API_REFERENCE.md](API_REFERENCE.md) để biết endpoint hợp lệ |
| 422 | Validation error (sai schema) | Kiểm tra request body |
| 500 | Server error | Báo admin, xem log: `docker logs conversational-rag-api --tail 20` |
| 503 | Readiness fail (chỉ `/readyz`) | Qdrant down hoặc collection missing |
| Connection refused | API chưa chạy hoặc sai URL | Kiểm tra URL + port (8005) |

## 10. FAQs

**Q: Làm sao để biết API đang chạy?**

A: `curl http://localhost:8005/healthz` trả về `{"status":"ok"}`.

**Q: Làm sao biết Qdrant + collection ready?**

A: `curl http://localhost:8005/readyz` → 200 + `{"status":"ready", "collection":"..."}`.

**Q: Kết quả trả về rỗng?**

A: Collection chưa có dữ liệu hoặc `collection_name` sai. Liên hệ team
ingestion hoặc kiểm tra tại Qdrant dashboard.

**Q: Score cao = document rất phù hợp?**

A: Score từ reranker (BGE-reranker qua remote server, normalized 0-1). Score cao hơn
là relevant hơn. Tuy nhiên không phải threshold tuyệt đối — cùng query, scores so
sánh giữa các documents. Nếu `RERANK_PROVIDER=none`, `score` sẽ là `null`.

**Q: Tôi muốn upload file PDF của tôi lên hệ thống?**

A: Repo này không upload được. Liên hệ team ingestion để nạp data qua hệ
thống của họ.

**Q: Tôi muốn chạy test golden questions mà chưa có collection?**

A: Golden questions dùng cho DeepEval — cần Qdrant có data thật + collection
đã được ingestion ngoài dựng sẵn. Liên hệ team ingestion trước khi chạy
EVALUATION.

**Q: Timeout?**

A: Lần đầu API gọi có thể chậm do remote embedding server đang load model
(1-2 phút). Repo Docker image không load model — tất cả nằm trên remote server.
Nếu timeout, tăng `EMBEDDING_TIMEOUT` / timeout client lên 120s cho request đầu.
