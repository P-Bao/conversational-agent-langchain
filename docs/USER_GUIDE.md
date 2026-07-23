# Hướng Dẫn Sử Dụng API — User Guide

> Tài liệu dành cho end-user muốn gọi API RAG để lấy context cho ứng dụng của mình.

## 1. API Là Gì?

API này là một **Retrieval Service** (dịch vụ truy xuất tài liệu). Nó nhận câu hỏi
của bạn, tìm kiếm trong cơ sở dữ liệu các đoạn tài liệu liên quan nhất (chunks),
sắp xếp theo độ relevance, và trả về cho bạn.

**API KHÔNG tự sinh câu trả lời.** Nó chỉ trả về các đoạn tài liệu gốc để bạn
(hoặc LLM downstream) tự tổng hợp câu trả lời.

## 2. Kết Nối Nhanh

### Yêu cầu:

- API endpoint URL (ví dụ: `http://rag-api.organization.com:8001`)
- API đã được deploy và chạy

### Ví dụ Gọi API (cURL):

```bash
curl -X POST http://localhost:8001/rag/ \
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
      "page": 2,
      "source": "1706.03762v5.pdf",
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

def get_relevant_docs(query: str, api_url: str = "http://localhost:8001") -> list[dict]:
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
    print(f"[{doc['score']:.2f}] {doc['source']} (p.{doc['page']})")
    print(f"  {doc['text'][:150]}...")
    print()
```

### JavaScript / Node.js:

```javascript
const response = await fetch("http://localhost:8001/rag/", {
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

Mỗi document trong mảng `documents` có:

| Field | Luôn có? | Mô tả |
|---|---|---|
| `text` | Yes | Nội dung chunk (có thể cắt ngắn ~750-1500 ký tự) |
| `page` | Maybe | Số trang gốc (nếu là PDF) |
| `source` | Maybe | Tên file hoặc URL nguồn |
| `score` | Yes | Relevance score (0-1) từ reranker. Cao hơn = relevant hơn. Sắp xếp giảm dần. |
| `metadata` | Yes | Object chứa thông tin bổ sung (document_id, global_id, chunk_index...). Dùng cho traceability. |

## 5. Collection (Bộ Sưu Tập)

Dữ liệu được tổ chức thành các **collections**. Mỗi collection chứa documents của
một chủ đề / tổ chức / khoá học.

- Mặc định: `collection_name = "documents"`
- Nếu có nhiều collection, gọi API với `collection_name` khác nhau.

Để xem danh sách collection: `http://localhost:6333/collections` (Qdrant dashboard).

## 6. Streaming (Đọc Từng Dòng)

Nếu muốn hiển thị tiến trình khi retrieval:

```python
import httpx, json

with httpx.stream(
    "POST", "http://localhost:8001/rag/stream",
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

## 7. Direct Search (Không Rerank)

Endpoint `/semantic/search` trả về kết quả nhanh hơn (bỏ qua reranker nhưng
không có metadata đầy đủ).

```bash
curl -X POST http://localhost:8001/semantic/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention is all you need", "k": 3, "collection_name": "documents"}'
```

Phù hợp khi cần kết quả nhanh, không cần rerank chính xác.

## 8. Error Handling

| HTTP Status | Nghĩa | Xử lý |
|---|---|---|
| 200 | Success | Parse response JSON |
| 400 | Bad request (thiếu params, sai format) | Kiểm tra request body |
| 500 | Server error | Báo admin, xem log: `docker logs conversational-rag-api --tail 20` |
| Connection refused | API chưa chạy hoặc sai URL | Kiểm tra URL + port |

## 9. FAQs

**Q: Làm sao để biết API đang chạy?**

A: `curl http://localhost:8001/` trả về `"Welcome to the RAG Backend..."`

**Q: Kết quả trả về rỗng?**

A: Collection chưa có dữ liệu hoặc `collection_name` sai. Kiểm tra tại Qdrant dashboard.

**Q: Score cao = document rất phù hợp?**

A: Score từ BGE-reranker (normalized 0-1). Score cao hơn là relevant hơn.
Tuy nhiên không phải threshold tuyệt đối — cùng query, scores so sánh giữa các documents.

**Q: Có thể upload dữ liệu của tôi không?**

A: Được — dùng endpoint `POST /embeddings/documents`. Xem [API_REFERENCE.md](API_REFERENCE.md) section 6.

**Q: Timeout?**

A: Lần đầu API gọi có thể chậm do model load (2-5 phút). Các request sau nhanh hơn.
Nếu timeout, tăng timeout lên 120s cho request đầu tiên.
