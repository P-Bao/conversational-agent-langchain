# Nhập Dữ Liệu — Data Ingestion

> **QUAN TRỌNG (v7.0.0):** Repo này **không còn** thực hiện ingestion (upload
> file, embed, tạo collection, xoá document, migration Mongo dump).
> Tất cả các endpoint và script sau đã bị **loại bỏ**:
>
> - `POST /collection/create/{name}`
> - `POST /embeddings/documents`
> - `POST /embeddings/string/`
> - `DELETE /embeddings/delete/{source}`
> - `python -m agent.scripts.migrate_dump_to_qdrant`
> - `python -m agent.scripts.load_dummy_data`
>
> Toàn bộ ingestion pipeline nằm ở **hệ thống quản lý Qdrant riêng**
> (ngoài repo này).

## 1. Vì sao tách ingestion khỏi retrieval?

| Trước (v6) | Sau (v7) |
|---|---|
| Một repo FastAPI vừa đọc vừa ghi Qdrant | Repo này chỉ **đọc** Qdrant |
| Ingestion dùng chung embedding model với retrieval, dễ xung đột tài nguyên | Ingestion chạy repo riêng, có thể scale riêng và không ảnh hưởng retrieval latency |
| API public lộ ra ngoài có `/embeddings/documents` → rủi ro upload OOM / RFI | Endpoint file upload đã bỏ hoàn toàn, attack surface nhỏ hơn |
| Một team chịu trách nhiệm full pipeline | Tách team ingestion (collection mgmt) vs team retrieval (latency, SLA search) |

## 2. Hệ thống ingestion liên quan

Các bước ingestion (chunk → embed → upsert Qdrant) hiện được thực hiện bởi:

- **Một repo ingestion riêng** (ví dụ: repo có tên `qdrant-ingestion-pipeline`) —
  đọc Mongo dump / API upload, embed bằng BGE-m3 (chạy server HTTP riêng hoặc
  trong pipeline ingestion), upsert vào Qdrant.
- **Qdrant Dashboard / REST API** — nếu chỉ cần thao tác thủ công trên dev.

Liên hệ team ingestion để biết:
- Collection name & schema (dense vector size 1024 + sparse named vector cho hybrid).
- Chunking strategy (Markdown-aware 1500/100 hay simple 750/200).
- Checkpoint / resume format.
- Pipeline khởi chạy & monitoring.

## 3. Schema của Qdrant mà repo này đọc

Để repo retrieval đọc đúng, collection cần thoả:

| Field | Requirement |
|---|---|
| `vectors_config` | Dense vector size 1024 (BGE-m3), distance = `COSINE`; sparse named vector (BGE-m3 sparse) nếu dùng hybrid fusion. |
| `payload` | tối thiểu `source` (string), `page` (int) — repo này đọc qua `metadata.source`, `metadata.page` |

> Nếu collection không tồn tại, `/readyz` sẽ trả `503` với `reason: collection_missing`.

## 4. Smoke test sau khi ingestion xong

```bash
# 1. Health
curl http://<rag-api>:8005/healthz
# {"status":"ok"}

# 2. Readiness — phải có collection
curl http://<rag-api>:8005/readyz
# {"status":"ready","collection":"documents"}

# 3. Search thử — kiểm tra retrieval hoạt động
curl -X POST http://<rag-api>:8005/semantic/search \
  -H "Content-Type: application/json" \
  -d '{"query":"smoke test","k":3,"collection_name":"default"}'
```

## 5. FAQ

**Q: Tôi cần nạp 1 file PDF nhanh cho dev, có endpoint nào không?**

A: Không. Liên hệ team ingestion để nạp qua repo ingestion riêng hoặc dùng
Qdrant Dashboard.

**Q: Tôi cần xoá 1 document đã upsert nhầm?**

A: Dùng Qdrant Dashboard hoặc gọi trực tiếp Qdrant REST API (Collections API
+ Points selector theo metadata filter).

**Q: Repo này có thể tự động tạo collection khi chưa có?**

A: Không — đây là behavior cố ý của v7. `/readyz` fail nhanh để bạn biết
chạy ingestion trước.
