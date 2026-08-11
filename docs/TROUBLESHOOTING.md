# Xử Lý Lỗi Thường Gặp — Troubleshooting Guide (v8.1.0)

## 1. API không start được (CrashLoopBackOff)

### Lỗi: `Connection refused [Errno 111]` với Qdrant

**Nguyên nhân**: API không kết nối được Qdrant.

> **Khác v6**: ở v7, API không crash khi Qdrant down (chỉ `/readyz` trả 503).
> Nếu thấy crash thật sự, nguyên nhân nằm ở chỗ khác — kiểm tra `docker logs`
> kỹ hơn.

**Checklist:**

```bash
# 1. Qdrant có chạy không?
docker ps | grep qdrant

# 2. Network test_network có tồn tại không?
docker network ls | grep test_network

# 3. Cả 2 container có cùng network không?
docker network inspect test_network | grep -E "qdrant|conversational"

# 4. Env QDRANT_URL trong container đúng không?
docker exec conversational-rag-api bash -c 'env | grep QDRANT'

# 5. Test kết nối từ container đến Qdrant
docker exec conversational-rag-api curl http://qdrant:6333/
```

**Fix:**

| Scenario | Fix |
|---|---|
| Qdrant chạy host, API chạy Docker | `.env`: `QDRANT_URL=http://host.docker.internal` |
| Qdrant chạy container khác, cùng network | `.env`: `QDRANT_URL=http://qdrant` (service name) |
| Qdrant chạy container khác, khác network | Gắn cùng network: `docker network connect test_network qdrant` |
| Chưa tạo network | `docker network create test_network` |

### Lỗi: `Failed to obtain server version` (Qdrant client warning)

Warning, không fatal. Qdrant client không check được version compatibility. Set
`check_compatibility=False` hoặc ignore.

### Lỗi: `/readyz` trả về 503

Xem nguyên nhân cụ thể ở `reason`:

```bash
curl -i http://localhost:8005/readyz
```

| `reason` | Nguyên nhân | Fix |
|---|---|---|
| `collection_missing` | Collection `QDRANT_COLLECTION_NAME` chưa tồn tại trên Qdrant | Chạy ingestion (repo ngoài) hoặc tạo collection thủ công qua Qdrant Dashboard |
| `qdrant_unreachable` | Qdrant down hoặc `QDRANT_URL` sai | Verify Qdrant running + URL correct |
| `qdrant_error` | Qdrant trả non-2xx (auth, network policy, ...) | Check Qdrant logs, `QDRANT_API_KEY` nếu dùng cloud |

### Lỗi: `API key is used with an insecure connection`

Warning, không fatal. Gọi API Qdrant không có SSL. Dùng HTTP cho local dev.

## 2. Remote Embedding / Rerank Server

### Lỗi: `Cannot reach remote embedding server` / httpx ConnectError

**Nguyên nhân**: `EMBEDDING_BASE_URL` sai, không set, hoặc remote server
(GPU self-host) đang down.

**Checklist:**

```bash
# 1. Env trong container đúng không?
docker exec conversational-rag-api bash -c 'env | grep EMBEDDING'

# 2. URL có ping được từ container không?
docker exec conversational-rag-api curl -i $EMBEDDING_BASE_URL/
```

**Fix:**

| Scenario | Fix |
|---|---|
| `EMBEDDING_BASE_URL` trống | Set trong `.env` (bắt buộc với `EMBEDDING_PROVIDER=remote`) |
| Remote server down | Restart GPU server, update `.env` + restart API |
| Network chậm / timeout | Tăng `EMBEDDING_TIMEOUT` (mặc định 60s) trong `.env` |
| Server trả non-200 | Xem §3 lỗi "remote /embed returns non-200" |

### Lỗi: `Cannot reach remote rerank server` / httpx ConnectError (Rerank)

**Nguyên nhân**: `RERANK_BASE_URL` sai hoặc không set — từ v8.1.0 default
`RERANK_PROVIDER=remote` và behavior **fail-fast**: nếu remote rerank không gọi
được, request `/rag/` sẽ lỗi (không tự fallback sang local `bge`).

**Fix:**

| Scenario | Fix |
|---|---|
| Chưa có rerank server | Set `RERANK_PROVIDER=bge` (local FlagEmbedding) hoặc `none` (passthrough) |
| `RERANK_BASE_URL` sai | Set URL khả dụng; nếu server bind loopback (127.0.0.1) ngoài host, dùng `http://host.docker.internal:8010` |
| Muốn giữ fail-fast | Sửa `RERANK_BASE_URL` cho đúng + restart API |

### Lỗi: `remote /embed returns non-200` / `remote /rerank returns non-200`

**Nguyên nhân**: remote server up nhưng endpoint trả lỗi (sai schema, server
đang load model, OOM trên GPU).

**Fix:**
- Mở server log (traceback Python).
- Thường là server đang load model lần đầu → đợi ~1-2 phút rồi retry.
- Nếu OOM trên GPU → giảm batch, restart runtime.

### Lỗi: Model load rất chậm trên remote server (phút)

Lần đầu server tải BGE-m3 (~2.2GB) từ HuggingFace → phụ thuộc bandwidth.
Sau khi model đã nằm trong runtime, các request sau nhanh. Repo Docker
image này **không** tải model — tất cả nằm trên remote server.

> Từ v8.1.0 default `RERANK_PROVIDER=remote` → server còn tải thêm reranker
> (~200MB) nếu dùng chung server. Nếu set `RERANK_PROVIDER=none` thì bỏ qua
> bước rerank.

## 3. Search / Retrieval

### Lỗi: Search trả về empty documents

```bash
# 1. Collection có data không?
curl http://localhost:6333/collections/documents/points/count

# 2. Collection name đúng không?
curl http://localhost:6333/collections

# 3. Test search trực tiếp Qdrant
curl -X POST http://localhost:6333/collections/documents/points/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0]*1024, "limit": 1}'
```

### Lỗi: `Unknown reranker provider: '...'`

**Nguyên nhân**: `RERANK_PROVIDER` trong `.env` không hợp lệ.

**Fix**: Chỉ chấp nhận `none`, `remote` (default), hoặc `bge`. Xem [CONFIGURATION.md](CONFIGURATION.md) §3.

### Lỗi: 422 Validation Error ở `/rag/` hoặc `/semantic/search`

Thường do thiếu field `messages` / `query` / `collection_name`. Kiểm tra request
body khớp schema trong [API_REFERENCE.md](API_REFERENCE.md).

## 4. Migration / Ingestion (LOẠI BỎ ở v7)

> Repo này **không còn** script migration hay endpoint ingestion. Nếu bạn
> thấy reference tới `python -m agent.scripts.migrate_dump_to_qdrant` —
> script đã xoá. Liên hệ team ingestion để biết repo của họ.

## 5. Kết nối mạng giữa các container

### Diagnostic Script (chạy từ host):

```bash
echo "=== Network Check ==="
docker network inspect test_network | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print([c['Name'] for c in d[0]['Containers'].values()])"

echo "=== Qdrant Direct ==="
docker exec conversational-rag-api curl -s http://qdrant:6333/ | head -c 200
echo

echo "=== Env Check ==="
docker exec conversational-rag-api bash -c 'echo QDRANT_URL=${QDRANT_URL:-MISSING}'
```

## 6. Out of Memory / Performance

**Symptom**: API chậm dần theo thời gian, hoặc crash khi có nhiều request.

**Check:**

```bash
docker stats conversational-rag-api --no-stream
```

**Fix**:
- Giới hạn memory container: `deploy.resources.limits.memory: 12G`
- Tắt rerank: `RERANK_PROVIDER=none` → bỏ HTTP rerank hop (~100ms/request)
- Load test trước khi deploy production
- Cache warm: request 1 query ngẫu nhiên sau deploy

## 7. Endpoints Đã Bỏ (404)

Nếu client cũ gọi các endpoint ở v6, sẽ nhận 404:

```bash
curl -i -X POST http://localhost:8005/collection/create/x?embeddings_size=1024
# HTTP/1.1 404 Not Found

curl -i -X POST http://localhost:8005/embeddings/documents?collection_name=x
# HTTP/1.1 404 Not Found

curl -i -X DELETE http://localhost:8005/embeddings/delete/x?collection_name=x
# HTTP/1.1 404 Not Found
```

> **Cập nhật client** để chỉ gọi `/rag/`, `/rag/stream`, `/semantic/search`,
> `/healthz`, `/readyz`. Ingestion chuyển sang repo ngoài.

## 8. Common Known Bugs (v8.1.0)

| Bug | Workaround | Status |
|---|---|---|
| `get_reranker(provider="cohere")` raise `ValueError` | Cohere/FlashRank đã bỏ — dùng `provider="remote"` (default), `"bge"` (local) hoặc `"none"` | By design |
| Remote rerank fail nhưng request lỗi (không trả kết quả) | Fail-fast by design: sửa `RERANK_BASE_URL` hoặc chuyển `RERANK_PROVIDER=none/bge` | By design |
| Frontend cũ (v6) gọi `/embeddings/documents` → 404 | Frontend phải làm việc với API v8 (chỉ retrieval); ingestion tách riêng | Migrate frontend |
| Conftest cũ dùng env var cũ (`AU_EMBED_MODEL_NAME`, `EMBEDDING_MODEL`) | Conftest v8.1 set env remote (`EMBEDDING_BASE_URL`, `RERANK_BASE_URL`, v.v.). `EMBEDDING_MODEL`/`SPARSE_MODEL`/`FUSION_ALGORITHM` đã bỏ | Remove old vars |
