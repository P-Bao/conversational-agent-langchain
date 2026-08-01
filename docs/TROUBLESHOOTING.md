# Xử Lý Lỗi Thường Gặp — Troubleshooting Guide v7.1.0

> **Khác v7.0.0:** Port `8005` (thay `8001`), network `ami-network` (thay
> `test_network`), reranker local cần GPU, query transformation optional qua
> Qwen, embedding server auth header.

## 1. API không start được (CrashLoopBackOff)

### Lỗi: `Connection refused [Errno 111]` với Qdrant

> Khác v6: ở v7 trở đi, API **không crash** khi Qdrant down — chỉ `/readyz`
> trả 503. Nếu thấy crash thật, kiểm tra `docker logs` kỹ hơn.

**Checklist:**

```bash
# 1. Qdrant có chạy không?
docker ps | grep qdrant

# 2. Network ami-network có tồn tại không?
docker network ls | grep ami-network

# 3. Cả 2 container có cùng network không?
docker network inspect ami-network | grep -E "qdrant|conversational"

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
| Qdrant chạy container khác, khác network | Gắn cùng network: `docker network connect ami-network qdrant` |
| Chưa tạo network | `docker network create ami-network` |

### Lỗi: `Cannot reach local reranker` / `CUDA out of memory`

**Nguyên nhân:** `RERANK_PROVIDER=bge` (default) nhưng host/container không
có GPU, hoặc GPU đầy.

**Fix:**

```bash
# 1. Kiểm tra GPU trong container
docker exec conversational-rag-api python -c "import torch; print(torch.cuda.is_available())"

# 2. Kiểm tra VRAM
nvidia-smi
```

| Scenario | Fix |
|---|---|
| Không có GPU | `.env`: `RERANK_PROVIDER=none` (passthrough) |
| GPU đầy | Stop process khác đang chiếm GPU, hoặc dùng GPU lớn hơn |
| Container không thấy GPU | Sửa `docker-compose.yml`: đảm bảo block `deploy.resources.reservations.devices` |

### Lỗi: `/readyz` trả về 503

```bash
curl -i http://localhost:8005/readyz
```

| `reason` | Nguyên nhân | Fix |
|---|---|---|
| `collection_missing` | Collection `QDRANT_COLLECTION_NAME` chưa tồn tại | Chạy ingestion (repo ngoài) hoặc tạo qua Qdrant Dashboard |
| `qdrant_unreachable` | Qdrant down hoặc `QDRANT_URL` sai | Verify Qdrant running + URL correct |
| `qdrant_error` | Qdrant trả non-2xx (auth, network policy) | Check Qdrant logs, `QDRANT_API_KEY` nếu dùng cloud |

### Lỗi: `API key is used with an insecure connection`

Warning, không fatal. Gọi API Qdrant không có SSL — bình thường cho local dev.

## 2. Remote Embedding Server

### Lỗi: `Cannot reach remote embedding server` / httpx ConnectError

**Nguyên nhân:** `EMBEDDING_BASE_URL` sai, không set, hoặc embedding-server
(repo ngoài) đang down.

**Checklist:**

```bash
# 1. Env trong container đúng không?
docker exec conversational-rag-api bash -c 'env | grep EMBEDDING'

# 2. URL có ping được từ container không?
docker exec conversational-rag-api curl -i $EMBEDDING_BASE_URL/

# 3. Trong Docker network: dùng service name (DNS), KHÔNG phải localhost
docker exec conversational-rag-api curl http://bge-m3-embed:8008/
```

**Fix:**

| Scenario | Fix |
|---|---|
| `EMBEDDING_BASE_URL` trống | Set trong `.env` theo repo embedding-server |
| Embedding-server down | `cd ../embedding-server && docker compose up -d` |
| Network chậm / timeout | Tăng `EMBEDDING_TIMEOUT` (mặc định 60s) |
| 401 / 403 (auth fail) | `EMBEDDING_API_KEY` trong `.env` phải khớp `BGE_API_KEY` trong `embedding-server/.env` |
| Endpoint trả non-200 | Xem logs `docker logs bge-m3-embed` |

### Lỗi: `remote /embed returns non-200`

**Nguyên nhân:** embedding-server up nhưng trả lỗi (sai schema, server đang
load model, OOM trên GPU).

**Fix:**
- `docker logs bge-m3-embed` xem traceback.
- Model load lần đầu → đợi ~1-2 phút rồi retry.
- OOM GPU trên embedding-server → giảm batch, restart container.

### Lỗi: Model load rất chậm trên embedding-server (phút)

Lần đầu tải BGE-m3 (~2.7GB) từ HuggingFace → phụ thuộc bandwidth. Sau khi
model đã nằm trong runtime, các request sau nhanh. **Repo API Docker image
không tải model embedding** — tất cả nằm trên embedding-server.

## 3. Query Transformation (Qwen)

### Lỗi: `Query transformation failed, falling back to original query`

Log này xuất hiện khi `QUERY_TRANSFORM_ENABLED=true` nhưng Qwen server không
reach được, hoặc LLM lỗi. **API vẫn hoạt động** — fallback về câu hỏi gốc.

**Checklist:**

```bash
# 1. Qwen server có chạy không?
curl http://localhost:8000/v1/models

# 2. Trong Docker network (nếu Qwen cùng ami-network)
docker exec conversational-rag-api curl http://qwen:8000/v1/models

# 3. Env trong container
docker exec conversational-rag-api bash -c 'env | grep QWEN'

# 4. Log chi tiết lỗi
docker logs conversational-rag-api 2>&1 | grep "Query transformation failed"
```

**Fix:**

| Scenario | Fix |
|---|---|
| Qwen server down | Start lại Qwen server (self-host vLLM/TGI/Ollama) |
| `QWEN_BASE_URL` sai | Update `.env`: dùng `http://qwen:8000/v1` trong Docker, `http://localhost:8000/v1` local |
| `QWEN_MODEL` sai tên | Xem `curl http://<qwen>/v1/models` để lấy đúng model name |
| Rate limit | Giảm concurrency hoặc dùng GPU cho Qwen |
| Không cần query transform | `.env`: `QUERY_TRANSFORM_ENABLED=false` → pipeline legacy |

### Lỗi: Query transform bật nhưng chạy dường như không có effect

- Verify `QUERY_TRANSFORM_ENABLED=true` (chứ không phải "true" string được
  bỏ qua). Check trong log: phải thấy `Query transformation done: ...`.
- Nếu chỉ thấy `Query transformation failed`, LLM đã fail → fallback về
  query gốc. Xem mục lỗi trên.

## 4. Reranker Local (BGE-reranker-v2-m3)

### Lỗi: Reranker model tải rất chậm lần đầu

`BAAI/bge-reranker-v2-m3` (~1.1GB) tải lần đầu từ HuggingFace → cache vào
`hf-cache/` volume. Sau đó model nằm sẵn, request sau nhanh.

```bash
# Kiểm tra cache
ls -lh hf-cache/huggingface/
```

### Lỗi: `Unknown reranker provider: '...'`

Chỉ chấp nhận `bge` (default), `remote` (legacy), `none` (passthrough).
Giá trị `cohere` / `flashrank` đã bỏ hoàn toàn. Xem [CONFIGURATION.md](CONFIGURATION.md) §4.

### Lỗi: Rerank chậm sau khi đã tải model

```bash
docker exec conversational-rag-api python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Nếu `CUDA: False` → FlagReranker chạy CPU (chậm). Kiểm tra GPU reservation
trong `docker-compose.yml` + driver NVIDIA.

## 5. Search / Retrieval

### Lỗi: Search trả về empty documents

```bash
# 1. Collection có data không?
curl http://localhost:6333/collections/documents/points/count

# 2. Collection name đúng không?
curl http://localhost:6333/collections

# 3. Test search trực tiếp Qdrant (cần dense vector 1024-dim)
curl -X POST http://localhost:6333/collections/documents/points/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0]*1024, "limit": 1}'
```

### Lỗi: 422 Validation Error ở `/rag/` hoặc `/semantic/search`

Thường do thiếu field `messages` / `query`. Kiểm tra request body khớp schema
trong [API_REFERENCE.md](API_REFERENCE.md).

### Lỗi: `/rag/` trả document trùng lặp (multi-query mode)

Khi `QUERY_TRANSFORM_ENABLED=true`, pipeline retrieve tuần tự ~5-7 queries.
Dedupe dựa vào `metadata.global_id`. Nếu chunks không có `global_id`:

- Fallback: hash `page_content`. Vẫn có thể trùng nếu whitespace khác nhau
  một chút.
- Fix phía ingestion (repo ngoài): đảm bảo tất cả chunks có `global_id` metadata.

## 6. Kết nối mạng giữa các container

### Diagnostic Script:

```bash
echo "=== Network Check ==="
docker network inspect ami-network | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print([c['Name'] for c in d[0]['Containers'].values()])"

echo "=== Qdrant Direct ==="
docker exec conversational-rag-api curl -s http://qdrant:6333/ | head -c 200
echo

echo "=== Embedding Server ==="
docker exec conversational-rag-api curl -s http://bge-m3-embed:8008/ | head -c 200
echo

echo "=== Qwen Server (optional) ==="
docker exec conversational-rag-api curl -s http://qwen:8000/v1/models | head -c 200
echo

echo "=== Env Check ==="
docker exec conversational-rag-api bash -c 'echo QDRANT_URL=${QDRANT_URL:-MISSING}; echo EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL:-MISSING}; echo QWEN_BASE_URL=${QWEN_BASE_URL:-MISSING}'
```

## 7. Out of Memory / Performance

**Symptom:** API chậm dần, hoặc crash khi nhiều request.

**Check:**

```bash
docker stats conversational-rag-api --no-stream
nvidia-smi
```

**Fix:**

- Giới hạn memory container: `deploy.resources.limits.memory: 6G`
- `RERANK_PROVIDER=none` → giảm ~1.4GB VRAM + CPU
- `QUERY_TRANSFORM_ENABLED=false` → giảm ~2s latency, giảm LLM calls
- Cache warm: request 1 query ngẫu nhiên sau deploy
- Load test trước khi deploy production

## 8. Endpoints Đã Bỏ (404)

Nếu client cũ gọi các endpoint từ v6, sẽ nhận 404:

```bash
curl -i -X POST http://localhost:8005/collection/create/x?embeddings_size=1024
# HTTP/1.1 404 Not Found

curl -i -X POST http://localhost:8005/embeddings/documents?collection_name=x
# HTTP/1.1 404 Not Found

curl -i -X DELETE http://localhost:8005/embeddings/delete/x?collection_name=x
# HTTP/1.1 404 Not Found
```

> Cập nhật client chỉ gọi `/rag/`, `/rag/stream`, `/semantic/search`,
> `/healthz`, `/readyz`. Ingestion chuyển sang repo ngoài.

## 9. Common Known Bugs (v7.1.0)

| Bug | Workaround | Status |
|---|---|---|
| `get_reranker(provider="cohere")` raise `ValueError` | Cohere/FlashRank đã bỏ — dùng `bge` (default) / `remote` (legacy) / `none` | By design |
| Frontend cũ (v6) gọi `/embeddings/documents` → 404 | Frontend phải làm việc với API v7 (chỉ retrieval); ingestion tách riêng | Migrate |
| Conftest cũ dùng env var `AU_EMBED_MODEL_NAME`, `EMBEDDING_MODEL` | Đã bỏ. Set `EMBEDDING_BASE_URL` + `EMBEDDING_API_KEY` thay thế | Remove old vars |
| Query transform fail nhưng vẫn trả docs (fallback) | By design — không break pipeline. Kiểm tra Qwen server | By design |
| Docker image nặng ~6GB | Do PyTorch base + FlagEmbedding + transformers. Có thể multi-stage build để giảm size | Future improvement |
