# Xử Lý Lỗi Thường Gặp — Troubleshooting Guide (v7.0.0)

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
curl -i http://localhost:8001/readyz
```

| `reason` | Nguyên nhân | Fix |
|---|---|---|
| `collection_missing` | Collection `QDRANT_COLLECTION_NAME` chưa tồn tại trên Qdrant | Chạy ingestion (repo ngoài) hoặc tạo collection thủ công qua Qdrant Dashboard |
| `qdrant_unreachable` | Qdrant down hoặc `QDRANT_URL` sai | Verify Qdrant running + URL correct |
| `qdrant_error` | Qdrant trả non-2xx (auth, network policy, ...) | Check Qdrant logs, `QDRANT_API_KEY` nếu dùng cloud |

### Lỗi: `API key is used with an insecure connection`

Warning, không fatal. Gọi API Qdrant không có SSL. Dùng HTTP cho local dev.

## 2. Model Download / Loading

### Lỗi: `OSError: Can't load tokenizer for 'BAAI/bge-m3'`

**Nguyên nhân**: Không truy cập được HuggingFace Hub (network restriction / proxy).

**Fix:**

```bash
# Dùng mirror cho China / nội bộ
HF_ENDPOINT=https://hf-mirror.com docker compose up --build -d

# Hoặc download model trước, mount thủ công
uv run huggingface-cli download BAAI/bge-m3
```

Nếu môi trường không có internet, tải model từ máy khác và mount vào container:

```yaml
volumes:
  - /local/path/to/models:/root/.cache/huggingface
```

### Lỗi: `RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'`

**Nguyên nhân**: CPU không support FP16. Code đã fix bằng `use_fp16 = torch.cuda.is_available()`. Nếu vẫn gặp:

**Fix**: Kiểm tra torch version. Cài CPU-only torch nếu không có GPU.

### Lỗi: Model load rất chậm (phút)

Lần đầu load model (2.2GB) từ HuggingFace → phụ thuộc bandwidth.
Các lần sau dùng cache volume `bge_hf_cache`. Nếu vẫn chậm → kiểm tra tốc độ mạng.

> Ở v7 mặc định `RERANK_PROVIDER=none` → **chỉ load embedding** (1 model).
> Nếu set `RERANK_PROVIDER=bge` còn phải load thêm reranker (~200MB thời gian).

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

### Lỗi: `Unknown fusion_algorithm: '...'`

**Nguyên nhân**: `FUSION_ALGORITHM` trong `.env` không hợp lệ.

**Fix**: Chỉ chấp nhận `rrf` hoặc `dbsf`. Xem [CONFIGURATION.md](CONFIGURATION.md) §3.

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
- Tắt reranker: `RERANK_PROVIDER=none` → giảm ~2GB RAM + ~100ms/request
- Load test trước khi deploy production
- Cache warm: request 1 query ngẫu nhiên sau deploy

## 7. Endpoints Đã Bỏ (404)

Nếu client cũ gọi các endpoint ở v6, sẽ nhận 404:

```bash
curl -i -X POST http://localhost:8001/collection/create/x?embeddings_size=1024
# HTTP/1.1 404 Not Found

curl -i -X POST http://localhost:8001/embeddings/documents?collection_name=x
# HTTP/1.1 404 Not Found

curl -i -X DELETE http://localhost:8001/embeddings/delete/x?collection_name=x
# HTTP/1.1 404 Not Found
```

> **Cập nhật client** để chỉ gọi `/rag/`, `/rag/stream`, `/semantic/search`,
> `/healthz`, `/readyz`. Ingestion chuyển sang repo ngoài.

## 8. Common Known Bugs (v7.0.0)

| Bug | Workaround | Status |
|---|---|---|
| `get_reranker(provider="cohere")` raise `ValueError` | Cohere/FlashRank đã bỏ ở v7 — chuyển sang `provider="none"` hoặc `"bge"` | By design |
| Frontend cũ (v6) gọi `/embeddings/documents` → 404 | Frontend phải làm việc với API v7 (chỉ retrieval); ingestion tách riêng | Migrate frontend |
| Conftest cũ dùng env var cũ (`AU_EMBED_MODEL_NAME`) | Conftest v7 chỉ set các env mới (`EMBEDDING_MODEL`, v.v.). Code Config có alias backward-compat | Backward-compat |
