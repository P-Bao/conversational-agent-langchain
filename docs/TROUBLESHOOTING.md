# Xử Lý Lỗi Thường Gặp — Troubleshooting Guide

## 1. API không start được (CrashLoopBackOff)

### Lỗi: `Connection refused [Errno 111]` với Qdrant

**Nguyên nhân**: API container không kết nối được Qdrant.

**Checklist**:

```powershell
# 1. Qdrant có chạy không?
docker ps | findstr qdrant

# 2. Network test_network có tồn tại không?
docker network ls | findstr test_network

# 3. Cả 2 container có cùng network không?
docker network inspect test_network | grep -E "qdrant|conversational"

# 4. Env QDRANT_URL trong container đúng không?
docker exec conversational-rag-api python -c "import os; [print(f'{k}={v}') for k,v in os.environ.items() if 'QDRANT' in k]"

# 5. Test kết nối từ container đến Qdrant
docker exec conversational-rag-api python -c "import urllib.request; print(urllib.request.urlopen('http://qdrant:6333/').read()[:100])"
```

**Fix**:

| Scenario | Fix |
|---|---|
| Qdrant chạy host, API chạy Docker | `.env`: `QDRANT_URL=http://host.docker.internal` |
| Qdrant chạy container khác, cùng network | `.env`: `QDRANT_URL=http://qdrant` (service name) |
| Qdrant chạy container khác, khác network | Gắn cùng network: `docker network connect test_network qdrant` |
| Chưa tạo network | `docker network create test_network` |

### Lỗi: `Failed to obtain server version` (Qdrant client warning)

Warning, không fatal. Qdrant client không check được version compatibility. Set
`check_compatibility=False` trong code hoặc ignore.

### Lỗi: `API key is used with an insecure connection`

Warning, không fatal. Gọi API Qdrant không có SSL. Dùng HTTP cho local dev.

## 2. Model Download / Loading

### Lỗi: `OSError: Can't load tokenizer for 'BAAI/bge-m3'`

**Nguyên nhân**: Không truy cập được HuggingFace Hub (network restriction / proxy).

**Fix**:

```powershell
# Dùng mirror cho China / nội bộ
$env:HF_ENDPOINT = "https://hf-mirror.com"
docker compose up --build -d

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

Lần đầu load model (2.2GB mỗi model) từ HuggingFace → phụ thuộc bandwidth.
Các lần sau dùng cache volume `bge_hf_cache`. Nếu vẫn chậm → kiểm tra tốc độ mạng.

## 3. Migration

### Lỗi: `FileNotFoundError: input/`

**Nguyên nhân**: Thư mục input (Mongo dump) không đúng.

**Fix**: Kiểm tra `INPUT_DIR` trong `.env` (default `../input`). Tạo dir:

```powershell
New-Item -ItemType Directory -Path ../input -Force
# Đặt file dump JSON vào ../input/
```

### Lỗi: Migration rất chậm / tiến độ 0

Chạy local embedding CPU-bound. Với dataset lớn (>10k docs) có thể mất hàng giờ.

```powershell
# Test với limit trước
uv run python -m agent.scripts.migrate_dump_to_qdrant --limit 10

# Full migration
uv run python -m agent.scripts.migrate_dump_to_qdrant

# Resume nếu bị gián đoạn (chạy lại, tự skip đã upsert)
```

### Lỗi: `Conflict` / `Already exists` (Qdrant upsert)

Checkpoint tự skip, nhưng nếu muốn force re-upsert toàn bộ:

```powershell
Remove-Item migration_checkpoint.jsonl
uv run python -m agent.scripts.migrate_dump_to_qdrant --recreate
```

## 4. Embedding / Upload

### Lỗi: `No files were uploaded`

**Nguyên nhân**: Body multipart không đúng format.

**Fix**: Dùng đúng key `files` trong form-data:

```powershell
curl -X POST "http://localhost:8001/embeddings/documents?collection_name=default&file_ending=.pdf" -F "files=@test.pdf"
```

### Lỗi: Upload file rất chậm

Embedding CPU-bound. File lớn (100+ trang PDF) có thể mất 1-2 phút. Thiết kế là synchronous
(in thread pool) — một request chiếm toàn bộ CPU cho embedding.

## 5. Search / Retrieval

### Lỗi: Search trả về empty documents

```powershell
# 1. Qdrant có data không?
curl http://localhost:6333/collections/documents/points/count

# 2. Collection name đúng không?
curl http://localhost:6333/collections

# 3. Test search trực tiếp Qdrant
curl -X POST http://localhost:6333/collections/documents/points/search -H "Content-Type: application/json" -d '{"vector": [0]*1024, "limit": 1}'
```

### Lỗi: `Unknown fusion_algorithm: '...'`

**Nguyên nhân**: `FUSION_ALGORITHM` trong `.env` không hợp lệ.

**Fix**: Chỉ chấp nhận `rrf` hoặc `dbsf`. Xem [CONFIGURATION.md](CONFIGURATION.md).

## 6. Kết nối mạng giữa các container

### Diagnostic Script (chạy từ host):

```powershell
Write-Host "=== Network Check ===" -ForegroundColor Cyan
docker network inspect test_network | python -c "import sys,json; d=json.load(sys.stdin); [print(f'  {c}: {v["Name"]}') for c,v in d[0]['Containers'].items()]"

Write-Host "`n=== Qdrant Direct ===" -ForegroundColor Cyan
docker exec conversational-rag-api python -c "import urllib.request; print(urllib.request.urlopen('http://qdrant:6333/').read()[:200].decode())" -ErrorAction SilentlyContinue

Write-Host "`n=== Env Check ===" -ForegroundColor Cyan
docker exec conversational-rag-api python -c "import os; print('QDRANT_URL:', os.environ.get('QDRANT_URL','MISSING'))"
```

## 7. Out of Memory / Performance

**Symptom**: API chậm dần theo thời gian, hoặc crash khi có nhiều request.

**Check**:

```powershell
docker stats conversational-rag-api --no-stream
```

**Fix**:
- Giới hạn memory container: `deploy.resources.limits.memory: 12G`
- Load test trước khi deploy production
- Cache warm: request 1 query ngẫu nhiên sau deploy

## 8. Common Known Bugs (v6.0.0)

| Bug | Workaround | Fixed in |
|---|---|---|
| `QDRANT_URL==http://qdrant` double `=` typo trong `.env` | Sửa thành `QDRANT_URL=http://qdrant` | v6.0.1 |
| Frontend stream không show content (vì API chỉ trả documents) | Frontend cần update parse `type=documents` | v6.1.0 |
