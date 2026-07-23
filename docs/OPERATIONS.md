# Runbook Vận Hành — Operations Guide (v7.0.0)

## 1. Startup Sequence

### Order:

```
1. [Create network]   docker network create test_network
2. [Start Qdrant]     docker compose up -d           (qdrant_docker/)
3. [Ingestion]        Repo ingestion NGOÀI — tao collection + upsert data
4. [Start API]        docker compose up --build -d   (conversational-agent-langchain/)
5. [Verify healthz]   curl http://localhost:8001/healthz
6. [Verify readyz]    curl http://localhost:8001/readyz  # phai 200 ready
```

API container **không cần Qdrant ready trước khi start**. Nếu Qdrant down,
API vẫn start được vì client lazy (chỉ kết nối khi route được gọi). Nếu
`/readyz` fail, block traffic ở LB layer cho đến khi Qdrant + collection sẵn sàng.

## 2. Common Operations

| Tác vụ | Lệnh |
|---|---|
| Start API | `docker compose up --build -d` |
| Stop API | `docker compose down` |
| Restart API | `docker compose restart` |
| Rebuild + start | `docker compose up --build -d` |
| Start Qdrant | `cd ../qdrant_docker && docker compose up -d` |
| Stop Qdrant | `cd ../qdrant_docker && docker compose down` |
| Full stack stop | `docker compose down` (cả 2 thư mục) |
| View logs API | `docker logs -f conversational-rag-api` |
| View logs Qdrant | `docker logs -f qdrant` |

## 3. Health & Readiness

### Liveness probe (`/healthz`)

```bash
curl http://localhost:8001/healthz
# 200 {"status":"ok"}    ← process con song
```

### Readiness probe (`/readyz`)

```bash
curl http://localhost:8001/readyz
# 200 {"status":"ready","collection":"documents"}    ← Qdrant OK + collection ton tai
# 503 {"status":"fail","reason":"collection_missing","collection":"documents"}
# 503 {"status":"fail","reason":"qdrant_unreachable","details":"Connection refused"}
# 503 {"status":"fail","reason":"qdrant_error","details":"..."}
```

### API startup logs (success path):

```
Loading BGE-m3 model: BAAI/bge-m3
Startup: Retrieval & Search API v7.0.0
Loading REST API Finished.
```

Nếu `RERANK_PROVIDER=bge` thì thêm dòng:
```
Loading BGE-reranker-v2-m3 model
```

## 4. Logs & Troubleshooting

### API container logs:

```bash
# Follow logs
docker logs -f conversational-rag-api

# Last 100 lines
docker logs --tail 100 conversational-rag-api

# Filter by log level
docker logs conversational-rag-api 2>&1 | grep -E "ERROR|CRITICAL"

# Filter specifc component
docker logs conversational-rag-api 2>&1 | grep "BGE-m3"
docker logs conversational-rag-api 2>&1 | grep "Qdrant"
```

### Qdrant logs:

```bash
docker logs -f qdrant
curl http://localhost:6333/healthz  # Qdrant health check
```

## 5. Data Backup & Restore

### Qdrant data backup:

Qdrant lưu data trong volume `/qdrant/storage` (map đến `../vector_db` trên host).

```bash
# Backup Qdrant data
cp -r vector_db/ "vector_db.backup.$(date +%Y%m%d)/"
```

Restore: stop Qdrant, replace thư mục `vector_db/` với bản backup, start lại.

> **Không có migration script trong repo này** — việc tạo collection + upsert
> data thuộc repo ingestion ngoài.

## 6. Monitoring

### Health Endpoints:

```bash
# Process up?
curl http://localhost:8001/healthz

# Qdrant + collection ready?
curl http://localhost:8001/readyz
```

### Key Metrics:

| Metric | How to check | Threshold |
|---|---|---|
| API response time | `curl -w "%{time_total}"` | < 2s (with cache warm) |
| Embedding time (first) | Container logs: "Loading BGE-m3 model" → Done | < 180s |
| Rerank time | Logs: "BGE reranked N docs to top M" (chỉ khi RERANK_PROVIDER=bge) | < 2s |
| Memory usage | `docker stats conversational-rag-api` | < 8GB nếu RERANK_PROVIDER=none |
| Memory usage (reranker on) | `docker stats conversational-rag-api` | < 12GB |
| Disk | `docker system df` | Varies |

### Alert Triggers:

- API container restarting (CrashLoopBackOff): check Qdrant connectivity
- `Connection refused [Errno 111]` in logs: Qdrant down hoặc `QDRANT_URL` sai
- `/readyz` liên tục 503: collection chưa tồn tại hoặc Qdrant mất kết nối
- `CUDA out of memory` (GPU): giảm concurrent requests, bật `RERANK_PROVIDER=none`

## 7. Capacity Planning

| Tài nguyên | Dự kiến dùng | Ghi chú |
|---|---|---|
| RAM per API container (no rerank) | 4-6 GB | Chỉ BGE-m3 embed loaded |
| RAM per API container (rerank bge) | 6-10 GB | BGE-m3 + BGE-reranker-v2-m3 loaded |
| RAM per Qdrant | 1-4 GB | Tuỳ số vector |
| Disk per Qdrant | 100MB - 10GB | Tuỳ dataset |
| CPU per request | 1-2 core seconds | Embedding CPU-bound |
| HF model cache | ~5 GB | `~/.cache/huggingface` volume |

## 8. Updating Models

Cập nhật model embedding:

1. Update `.env`:
   ```env
   EMBEDDING_MODEL=<new-model>
   EMBEDDING_SIZE=<new-dim>
   ```
2. Rebuild container:
   ```bash
   docker compose build --no-cache && docker compose up -d
   ```
3. **Repo ingestion ngoài** cần re-create collection với `EMBEDDING_SIZE` mới
   và re-upsert toàn bộ data.

## 9. Cleanup

```bash
# Docker housekeeping
docker system prune -f          # xóa stopped containers, unused networks
docker builder prune -f         # xóa build cache

# Xóa HF model cache (sẽ re-download)
rm -rf ~/.cache/huggingface/models--BAAI--*

# Không có migration script / checkpoint files để xóa trong repo này
```
