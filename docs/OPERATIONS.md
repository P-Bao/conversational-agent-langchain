# Runbook Vận Hành — Operations Guide v7.1.0

> **Khác v7.0.0:** Port API `8001` → **`8005`**, network `test_network` →
> `ami-network`. Reranker local FlagReranker cần GPU + cache model ~1.1GB.
> Tuỳ chọn query transformation thêm 1 LLM round latency.

## 1. Startup Sequence

### Order:

```
1. [Create network]      docker network create ami-network
2. [Start Qdrant]        docker compose up -d          (qdrant_docker/)
3. [Start embed server]  docker compose up -d          (embedding-server/)   # repo ngoài
4. [Ingestion]           Repo ingestion NGOÀI — tạo collection + upsert data
5. [Start Qwen server]   (optional, chỉ khi QUERY_TRANSFORM_ENABLED=true)
6. [Start API]           docker compose up --build -d  (conversational-agent-langchain/)
7. [Verify healthz]      curl http://localhost:8005/healthz
8. [Verify readyz]       curl http://localhost:8005/readyz   # phải 200 ready
```

API container **không cần Qdrant ready trước khi start** (client lazy). Nếu
`/readyz` fail, block traffic ở LB layer đến khi Qdrant + collection sẵn sàng.

> Lần đầu start: FlagReranker tải model `BAAI/bge-reranker-v2-m3` (~1.1GB)
> → request đầu tiên có thể chậm ~30-60s. Cache vào `hf-cache/` volume.

## 2. Common Operations

| Tác vụ | Lệnh |
|---|---|
| Start API | `docker compose up --build -d` |
| Stop API | `docker compose down` |
| Restart API | `docker compose restart` |
| Rebuild + start | `docker compose up --build -d` |
| Start Qdrant | `cd ../qdrant_docker && docker compose up -d` |
| Start Embedding server | `cd ../embedding-server && docker compose up -d` |
| Stop Qdrant | `cd ../qdrant_docker && docker compose down` |
| Stop Embedding server | `cd ../embedding-server && docker compose down` |
| Full stack stop | `docker compose down` (cả 3 thư mục) |
| View logs API | `docker logs -f conversational-rag-api` |
| View logs Qdrant | `docker logs -f qdrant` |
| View logs Embed server | `docker logs -f bge-m3-embed` |
| Xóa cache model (force reload) | `docker compose down -v && rm -rf hf-cache` |

## 3. Health & Readiness

### Liveness probe (`/healthz`)

```bash
curl http://localhost:8005/healthz
# 200 {"status":"ok"}    ← process còn sống
```

### Readiness probe (`/readyz`)

```bash
curl http://localhost:8005/readyz
# 200 {"status":"ready","collection":"documents"}    ← Qdrant OK + collection tồn tại
# 503 {"reason":"collection_missing","collection":"documents"}
# 503 {"reason":"qdrant_unreachable","details":"Connection refused"}
# 503 {"reason":"qdrant_error","details":"..."}
```

### API startup logs (success path):

```
Using remote BGE-m3 embedding endpoint: http://bge-m3-embed:8008 (sparse enabled)
Loading local BGE reranker: BAAI/bge-reranker-v2-m3 (fp16=True)
Startup: Retrieval & Search API v7.1.0
Loading REST API Finished.
```

Nếu `RERANK_PROVIDER=bge` thì log thêm khi rerank thật sự được gọi:

```
Local reranked N documents to top M
```

Nếu `QUERY_TRANSFORM_ENABLED=true`:

```
Query transformation done: rewritten='...', step_back='...', N sub-queries
```

Nếu query transform fail:

```
Query transformation failed, falling back to original query: <error>
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

# Filter theo component
docker logs conversational-rag-api 2>&1 | grep "remote BGE-m3"
docker logs conversational-rag-api 2>&1 | grep "Local rerank"
docker logs conversational-rag-api 2>&1 | grep "Query transformation"
docker logs conversational-rag-api 2>&1 | grep "Qdrant"
```

### GPU monitoring:

```bash
# GPU usage trong container
docker exec conversational-rag-api python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"

# Host GPU stats
nvidia-smi -l 2   # refresh mỗi 2s
```

### Embedding server logs:

```bash
docker logs -f bge-m3-embed
docker logs bge-m3-embed 2>&1 | grep "POST /embed"
```

## 5. Data Backup & Restore

### Qdrant data backup:

Qdrant lưu data trong volume `/qdrant/storage` (map đến `../vector_db`
trên host).

```bash
cp -r vector_db/ "vector_db.backup.$(date +%Y%m%d)/"
```

Restore: stop Qdrant, replace thư mục `vector_db/`, start lại.

### Model cache backup (reranker):

Volume `hf-cache/` chứa BGE-reranker-v2-m3 đã tải. Backup để tránh tải lại
~1.1GB khi rebuild:

```bash
cp -r hf-cache/ "hf-cache.backup.$(date +%Y%m%d)/"
```

> **Không có migration script trong repo này** — collection + upsert thuộc
> repo ingestion ngoài.

## 6. Monitoring

### Health Endpoints:

```bash
curl http://localhost:8005/healthz   # process up?
curl http://localhost:8005/readyz    # Qdrant + collection ready?
```

### Key Metrics:

| Metric | How to check | Threshold |
|---|---|---|
| API response time | `curl -w "%{time_total}" http://localhost:8005/healthz` | < 200ms (liveness), < 5s (with rerank + query transform) |
| Embedding time | Logs: `Using remote BGE-m3` → first /rag response | < 60s (network tới embedding-server) |
| Rerank time | Logs: `Local reranked N docs to top M` | < 2s (local GPU) |
| Query transform time | Logs: `Query transformation done` | < 2s (3 LLM calls song song tới Qwen) |
| GPU memory | `nvidia-smi` | < 2GB VRAM (reranker model) |
| Memory (container) | `docker stats conversational-rag-api` | ~3-5GB RAM + ~1.4GB VRAM |
| Disk | `docker system df` | `hf-cache/` ~1.1GB |

### Alert Triggers:

- API container restarting (CrashLoopBackOff): check Qdrant + embedding-server
- `Connection refused` in logs: Qdrant down hoặc `QDRANT_URL` sai
- `/readyz` liên tục 503: collection chưa tồn tại hoặc Qdrant mất kết nối
- Embedding timeout: embedding-server down hoặc `EMBEDDING_BASE_URL` sai
- `Query transformation failed` liên tục: Qwen server down — fallback vẫn
  hoạt động nhưng không được lợi recall
- `CUDA out of memory`: GPU đầy, chuyển `RERANK_PROVIDER=none` nếu tạm thời
- `Unknown reranker provider`: `RERANK_PROVIDER` sai giá trị (chỉ `bge`/`remote`/`none`)

## 7. Capacity Planning

| Tài nguyên | Dự kiến dùng | Ghi chú |
|---|---|---|
| RAM per API container | ~3-5 GB | Reranker model load trong VRAM; Python overhead |
| VRAM | ~1.4 GB | BGE-reranker-v2-m3 fp16 + FlagReranker context |
| RAM per Qdrant | 1-4 GB | Tuỳ số vector |
| Disk per Qdrant | 100MB - 10GB | Tuỳ dataset |
| Disk `hf-cache/` | ~1.1 GB | Cache reranker model (1 lần duy nhất) |
| CPU per request | thấp | Embedding/LLM delegate ra ngoài; rerank dùng GPU |
| Embedding server (ngoài) | ~6-10 GB | Chạy BGE-m3 ~2.7GB |
| Qwen server (optional) | Tuỳ model | Self-host, VRAM tuỳ model size |

## 8. Updating Models

### Đổi reranker model (`RERANK_MODEL`):

1. Cập nhật `.env`: `RERANK_MODEL=<new-hf-id>`.
2. Xoá cache model cũ:
   ```bash
   docker compose down
   rm -rf hf-cache/*
   ```
3. Restart: `docker compose up --build -d`.
4. Request đầu tiên sẽ tải model mới (~1.1GB) — chờ ~30-60s.

### Đổi embedding model:

Embedding model sống trên **embedding-server** (repo ngoài), không trong API
container. Cập nhật repo embedding-server, restart container đó. Repo API
không cần rebuild — chỉ gọi HTTP.

> **Hệ thống ingestion ngoài** cần re-create collection Qdrant với config
> dense/sparse mới (nếu dim đổi) và re-upsert toàn bộ data.

### Đổi Qwen model (cho query transform):

1. Self-host Qwen server tải model mới.
2. Cập nhật `.env`: `QWEN_MODEL=<new-model-name>`.
3. Restart API: `docker compose restart`.
4. Không cần rebuild — `QWEN_MODEL` chỉ là tên model trong request.

## 9. Cleanup

```bash
# Docker housekeeping
docker system prune -f          # xóa stopped containers, unused networks
docker builder prune -f         # xóa build cache

# Khi build fail / muốn khởi động lại sạch hoàn toàn
make docker-clean
# = docker compose down --remove-orphans -v && docker system prune -a --volumes -f

# Xóa cache reranker (force model reload lần sau)
rm -rf hf-cache/*

# Migration script / checkpoint files → không có trong repo này
```
