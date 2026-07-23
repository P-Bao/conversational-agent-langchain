# Runbook Vận Hành — Operations Guide

## 1. Startup Sequence

### Order:

```
1. [Create network]   docker network create test_network
2. [Start Qdrant]     docker compose up -d           (qdrant_docker/)
3. [Start API]        docker compose up --build -d   (conversational-agent-langchain/)
4. [Verify]           curl http://localhost:8001/
```

API container cần Qdrant ready trước — nhưng không có `depends_on` cross-compose.
Nếu API start trước Qdrant, nó sẽ crash vì không connect được Qdrant ở startup
(`initialize_all_vector_dbs` tại thời điểm import). Đây là behavior cố ý (fail fast).

## 2. Common Operations

| Tác vụ | Lệnh |
|---|---|
| Start API | `docker compose up --build -d` |
| Stop API | `docker compose down` |
| Restart API | `docker compose restart` |
| Rebuild + start | `docker compose up --build -d` |
| Start Qdrant | `docker compose up -d` (ở qdrant_docker/) |
| Stop Qdrant | `docker compose down` (ở qdrant_docker/) |
| Full stack stop | `docker compose down` (cả 2 thư mục) |
| View logs API | `docker logs -f conversational-rag-api` |
| View logs Qdrant | `docker logs -f qdrant` |

## 3. Logs & Troubleshooting

### API container logs:

```powershell
# Follow logs
docker logs -f conversational-rag-api

# Last 100 lines
docker logs --tail 100 conversational-rag-api

# Filter by log level
docker logs conversational-rag-api 2>&1 | findstr "ERROR|CRITICAL"

# Filter specifc component
docker logs conversational-rag-api 2>&1 | findstr "BGE-m3"
docker logs conversational-rag-api 2>&1 | findstr "Qdrant"
```

### Qdrant logs:

```powershell
docker logs -f qdrant
curl http://localhost:6333/healthz  # Qdrant health check
```

### API startup logs (success path):

```
Bytecode compiled X files in Y.Zs
Loading BGE-m3 model: BAAI/bge-m3
SUCCESS: Collection documents already exists.
Loading REST API Finished.
```

## 4. Data Backup & Restore

### Qdrant data backup:

Qdrant lưu data trong volume `/qdrant/storage` (map đến `../vector_db` trên host).

```powershell
# Backup Qdrant data
Copy-Item -Recurse vector_db/ vector_db.backup.$(Get-Date -Format yyyyMMdd)/
```

Restore: stop Qdrant, replace thư mục `vector_db/` với bản backup, start lại.

### Checkpoint files:

- `migration_checkpoint.jsonl`: resume migration nếu script bị gián đoạn.
- `chunk_checkpoint.jsonl`: resume chunking.

Backup cả 2 file này trước khi re-migration:

```powershell
Copy-Item migration_checkpoint.jsonl "migration_checkpoint.jsonl.$(Get-Date -Format yyyyMMdd)"
```

## 5. Data Migration (Production)

Khi chạy migration production:

```powershell
# 1. Verify Qdrant đang chạy
curl http://localhost:6333/

# 2. Dry-run: kiểm tra có input không, chunk + embed 10 docs
uv run python -m agent.scripts.migrate_dump_to_qdrant --limit 10

# 3. Full migration (--recreate nếu cần collection mới)
uv run python -m agent.scripts.migrate_dump_to_qdrant --recreate

# 4. Chạy locator để cập nhật golden dataset
uv run python tests/locate_expected_chunks.py

# 5. Chạy DeepEval để verify chất lượng
$env:ALLOW_NETWORK_TESTS = "1"
uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
```

Tham khảo [DATA_INGESTION.md](DATA_INGESTION.md) cho chi tiết CLI flags.

## 6. Monitoring

### Health Check Endpoint:

```powershell
curl http://localhost:8001/
# Response 200: "Welcome to the RAG Backend..."

curl http://localhost:6333/
# Qdrant version JSON
```

### Key Metrics:

| Metric | How to check | Threshold |
|---|---|---|
| API response time | `curl -w "%{time_total}"` | < 2s (with cache warm) |
| Embedding time (first) | Container logs: "Loading BGE-m3 model" → Done | < 120s |
| Rerank time | Logs: "BGE reranked N docs to top M" | < 2s |
| Memory usage | `docker stats conversational-rag-api` | < 10GB |
| Disk | `docker system df` | Varies |

### Alert Triggers:

- API container restarting (CrashLoopBackOff): usually Qdrant not reachable
- `Connection refused [Errno 111]` in logs: Qdrant down or wrong URL
- `FileNotFoundError: input/` in migration: INPUT_DIR sai
- `CUDA out of memory` (GPU): reduce batch size

## 7. Capacity Planning

| Tài nguyên | Dự kiến dùng | Ghi chú |
|---|---|---|
| RAM per API container | 6-10 GB | 2 model (embed+rerank) loaded |
| RAM per Qdrant | 1-4 GB | Phụ thuộc số lượng vector |
| Disk per Qdrant | 100MB - 10GB | Tuỳ dataset |
| CPU per request | 1-2 core seconds | Embedding CPU-bound |
| HF model cache | ~5 GB | `~/.cache/huggingface` volume |

## 8. Updating Models

Cập nhật model embedding/reranker:

1. Update `.env`:
   ```env
   AU_EMBED_MODEL_NAME=<new-model>
   AU_RERANK_MODEL_NAME=<new-reranker>
   ```
2. Rebuild container (cache thường dùng model mới download):
   ```powershell
   docker compose build --no-cache && docker compose up -d
   ```
3. Nếu `embedding_size` thay đổi, cần recreate Qdrant collection (xóa + migrate lại).

## 9. Cleanup

```powershell
# Docker housekeeping
docker system prune -f          # xóa stopped containers, unused networks
docker builder prune -f         # xóa build cache

# Xóa checkpoint (nếu muốn re-run migration từ đầu)
Remove-Item migration_checkpoint.jsonl, chunk_checkpoint.jsonl

# Xóa HF model cache (sẽ re-download)
Remove-Item -Recurse ~/.cache/huggingface/models--BAAI--*
```
