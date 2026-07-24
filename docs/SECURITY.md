# Bảo Mật — Security Guide (v7.0.0)

## 1. Secret Management

Hiện tại secrets được lưu trong `.env` file ở project root (gitignored).
Các biến mang tính nhạy cảm:

| Biến | Mức nhạy cảm | Ghi chú |
|---|---|---|
| `NVIDIA_API_KEY` | Cao | API key trả phí (DeepEval qwen test only) |
| `QWEN_EVAL_API_KEY` | Thấp | API key nội bộ, thường placeholder |
| `QDRANT_API_KEY` | Cao | API key Qdrant Cloud nếu dùng cloud |

### Production khuyến nghị:

```yaml
# Docker Compose secrets
secrets:
  nvidia_api_key:
    file: ./secrets/nvidia_key.txt
  qdrant_api_key:
    file: ./secrets/qdrant_key.txt
```

## 2. Network Exposure

| Service | Port | Expose ra ngoài? | Khuyến nghị |
|---|---|---|---|
| API | 8001 | Optional | Chỉ expose sau reverse proxy (nginx/traefik) có TLS |
| Qdrant REST | 6333 | Không | Chỉ access từ API container hoặc management network |
| Qdrant gRPC | 6334 | Không | Internal |

### Reverse proxy setup (ví dụ nginx):

```nginx
server {
    listen 443 ssl;
    server_name rag-api.example.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /readyz {
        proxy_pass http://localhost:8001;
        # Không cache health endpoint
        proxy_cache off;
    }
}
```

> Liveness (`/healthz`) và readiness (`/readyz`) **không nên cache** ở proxy.

## 3. Model Security

### Model Provenance

> Từ v7.1 repo này **không pull/tải model** vào Docker container. BGE-m3 +
> BGE-reranker-v2-m3 chạy trên **remote server** (Colab ngrok hoặc server GPU
> riêng) — xem notebook `rag_test_bge_m3_reranker_ngrok.ipynb`. Container API
> chỉ gọi HTTP tới `EMBEDDING_BASE_URL` / `RERANK_BASE_URL`.

| Model | Chạy ở đâu | HuggingFace ID | Checksum / hash? |
|---|---|---|---|
| Dense (BGE-m3) | Remote server (ngoài container) | `BAAI/bge-m3` | Do remote server verify |
| Reranker (optional) | Remote server (ngoài container) | `BAAI/bge-reranker-v2-m3` | Do remote server verify |

**Khuyến nghị production**:
- Verify hash/sốported của remote server (nó chịu trách nhiệm tải model).
- Bảo vệ ngrok URL / server endpoint bằng auth hoặc IP allowlist khi production
  (Colab ngrok public URL là dev-only, không nên dùng production).
- Pin phiên bản notebook server khi reproducibility quan trọng.

### File Upload (RFI)

> **Repo v7 không có endpoint upload file** (`/embeddings/documents` đã bỏ).
> Do đó rủi ro RFI / OOM từ user upload trực tiếp đã được loại bỏ.

Ingestion nằm ở repo ngoài — áp dụng security review cho repo ingestion.

## 4. Dependency Security

Project quản lý dependencies qua `uv.lock` — deterministic, reproducible builds.
Từ v7.1 Docker image **không còn torch/transformers/sentence-transformers/
FlagEmbedding/CUDA wheels** — embedding/rerank chạy trên HTTP server ngoài.

| Package | Rủi ro | Mitigation |
|---|---|---|
| `httpx` | Network dependency (gọi remote server) | Pin version, set `EMBEDDING_TIMEOUT`/`RERANK_TIMEOUT` |
| `qdrant-client` | Network dependency | Pin minor version |

Scan định kỳ:
```bash
uv run pip-audit
# Hoặc dùng docker scan / trivy
docker scan conversational-rag-api
```

## 5. Docker Security

- Container chạy với user `root` (mặc định của `uv:python3.13-bookworm-slim`).
  Khuyến nghị thêm non-root user:
  ```dockerfile
  RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /src
  USER appuser
  ```
- `restart: unless-stopped` — container tự restart khi crash.
- Network mode: bridge (mặc định), không dùng host network.

## 6. API Authentication

Hiện tại API không có authentication. Tất cả endpoints đều public.

**Nếu cần deploy production**:
- Thêm API key middleware (FastAPI Middleware)
- Hoặc reverse proxy với basic auth / OAuth2
- Rate limiting để prevent abuse

Example middleware:

```python
@app.middleware("http")
async def api_key_check(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if api_key != os.getenv("RAG_API_KEY"):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)
```

> **Health endpoints (`/healthz`, `/readyz`) thường được ALB/LB gọi mà không
> có auth**, vì vậy phải đảm bảo middleware không block các path này.

## 7. Logging Security

- Logger (`loguru`) output ra stdout/stderr container logs.
- Logs có thể chứa query người dùng — xem xét PII policy.
- Không log API keys nếu có thể.
- `/readyz` failure (`reason` + `details`) có thể leak internal info — chấp nhận
  được vì chỉ dành cho LB / health-check, không public.
