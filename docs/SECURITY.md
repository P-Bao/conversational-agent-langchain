# Bảo Mật — Security Guide

## 1. Secret Management

Hiện tại secrets được lưu trong `.env` file ở project root (gitignore'd).
Các biến mang tính nhạy cảm:

| Biến | Mức nhạy cảm | Ghi chú |
|---|---|---|
| `NVIDIA_API_KEY` | Cao | API key trả phí, không commit |
| `COHERE_API_KEY` | Cao | API key trả phí |
| `QWEN_EVAL_API_KEY` | Thấp | API key nội bộ, thường placeholder |
| `OPENAI_API_KEY` | Cao | Legacy API key |

Quy tắc:
- `.env` đã có trong `.gitignore` — không bao giờ commit.
- `template.env` là mẫu — không chứa giá trị thật.
- Các API key cloud (NVIDIA, Cohere) không nên hardcode trong .env file của
  production, nên dùng Docker secrets hoặc vault.

### Production khuyến nghị:

```yaml
# Docker Compose secrets
secrets:
  nvidia_api_key:
    file: ./secrets/nvidia_key.txt
  cohere_api_key:
    file: ./secrets/cohere_key.txt
```

## 2. Network Exposure

| Service | Port | Expose ra ngoài? | Khuyến nghị |
|---|---|---|---|
| API | 8001 | Optional | Chỉ expose sau reverse proxy (nginx/traefik) có TLS. |
| Qdrant REST | 6333 | Không | Chỉ access từ API container hoặc management network. |
| Qdrant gRPC | 6334 | Không | Internal. |
| Frontend | 8501 | Optional | Streamlit dev server, nên deploy sau reverse proxy. |

### Reverse proxy setup (ví dụ nginx):

```nginx
server {
    listen 443 ssl;
    server_name rag-api.org;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 3. Model Security

### Model Provenance

Các model được tải từ HuggingFace Hub:

| Model | HuggingFace ID | Checksum / hash? |
|---|---|---|
| Dense | `BAAI/bge-m3` | Không verify |
| Sparse | `BAAI/bge-m3` (shared) | Không verify |
| Reranker | `BAAI/bge-reranker-v2-m3` | Không verify |

**Khuyến nghị production**:
- Cache model và verify hash trước khi deploy.
- Nếu môi trường internal không có internet, pre-download model từ máy tin cậy,
  mount volume vào container.

### RFI / File Upload

Upload endpoint (`POST /embeddings/documents`) cho phép upload PDF/txt.
Các rủi ro:

- **File size**: Backend không giới hạn file size → có thể gây OOM.
  Khuyến nghị thêm `limit` trong nginx:
  ```
  client_max_body_size 50M;
  ```
- **File type**: Backend kiểm tra `file_ending` nhưng không validate MIME type.
  Khuyến nghị thêm MIME check trong production.

## 4. Dependency Security

Project quản lý dependencies qua `uv.lock` — deterministic, reproducible builds.
Các dependency đáng chú ý:

| Package | Rủi ro | Mitigation |
|---|---|---|
| `torch` | Lớn, nhiều CVE | Update theo advisory |
| `FlagEmbedding` | Ít maintainer | Pin version, monitor |
| `qdrant-client` | Network dependency | Pin minor version |

Scan định kỳ:
```powershell
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

## 7. Logging Security

- Logger (`loguru`) output ra stdout/stderr container logs.
- Logs có thể chứa query người dùng — xem xét PII policy.
- Không log API keys nếu có thể.
