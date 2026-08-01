# Triển Khai — Deployment Guide (Docker) v7.1.0

> **Khác v7.0.0:**
> - Port `8001` → **`8005`**, network `test_network` → **`ami-network`** (external).
> - Base image đổi từ `uv:python3.13-bookworm-slim` (CUDA-free) →
>   **`pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`** (reranker local cần GPU).
> - Quản lý deps chuyển từ `uv sync` sang `pip install -r requirements.txt` trong
>   Dockerfile.
> -docker-compose.yml thêm block `deploy.resources` (GPU reservation) +
>   volume `./hf-cache:/app/.cache/huggingface` (cache model reranker).

## 1. Kiến Trúc Triển Khai

Hệ thống gồm **3 stack Docker** riêng biệt (cùng network `ami-network`):

```
[Host Machine]
    |
    +-- Qdrant Stack (qdrant_docker/)
    |     container: qdrant              port: 6333
    |     (collection do hệ ingestion ngoài quản lý)
    |
    +-- Embedding Server Stack (embedding-server/)
    |     container: bge-m3-embed        port: 8008
    |     (remote BGE-m3 — repo ngoài)
    |
    +-- API Stack (conversational-agent-langchain/)
    |     container: conversational-rag-api   port: 8005   [cần GPU]
    |
    +-- (Optional) Qwen server          port: 8000    (query transformation)
    |
    +-- (Optional) Frontend Streamlit — repo riêng
```

Giao tiếp giữa các stack qua Docker shared network `ami-network` (external).

## 2. Setup Network Chung

```bash
docker network create ami-network
```

> Nếu Qdrant / embedding-server đã có network riêng, gắn thêm vào
> `ami-network`: `docker network connect ami-network <container>`.

## 3. Khởi Động Qdrant

```bash
cd ../qdrant_docker
docker compose up -d
```

Kiểm tra:

```bash
curl http://localhost:6333/
# { "title": "qdrant - vector search engine", "version": "1.18.x" }
```

Dashboard: `http://localhost:6333/dashboard`.

## 4. Khởi Động Embedding Server

Embedding server là **repo ngoài**, chạy BGE-m3qua HTTP. Nó expose
`POST /embed` trả dense + sparse vectors.

```bash
cd ../embedding-server
docker compose up -d
```

Verify (từ host):

```bash
curl http://localhost:8008/
# hoặc trong Docker network:
docker exec conversational-rag-api curl http://bge-m3-embed:8008/
```

> Repo API này **không** tải BGE-m3 model vào container — chỉ gọi HTTP.
> Embedding server chịu trách nhiệm tải + serve model (~2.7GB).

## 5. Khởi Động API

```bash
cd conversational-agent-langchain
docker compose up --build -d
```

> **Lần đầu build** tải PyTorch base image (~6GB) + cài `FlagEmbedding`,
> `transformers` qua `pip`. Có thể mất ~10-15 phút tuỳ băng thông.
> Các lần sau dùng cache layer — build nhanh hơn nhiều.

## 6. Cấu Hình Docker Compose (API)

File `docker-compose.yml`:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: conversational-rag-api
    restart: unless-stopped
    ports:
      - "8005:8005"
    env_file:
      - .env
    volumes:
      - ./hf-cache:/app/.cache/huggingface    # cache model reranker
    networks:
      - ami-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test:
        - CMD-SHELL
        - "uv run python -c \"import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8005/healthz', timeout=5).status == 200 else 1)\""
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

networks:
  ami-network:
    external: true
```

### Dockerfile v7.1.0

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
RUN mkdir -p /app/.cache/huggingface

EXPOSE 8005
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8005/healthz || exit 1

CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8005"]
```

### `.env` (cần khớp):

```env
QDRANT_URL=http://qdrant              # DNS name trong ami-network
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents      # collection đã tồn tại

EMBEDDING_PROVIDER=remote
EMBEDDING_BASE_URL=http://bge-m3-embed:8008    # service name trong ami-network
EMBEDDING_API_KEY=                                # trống nếu server không bật auth

RERANK_PROVIDER=bge                              # local FlagReranker (default)
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TOP_K=5

# (Optional) Query transformation
QUERY_TRANSFORM_ENABLED=false
QWEN_BASE_URL=http://host.docker.internal:8000/v1   # hoặc service name nếu Qwen cùng network
QWEN_API_KEY=dummy
QWEN_MODEL=qwen
```

## 7. Multi-Stack Workflow

### Lần đầu:

```bash
# 1. Tạo network
docker network create ami-network

# 2. Start Qdrant
cd ../qdrant_docker && docker compose up -d

# 3. Start embedding-server (repo ngoài)
cd ../embedding-server && docker compose up -d

# 4. Start API (lần đầu build chậm do tải PyTorch base)
cd ../conversational-agent-langchain && docker compose up --build -d

# 5. Verify healthz
curl http://localhost:8005/healthz

# 6. Verify readyz
curl http://localhost:8005/readyz
```

### Routine restart:

```bash
# Restart API (không rebuild)
cd conversational-agent-langchain && docker compose restart

# Restart API + rebuild
docker compose down && docker compose up --build -d

# Restart Qdrant
cd ../qdrant_docker && docker compose restart
```

### Shutdown:

```bash
cd conversational-agent-langchain && docker compose down
cd ../qdrant_docker && docker compose down
cd ../embedding-server && docker compose down
```

## 8. Production Considerations

| Area | Khuyến nghị |
|---|---|
| GPU | **Bắt buộc** cho `RERANK_PROVIDER=bge` (local FlagReranker). Nếu không có GPU → `RERANK_PROVIDER=none` và bỏ block `deploy.resources` |
| Memory | API container ~3-5GB RAM + ~1.4GB VRAM (reranker model). Embedding server giữ BGE-m3 ~2.7GB riêng |
| Model cache | Volume `./hf-cache:/app/.cache/huggingface` cache `BAAI/bge-reranker-v2-m3` (~1.1GB). Backup thư mục này giữa các rebuild |
| Healthcheck | `/healthz` (liveness); `/readyz` (Qdrant + collection). Start period 60s cho model tải lần đầu |
| Log rotation | Default driver `json-file`. Set `logging.driver=json-file` + `max-size=10m` nếu cần |
| Network security | Qdrant port 6333 không expose ra ngoài. Embedding server port 8008 chỉ trong `ami-network`. API port 8005 sau reverse proxy |
| Ingestion tách riêng | Repo này không ingestion. Đảm bảo quy trình ingestion ngoài chạy **trước** khi API nhận traffic |
| GPU multi-tenant | Nếu host chia sẻ, giới hạn `count: 1` thay vì `all` trong `deploy.resources` |

## 9. Rebuild on Code Change

```bash
# Fast rebuild (cache layers cũ)
docker compose build --no-cache-pull
docker compose up -d

# Full rebuild từ đầu (xóa cache)
docker compose build --no-cache
docker compose up -d
```

> Nếu đổi reranker model (`RERANK_MODEL`), xoá volume `hf-cache` để load
> model mới:
> ```bash
> docker compose down -v && rm -rf hf-cache && docker compose up --build -d
> ```

Nếu build fail (disk đầy / cache hỏng):

```bash
make docker-clean   # docker compose down -v && docker system prune -a --volumes -f
```

## 10. Verify Deployment

```bash
# Check tất cả container
docker ps

# Check container trên cùng network
docker network inspect ami-network | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print([c['Name'] for c in d[0]['Containers'].values()])"

# Liveness probe
curl http://localhost:8005/healthz

# Readiness probe
curl http://localhost:8005/readyz

# Deep test: retrieval
curl -X POST http://localhost:8005/rag/ \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"test query"}]}'

# Embedding server reachable từ container API?
docker exec conversational-rag-api curl -s http://bge-m3-embed:8008/

# GPU available trong container?
docker exec conversational-rag-api python -c "import torch; print(torch.cuda.is_available())"
```

## 11. Kubernetes Readiness Probe (optional)

```yaml
readinessProbe:
  httpGet:
    path: /readyz
    port: 8005
  initialDelaySeconds: 60
  periodSeconds: 10

livenessProbe:
  httpGet:
    path: /healthz
    port: 8005
  initialDelaySeconds: 30
  periodSeconds: 30
```
