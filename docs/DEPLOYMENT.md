# Triển Khai — Deployment Guide (Docker)

## 1. Kiến Trúc Triển Khai

Hệ thống gồm 2 stack Docker riêng biệt:

```
[Host Machine]
    |
    +-- Qdrant Stack (qdrant_docker/)
    |     container: qdrant      port: 6333
    |     (collection do he ngoai quan ly theo quy trinh rieng)
    |
    +-- API Stack (conversational-agent-langchain/)
    |     container: conversational-rag-api   port: 8005
    |
    +-- Embed Server (GPU)                    port: 8008 (embed) / 8010 (rerank)
    |     BGE-m3 embedding + BGE reranker
    |
    +-- (Optional) Frontend Streamlit — repo riêng
```

Giao tiếp giữa 2 stack qua Docker shared network `test_network`.

## 2. Setup Network Chung

Tạo network (1 lần duy nhất):

```bash
docker network create test_network
```

## 3. Khởi Động Qdrant

```bash
cd qdrant_docker
docker compose up -d
```

Kiểm tra:

```bash
curl http://localhost:6333/
# Response: { "title": "qdrant - vector search engine", "version": "1.18.x" }
```

Dashboard Qdrant: `http://localhost:6333/dashboard`.

## 4. Khởi Động API

```bash
cd conversational-agent-langchain
docker compose up --build -d
```

## 5. Cấu Hình Docker Compose (API)

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
    environment:
      - QDRANT_URL=${QDRANT_URL:-http://qdrant}
      - QDRANT_PORT=${QDRANT_PORT:-6333}
    # Healthcheck dung /healthz (liveness) — process song la OK
    healthcheck:
      test: ["CMD-SHELL", "uv run python -c \"import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8005/healthz',timeout=5).status==200 else 1)\""]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s   # thao window khoi dong (khong phai model download)
    networks:
      - test_network

networks:
  test_network:
    external: true
```

> Không còn `hf_cache` volume (model chạy remote, Docker image không tải model)
> và không còn `extra_hosts` block. Network `test_network` vẫn external.

### `.env` (cần khớp):

```env
QDRANT_URL=http://qdrant    # DNS name trên network, KHONG phai localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents  # collection da ton tai tren Qdrant
EMBEDDING_PROVIDER=remote
EMBEDDING_BASE_URL=http://host.docker.internal:8008   # remote BGE-m3 embed server
RERANK_PROVIDER=remote                                # default — remote
RERANK_BASE_URL=http://host.docker.internal:8010      # remote BGE reranker
RERANK_MIN_SCORE=0.0                                  # lọc sau rerank
```

> Khi API chạy trong Docker mà embed/rerank server chạy trên host (bind
> `127.0.0.1`), dùng `host.docker.internal` thay vì `localhost` — `localhost`
> trong container là chính container.

## 6. Multi-Stack Workflow

### Lần đầu:

```bash
# 1. Tạo network
docker network create test_network

# 2. Start Qdrant
cd qdrant_docker
docker compose up -d

# 3. Start API
cd ../conversational-agent-langchain
docker compose up --build -d

# 4. Verify healthz
curl http://localhost:8005/healthz

# 5. Verify readyz — phải 200 nếu collection đã tồn tại
curl http://localhost:8005/readyz
```

### Routine restart:

```bash
# Restart API (không rebuild)
cd conversational-agent-langchain
docker compose restart

# Restart API + rebuild
docker compose down && docker compose up --build -d

# Restart Qdrant
cd ../qdrant_docker && docker compose restart
```

### Shutdown:

```bash
cd conversational-agent-langchain
docker compose down

cd ../qdrant_docker
docker compose down
```

## 7. Production Considerations

| Area | Khuyến nghị |
|---|---|
| GPU | API container không cần GPU (embedding/rerank remote). GPU server riêng giữ model. |
| Memory | API container ~512MB-1GB RAM (không load model; `bge` local sẽ tải thêm model). Remote server giữ BGE-m3 ~2.7GB + reranker ~2.2GB. |
| CPU | API container I/O bound (HTTP tới remote + Qdrant). Nếu load cao, scale bằng nhiều container + Qdrant cluster. |
| Volume | Không còn HF cache volume. Model files nằm trên remote server, không trong API container. |
| Remote server uptime | Embed/rerank server phải up khi API nhận traffic (fail-fast nếu `RERANK_PROVIDER=remote` mà không reachable). |
| Healthcheck | `/healthz` cho liveness (process sống); `/readyz` cho readiness (Qdrant OK + collection tồn tại). |
| Log rotation | Default driver json-file. Set `logging.driver=json-file` + `max-size=10m` nếu muốn. |
| Network security | Qdrant port 6333 không expose ra ngoài nếu không cần. API port 8005 sau reverse proxy. |
| Ingestion tách riêng | Repo này không ingestion. Đảm bảo quy trình ingestion (repo ngoài) chạy **trước** khi API nhận traffic — `/readyz` giúp phát hiện nhánh nào chưa sẵn sàng. |

## 8. Healthcheck Config

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8005/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s   # thao window khoi dong (khong phai model download)
```

Hoặc dùng readiness probe riêng cho Kubernetes:

```yaml
readinessProbe:
  httpGet:
    path: /readyz
    port: 8005
  initialDelaySeconds: 60
  periodSeconds: 10
```

## 9. Rebuild on Code Change

```bash
# Fast rebuild (cache layers cũ)
docker compose build --no-cache-pull
docker compose up -d

# Full rebuild từ đầu
docker compose build --no-cache
docker compose up -d
```

Không còn HF cache volume cần xoá — model chạy remote, image không giữ state.

Nếu build fail (layer cache hỏng / disk đầy), dọn sạch Docker:

```bash
make docker-clean   # docker compose down -v && docker system prune -a --volumes -f
```

## 10. Verify Deployment

```bash
# Check tất cả container
docker ps

# Check container trên cùng network
docker network inspect test_network | grep -E "qdrant|conversational"

# Liveness probe
curl http://localhost:8005/healthz

# Readiness probe
curl http://localhost:8005/readyz

# Deep test: search thử
curl -X POST http://localhost:8005/rag/ \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"test query"}],"collection_name":"documents"}'
```
