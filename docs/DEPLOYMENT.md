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
    |     container: conversational-rag-api   port: 8001
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
      - "8001:8001"
    env_file:
      - .env
    environment:
      - QDRANT_URL=${QDRANT_URL:-http://qdrant}
      - QDRANT_PORT=${QDRANT_PORT:-6333}
    volumes:
      - hf_cache:/root/.cache/huggingface
    # Healthcheck dung /healthz (liveness) — process song la OK
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s   # cho model download lan dau
    networks:
      - test_network

networks:
  test_network:
    external: true
    name: test_network

volumes:
  hf_cache:
    name: bge_hf_cache
```

### `.env` (cần khớp):

```env
QDRANT_URL=http://qdrant    # DNS name trên network, KHONG phai localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents  # collection da ton tai tren Qdrant
RERANK_PROVIDER=none              # default — không load reranker
```

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
curl http://localhost:8001/healthz

# 5. Verify readyz — phải 200 nếu collection đã tồn tại
curl http://localhost:8001/readyz
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
| GPU | Nếu có GPU NVIDIA, mount vào container (`deploy.resources.reservations.devices`). FP16 inference nhanh 3-5x. |
| Memory | Model BGE-m3 ~4GB RAM (CPU). Nếu `RERANK_PROVIDER=bge` thêm ~2GB. Memory limit recommend >= 6GB. |
| CPU | Embedding server CPU-bound. Nếu load cao, scale bằng nhiều container + Qdrant cluster. |
| Volume | HF cache volume (`bge_hf_cache`) giữa các lần restart, tránh re-download model (2.2GB/lần). |
| Healthcheck | `/healthz` cho liveness (process sống); `/readyz` cho readiness (Qdrant OK + collection tồn tại). |
| Log rotation | Default driver json-file. Set `logging.driver=json-file` + `max-size=10m` nếu muốn. |
| Network security | Qdrant port 6333 không expose ra ngoài nếu không cần. API port 8001 sau reverse proxy. |
| Ingestion tách riêng | Repo này không ingestion. Đảm bảo quy trình ingestion (repo ngoài) chạy **trước** khi API nhận traffic — `/readyz` giúp phát hiện nhánh nào chưa sẵn sàng. |

## 8. Healthcheck Config

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s   # cho model download lan dau (~2.2GB)
```

Hoặc dùng readiness probe riêng cho Kubernetes:

```yaml
readinessProbe:
  httpGet:
    path: /readyz
    port: 8001
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

Storage volume `hf_cache` giữ model files giữa các build → không phải re-download.

## 10. Verify Deployment

```bash
# Check tất cả container
docker ps

# Check container trên cùng network
docker network inspect test_network | grep -E "qdrant|conversational"

# Liveness probe
curl http://localhost:8001/healthz

# Readiness probe
curl http://localhost:8001/readyz

# Deep test: search thử
curl -X POST http://localhost:8001/rag/ \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"test query"}],"collection_name":"documents"}'
```
