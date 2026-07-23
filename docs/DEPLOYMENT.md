# Triển Khai — Deployment Guide (Docker)

## 1. Kiến Trúc Triển Khai

Hệ thống gồm 2 stack Docker riêng biệt:

```
[Host Machine]
    |
    +-- Qdrant Stack (qdrant_docker/)
    |     container: qdrant      port: 6333
    |
    +-- API Stack (conversational-agent-langchain/)
    |     container: conversational-rag-api   port: 8001
    |
    +-- (Optional) Frontend Streamlit
          port: 8501
```

Giao tiếp giữa 2 stack qua Docker shared network `test_network`.

## 2. Setup Network Chung

Tạo network (1 lần duy nhất):

```powershell
docker network create test_network
```

## 3. Khởi Động Qdrant

```powershell
cd qdrant_docker
docker compose up -d
```

Kiểm tra:

```powershell
curl http://localhost:6333/
# Response: { "title": "qdrant - vector search engine", "version": "1.18.x" }
```

Dashboard Qdrant: `http://localhost:6333/dashboard`

Cấu hình Qdrant container:
- `container_name: qdrant` → DNS name trên network là `qdrant`
- Lưu data ở `../vector_db:/qdrant/storage`
- Port 6333 (REST) + 6334 (gRPC) exposed

Xem file: `qdrant_docker/docker-compose.yml`

## 4. Khởi Động API

```powershell
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
    networks:
      - test_network     # phải khớp với network Qdrant

networks:
  test_network:
    external: true
    name: test_network   # đặt tên cố định tránh Compose v2 prefix

volumes:
  hf_cache:
    name: bge_hf_cache
```

### `.env` (cần khớp):

```env
QDRANT_URL=http://qdrant    # DNS name trên network, không phải localhost
QDRANT_PORT=6333
```

## 6. Multi-Stack Workflow

### Lần đầu:

```powershell
# 1. Tạo network
docker network create test_network

# 2. Start Qdrant
cd qdrant_docker
docker compose up -d

# 3. Start API
cd ../conversational-agent-langchain
docker compose up --build -d

# 4. Verify
curl http://localhost:8001/
curl http://localhost:6333/
```

### Routine restart:

```powershell
# Restart API (không rebuild)
cd conversational-agent-langchain
docker compose restart

# Restart API + rebuild
docker compose down && docker compose up --build -d

# Restart Qdrant
cd qdrant_docker
docker compose restart
```

### Shutdown:

```powershell
cd conversational-agent-langchain
docker compose down

cd qdrant_docker
docker compose down
```

## 7. Production Considerations

| Area | Khuyến nghị |
|---|---|
| GPU | Nếu có GPU NVIDIA, mount vào container (`deploy.resources.reservations.devices`). FP16 inference nhanh 3-5x. |
| Memory | Model BGE-m3 + rerancer ~8GB RAM trên CPU. Container cần memory limit >= 8GB. |
| CPU | Embedding server CPU-bound. Nếu load cao, scale bằng nhiều container + Qdrant cluster. |
| Volume | HF cache volume (`bge_hf_cache`) giữa các lần restart, tránh re-download model (2.2GB mỗi lần). |
| Healthcheck | Endpoint `GET /` trả về 200. Thêm `healthcheck` trong compose. |
| Log rotation | Log container default. Thêm `logging.driver=json-file` + `max-size=10m`. |
| Network security | Qdrant port 6333 không expose ra ngoài nếu không cần. API port 8001 sau reverse proxy. |

### Healthcheck config example:

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s   # cho model download
```

## 8. Rebuild on Code Change

```powershell
# Fast rebuild (cache layers cũ)
docker compose build --no-cache-pull
docker compose up -d

# Full rebuild từ đầu
docker compose build --no-cache
docker compose up -d
```

Storage volume `hf_cache` giữ model files giữa các build → không phải re-download.

## 9. Verify Deployment

```powershell
# Check tất cả container
docker ps

# Check container trên cùng network
docker network inspect test_network | grep -E "qdrant|conversational"

# Kiểm tra API health
curl -v http://localhost:8001/ 2>&1

# Kiểm tra logs
docker logs -f conversational-rag-api

# Deep test: search thử
$body = @{messages=@(@{role="user"; content="test query"})} | ConvertTo-Json
curl -X POST http://localhost:8001/rag/ -H "Content-Type: application/json" -d $body
```
