# Hướng Dẫn Cài Đặt — Retrieval & Search API v8.1.0

## 1. Yêu Cầu Hệ Thống

| Component | Version tối thiểu | Ghi chú |
|---|---|---|
| Python | 3.13+ | Bắt buộc (uv package manager) |
| uv | 0.5+ | Project package manager |
| Docker + Docker Compose | Docker 24+, Compose v2 | Cho production deployment |
| Qdrant | 1.18+ | Vector database chạy riêng biệt, **collection do hệ ngoài dựng** |
| Remote BGE-m3 server | — | GPU server (embed `:8008`, rerank `:8010`). API container chỉ gọi HTTP, không cần GPU/CUDA. Nếu không có rerank server: `RERANK_PROVIDER=bge` (local) hoặc `none` |

## 2. Cài Đặt Local (Development)

### Bước 1: Clone Repository

```bash
git clone <repo-url> conversational-agent-langchain
cd conversational-agent-langchain
git checkout feature/retrieval-search-only
```

### Bước 2: Cài uv (nếu chưa có)

```bash
# Windows (winget)
winget install astral.uv

# Hoặc pip
pip install uv

# Kiểm tra
uv --version
```

### Bước 3: Sync Dependencies

```bash
uv sync
```

Lệnh này tạo virtual env `.venv/` và cài đầy đủ dependencies trong `pyproject.toml`
(cả dev dependencies). Tương đương `pip install -r requirements.txt` nhưng nhanh hơn.

### Bước 4: Tạo .env từ template

```bash
cp template.env .env
```

Sửa `.env` (tối thiểu cho dev):

```env
# === Qdrant (collection phải có sẵn) ===
QDRANT_URL=http://localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# === Embedding (remote BGE-m3 — GPU server trước) ===
EMBEDDING_PROVIDER=remote
EMBEDDING_BASE_URL=http://localhost:8008
# EMBEDDING_TIMEOUT=60

# === Rerank (remote — default) ===
RERANK_PROVIDER=remote
RERANK_BASE_URL=http://localhost:8010
RERANK_MIN_SCORE=0.0
RERANK_TOP_K=5
# RERANK_TIMEOUT=60
```

### Bước 5: Verify

```bash
uv run uvicorn agent.api:app --reload --port 8005
```

Mở `http://localhost:8005/docs` để xem Swagger UI. Kiểm tra:

```bash
# Liveness
curl http://localhost:8005/healthz
# {"status":"ok"}

# Readiness (cần Qdrant + collection)
curl http://localhost:8005/readyz
```

Nếu `/readyz` trả về `503 collection_missing`, bạn cần nạp data vào Qdrant
qua hệ thống ingestion ngoài trước (xem [DATA_INGESTION.md](DATA_INGESTION.md)).

## 3. Cài Đặt Docker (Production / Test)

### Bước 1: Network + Qdrant

```bash
docker network create test_network
cd ../qdrant_docker
docker compose up -d
```

### Bước 2: Build & Start API

```bash
cd conversational-agent-langchain
docker compose up --build -d
```

### Bước 3: Kiểm Tra

```bash
curl http://localhost:8005/healthz
# {"status":"ok"}
curl http://localhost:8005/readyz
# (nếu collection tồn tại) {"status":"ready","collection":"documents"}
curl http://localhost:8005/docs
```

## 4. Frontend (Streamlit)

> **Repo frontend đã chuyển sang repository riêng** (vì v7 cắt ingest/CRUD
> frontend cũ cần nhiều feature ingestion). Repo frontend mới đặt tại nơi
> khác — liên hệ team frontend để biết URL clone.

## 5. Kiểm Tra Kết Nối

Sau khi setup, chạy smoke test:

```bash
# 1. Liveness (luôn 200 nếu process sống)
curl http://localhost:8005/healthz

# 2. Readiness (phải 200 + ready)
curl http://localhost:8005/readyz

# 3. Search thử
curl -X POST http://localhost:8005/semantic/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test","k":3,"collection_name":"documents"}'

# 4. Qdrant health (nếu cùng host)
curl http://localhost:6333/

# 5. Unit tests
uv run pytest tests/unit_tests -q
```

## 6. Các Tình Huống Thường Gặp

### `503 collection_missing` ở `/readyz`

Nguyên nhân: Collection do hệ thống ngoài dựng chưa tồn tại, hoặc `QDRANT_COLLECTION_NAME`
không khớp. Liên hệ team ingestion hoặc tạo qua Qdrant Dashboard.

### `503 qdrant_unreachable` ở `/readyz`

Nguyên nhân: Qdrant chưa chạy hoặc sai host/port. Kiểm tra:

```bash
docker ps | grep qdrant
curl http://localhost:6333/

# Trong container, localhost là container, không phải host
# -> dùng host.docker.internal (Windows/Mac) hoặc IP bridge
```

Chi tiết xem [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Embedding / Rerank server không trả lời

API container không tải model — embedding/rerank delegate tới remote server qua
`EMBEDDING_BASE_URL` / `RERANK_BASE_URL`. Nếu query treo hoặc timeout:

1. Khởi động GPU server (embed + rerank) trên host / server riêng.
2. Cập nhật `.env`: `EMBEDDING_BASE_URL` và `RERANK_BASE_URL` (nếu
   `RERANK_PROVIDER=remote`, mặc định). Khi API chạy trong Docker, dùng
   `http://host.docker.internal:<port>` thay `localhost`.
3. Restart API: `docker compose restart` (hoặc `uv run uvicorn ... --reload`).

> Nếu không có rerank server, set `RERANK_PROVIDER=bge` (local FlagEmbedding)
> hoặc `none` (passthrough). `EMBEDDING_TIMEOUT` / `RERANK_TIMEOUT` (default 60s)
> configurable nếu remote server chậm.

### Không còn torch / CUDA

API container không còn bắt buộc dependency `torch` / `transformers` /
`FlagEmbedding` (embedding remote). Nếu set `RERANK_PROVIDER=bge`, local cần cài
`FlagEmbedding` (qua `uv sync`). Nếu thấy import torch thất bại khi dùng `bge`,
kiểm tra `pyproject.toml` + chạy `uv sync` lại.
