# Hướng Dẫn Cài Đặt — Retrieval & Search API v7.1.0

> **Khác v7.0.0:**
> - Port API đổi `8001` → **`8005`**, network đổi `test_network` → **`ami-network`** (external).
> - Dockerfile dùng base image **PyTorch CUDA** (reranker chạy local FlagEmbedding, cần GPU).
> - Embedding server tách riêng — repo ngoài expose `http://bge-m3-embed:8008`.
> - Reranker default đổi `none` → **`bge`** (local FlagReranker, tải model ~1.1GB lần đầu).
> - Embedding trả cả **dense + sparse** (hybrid retrieval, không còn dense-only).
> - Tuỳ chọn: bật **query transformation** qua Qwen self-host.

## 1. Yêu Cầu Hệ Thống

| Component | Version tối thiểu | Ghi chú |
|---|---|---|
| Python | 3.10+ | Local dev (uv package manager) |
| uv | 0.5+ | Project package manager (`uv sync`) |
| Docker + Docker Compose | Docker 24+, Compose v2 | Production deployment |
| NVIDIA GPU | CUDA 12.6+ | **Bắt buộc** cho local reranker (FlagReranker). Có thể tắt bằng `RERANK_PROVIDER=none` nếu không có GPU |
| Qdrant | 1.18+ | Vector DB chạy riêng, **collection do hệ thống ingestion ngoài dựng** |
| Embedding server | — | Remote BGE-m3 server (repo ngoài, expose `http://<host>:8008`). API container chỉ gọi HTTP |
| Qwen server (optional) | — | Chỉ cần khi bật `QUERY_TRANSFORM_ENABLED=true` — OpenAI-compatible endpoint `http://<host>:8000/v1` |

## 2. Cài Đặt Local (Development)

### Bước 1: Clone Repository

```bash
git clone <repo-url> conversational-agent-langchain
cd conversational-agent-langchain
git checkout feat/qwen-query-transform-nim-eval   # hoặc branch release tương ứng
```

### Bước 2: Cài uv (nếu chưa có)

```bash
# Windows (winget)
winget install astral.uv

# Hoặc pip
pip install uv

uv --version
```

### Bước 3: Sync Dependencies

```bash
uv sync
```

Tạo `.venv/` và cài dependencies trong `pyproject.toml` (bao gồm `FlagEmbedding==1.4.0`, `transformers==4.57.1`, `langchain-openai>=0.3.0`).

> Reload LOCAL (không Docker) cần GPU để chạy reranker local. Nếu không có
> GPU: set `RERANK_PROVIDER=none` trong `.env` để passthrough.

### Bước 4: Start embedding-server (repo ngoài)

Embedding không chạy trong repo này. Bật embedding-server (repo ngoài) trước
khi start API. Server expose HTTP `POST /embed` trả dense + sparse vectors.
URL mặc định trong Docker network: `http://bge-m3-embed:8008`.

### Bước 5: Tạo .env từ template

```bash
cp template.env .env
```

Sửa `.env` (tối thiểu cho dev):

```env
# === Qdrant (collection phải có sẵn do ingestion ngoài dựng) ===
QDRANT_URL=http://localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# === Embedding (remote BGE-m3 — start embedding-server trước) ===
EMBEDDING_PROVIDER=remote
EMBEDDING_BASE_URL=http://localhost:8008    # hoặc http://bge-m3-embed:8008 trong Docker
EMBEDDING_API_KEY=                            # trống nếu server không bật auth

# === Reranker (local FlagReranker — cần GPU) ===
RERANK_PROVIDER=bge                           # 'bge' (local) | 'remote' | 'none'
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TOP_K=5

# === Query Transformation (tuỳ chọn — cần Qwen server) ===
QUERY_TRANSFORM_ENABLED=false                # true để bật
QWEN_BASE_URL=http://localhost:8000/v1
QWEN_API_KEY=dummy
QWEN_MODEL=qwen
```

### Bước 6: Verify Local

```bash
uv run uvicorn agent.api:app --reload --port 8005
```

Mở `http://localhost:8005/docs` để xem Swagger UI. Kiểm tra:

```bash
# Liveness (luôn 200 nếu process sống)
curl http://localhost:8005/healthz
# {"status":"ok"}

# Readiness (cần Qdrant + collection tồn tại)
curl http://localhost:8005/readyz
```

Nếu `/readyz` trả `503 collection_missing` → nạp data qua hệ thống ingestion
ngoài (xem [DATA_INGESTION.md](DATA_INGESTION.md)).

## 3. Cài Đặt Docker (Production / Test)

### Bước 1: Network + Qdrant + Embedding Server

```bash
# Tạo network chung (1 lần)
docker network create ami-network

# Start Qdrant (repo ngoài)
cd ../qdrant_docker && docker compose up -d

# Start embedding-server (repo ngoài — expose port 8008, cùng ami-network)
cd ../embedding-server && docker compose up -d
```

### Bước 2: Build & Start API

```bash
cd ../conversational-agent-langchain
docker compose up --build -d
```

> Container cần **GPU** (Docker Compose đã reserve `nvidia` driver, all GPUs).
> Nếu host không có GPU, sửa `docker-compose.yml` bỏ block `deploy.resources`
> và set `RERANK_PROVIDER=none` trong `.env`.

### Bước 3: Kiểm Tra

```bash
curl http://localhost:8005/healthz
# {"status":"ok"}

curl http://localhost:8005/readyz
# {"status":"ready","collection":"documents"}  (nếu Qdrant OK + collection tồn tại)

curl http://localhost:8005/docs   # Swagger UI
```

## 4. Query Transformation (Optional)

Để bật node `query_transform` rewrite + step-back + decompose trước retrieve:

1. Start Qwen server (self-host, OpenAI-compatible, port 8000).
2. Validate SDK:
   ```bash
   curl http://localhost:8000/v1/models
   ```
3. Cập nhật `.env`:
   ```env
   QUERY_TRANSFORM_ENABLED=true
   QWEN_BASE_URL=http://localhost:8000/v1
   QWEN_API_KEY=dummy
   QWEN_MODEL=qwen
   ```
4. Restart API: `docker compose restart` (hoặc `--reload` nếu đang chạy local).
5. Verify pipeline: logs sẽ ghi `Query transformation done: rewritten=...` khi gọi `/rag/`.

> Pipeline chạy song song 3 LLM calls (rewrite + step-back + decompose) qua
> `RunnableParallel` để giảm latency. Nếu Qwen server lỗi, fallback về câu
> hỏi gốc — retrieval vẫn hoạt động.

## 5. Kiểm Tra Kết Nối

```bash
# 1. Liveness
curl http://localhost:8005/healthz

# 2. Readiness
curl http://localhost:8005/readyz

# 3. Search thử (direct, không rerank)
curl -X POST http://localhost:8005/semantic/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test","k":3}'

# 4. Qdrant health
curl http://localhost:6333/

# 5. Embedding server health
curl http://localhost:8008/   # hoặc http://bge-m3-embed:8008 trong Docker

# 6. Qwen server health (nếu bật query transform)
curl http://localhost:8000/v1/models

# 7. Unit tests
uv run pytest tests/unit_tests -q
```

## 6. Các Tình Huống Thường Gặp

### `503 collection_missing` ở `/readyz`

Collection chưa tồn tại trên Qdrant — chạy hệ thống ingestion ngoài hoặc tạo
qua Qdrant Dashboard (`http://localhost:6333/dashboard`).

### `503 qdrant_unreachable` ở `/readyz`

Qdrant chưa chạy hoặc sai host/port. Trong container, `localhost` trỏ về
chính container — dùng `http://qdrant` (service name) hoặc
`http://host.docker.internal` (Windows/Mac).

Chi tiết: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Embedding server không trả lời

API container không tải model — embedding delegate tới `EMBEDDING_BASE_URL`.
Nếu query treo hoặc timeout:

1. Verify embedding-server đang chạy: `curl http://localhost:8008/`.
2. Đảm bảo `EMBEDDING_BASE_URL` đúng (Docker: `http://bge-m3-embed:8008`,
   local: `http://localhost:8008`).
3. Nếu server bật auth, set `EMBEDDING_API_KEY` trùng với `BGE_API_KEY`
   trong `embedding-server/.env`.
4. Tăng `EMBEDDING_TIMEOUT` (default 60s) nếu mạng chậm.

### Reranker tải model rất chậm lần đầu

`RERANK_PROVIDER=bge` tải `BAAI/bge-reranker-v2-m3` (~1.1GB) từ HuggingFace
**lần đầu tiên**. Sau đó model được cache trong `hf-cache/` volume (Docker)
hoặc `~/.cache/huggingface` (local). Các request sau sẽ nhanh.

> Không có GPU: set `RERANK_PROVIDER=none` để passthrough (truncate top-K).

### Query transformation không chạy

- Kiểm tra `QUERY_TRANSFORM_ENABLED=true` trong `.env`.
- Kiểm tra Qwen server reachable: `curl http://localhost:8000/v1/models`.
- Xem log API: nếu thấy `Query transformation failed, falling back to
  original query`, Qwen server có vấn đề (rate limit, model name sai, ...).
- Vẫn hoạt động ở fallback mode (dùng câu hỏi gốc).

### `Unknown reranker provider: '...'`

Chỉ chấp nhận `bge` (default), `remote`, `none`. Giá trị `cohere` / `flashrank`
đã bỏ. Xem [CONFIGURATION.md](CONFIGURATION.md) §4.

## 7. Frontend (Streamlit)

> **Repo frontend đã chuyển sang repository riêng** từ v7. Liên hệ team frontend
> để biết URL clone. Repo này chỉ cung cấp API.
