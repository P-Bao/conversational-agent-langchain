# Hướng Dẫn Cài Đặt — Conversational Agent LangChain v6.0.0

## 1. Yêu Cầu Hệ Thống

| Component | Version tối thiểu | Ghi chú |
|---|---|---|
| Python | 3.13+ | Bắt buộc (uv package manager) |
| uv | 0.5+ | Project package manager |
| Docker + Docker Compose | Docker 24+, Compose v2 | Cho production deployment |
| Qdrant | 1.18+ | Vector database chạy riêng biệt |
| GPU (CUDA) | Optional | Không bắt buộc — CPU fallback OK, chậm hơn 3-5x |

## 2. Cài Đặt Local (Development)

### Bước 1: Clone Repository

```powershell
git clone <repo-url> conversational-agent-langchain
cd conversational-agent-langchain
```

### Bước 2: Cài uv (nếu chưa có)

```powershell
# Windows (winget)
winget install astral.uv

# Hoặc pip
pip install uv

# Kiểm tra
uv --version
```

### Bước 3: Sync Dependencies

```powershell
uv sync
```

Lệnh này tạo virtual env `.venv/` và cài đầy đủ dependencies trong `pyproject.toml`
(cả dev dependencies). Tương đương `pip install -r requirements.txt` nhưng nhanh hơn.

### Bước 4: Tạo .env từ template

```powershell
copy template.env .env
```

Sửa `.env`:

```env
# === Qdrant ===
QDRANT_URL=http://localhost       # Nếu Qdrant chạy local trên host
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# === Embedding (giữ mặc định) ===
AU_EMBED_MODEL_NAME=BAAI/bge-m3
AU_EMBED_DIMENSION=1024
```

### Bước 5: Verify

```powershell
uv run uvicorn agent.api:app --reload --port 8001
```

Mở `http://localhost:8001/docs` để xem Swagger UI. Nếu thấy API response `500`,
kiểm tra Qdrant đã chạy chưa.

## 3. Cài Đặt Docker (Production / Test)

### Bước 1: Qdrant

Xem hướng dẫn chi tiết ở [DEPLOYMENT.md](DEPLOYMENT.md). Cơ bản:

```powershell
cd ../qdrant_docker
docker compose up -d
```

### Bước 2: Build & Start API

```powershell
cd conversational-agent-langchain
docker compose up --build -d
```

### Bước 3: Kiểm Tra

```powershell
curl http://localhost:8001/
# Response: "Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!"
curl http://localhost:8001/docs
```

## 4. Cài Đặt Frontend (Streamlit, Optional)

```powershell
cd frontend
uv sync
uv run streamlit run assistant.py --theme.base="dark"
```

Frontend chạy tại `http://localhost:8501`, gọi backend qua `BACKEND_HOST:BACKEND_PORT`
(mặc định `localhost:8001`).

## 5. Kiểm Tra Kết Nối

Sau khi setup, chạy smoke test:

```powershell
# Ping API
curl http://localhost:8001/

# Qdrant health check (nếu cùng host)
curl http://localhost:6333/

# Chạy unit tests
uv run pytest tests/unit_tests -q
```

## 6. Các Tình Huống Thường Gặp

### Connection refused với Qdrant

Nguyên nhân: Qdrant chưa chạy hoặc sai host/port. Kiểm tra:

```powershell
# Qdrant đang run?
docker ps | findstr qdrant

# Qdrant phản hồi?
curl http://localhost:6333/

# Trong container, localhost là container, không phải host
# -> dùng host.docker.internal (Windows/Mac) hoặc IP bridge
```

Chi tiết xem [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Embedding load chậm

Lần đầu chạy, model BGE-m3 (2.2GB) và BGE-reranker-v2-m3 được tự động tải từ
HuggingFace Hub về `~/.cache/huggingface`. Có thể mất 2-5 phút tùy bandwidth.
Các lần sau dùng cache nên nhanh.

### Lỗi torch / CUDA

Nếu không có GPU, code tự động dùng CPU (`use_fp16=False`). Không cần cài CUDA.
Nếu gặp lỗi torch, kiểm tra:

```powershell
uv run python -c "import torch; print(torch.__version__)"
```
