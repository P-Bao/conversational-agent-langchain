# Hướng Dẫn Vận Hành, Migration Dữ Liệu & Đánh Giá DeepEval Với NVIDIA NIM

Tài liệu này hướng dẫn chi tiết quy trình vận hành hệ thống RAG (Retrieval-Only v6.0.0), thực hiện migration dữ liệu từ Mongo dump vào Qdrant (bao gồm xử lý trùng/ghi đè collection), và chạy đánh giá DeepEval qua **NVIDIA NIM API** với giới hạn 30 requests/giây.

---

## 1. Kiến Trúc Môi Trường

- **Qdrant Vector Database**: Đang chạy riêng tại thư mục `D:\Code\Python\Ami\RAG\qdrant_docker` qua Docker, cung cấp endpoint tại `http://localhost:6333`.
- **RAG Retrieval Backend (v6.0.0)**: Chạy qua Docker Container (`docker-compose.yml`) hoặc môi trường `uv` local. Sử dụng model `BAAI/bge-m3` cho dense (1024-dim) + sparse (`bge-m3-sparse`) và `BAAI/bge-reranker-v2-m3` cho reranking.
- **NVIDIA NIM Eval LLM**: Sử dụng NVIDIA NIM API (`https://integrate.api.nvidia.com/v1`) làm LLM đánh giá cho DeepEval (`ContextualPrecisionMetric`, `ContextualRecallMetric`).

---

## 2. Khởi Động Các Dịch Vụ

### Bước 2.1: Bật Qdrant Vector Database
Di chuyển tới thư mục Qdrant riêng và khởi động container:
```powershell
cd D:\Code\Python\Ami\RAG\qdrant_docker
docker compose up -d
```
*(Kiểm tra dashboard Qdrant hoạt động tại `http://localhost:6333/dashboard`)*

### Bước 2.2: Khởi Động RAG Backend Service
Quay lại thư mục project `conversational-agent-langchain`:
```powershell
cd D:\Code\Python\Ami\RAG\conversational-agent-langchain
```

**Cách 1: Khởi động qua Docker (Khuyên dùng)**
```powershell
docker compose up --build -d
```
*API Swagger Documentation sẽ có tại: `http://localhost:8001/docs`*

**Cách 2: Khởi động qua Python/uv local**
```powershell
uv sync
uv run uvicorn agent.api:app --reload --port 8001
```

---

## 3. Migration Dữ Liệu Vào Qdrant

Script `agent.scripts.migrate_dump_to_qdrant` nạp dữ liệu từ các file dump JSON trong `input/`, thực hiện chunking, trích xuất dense + sparse vector BGE-m3 và upsert vào Qdrant.

### Trường Hợp A: Ghi Đè Collection Cũ (Xóa collection trùng và tạo mới hoàn toàn)
Khi bạn muốn làm sạch dữ liệu cũ hoặc schema Qdrant cũ chưa đúng chuẩn BGE-m3:

**Dùng cờ CLI `--recreate`:**
```powershell
uv run python -m agent.scripts.migrate_dump_to_qdrant --recreate
```

**Hoặc dùng biến môi trường:**
```powershell
$env:RECREATE_COLLECTION="true"
uv run python -m agent.scripts.migrate_dump_to_qdrant
```

### Trường Hợp B: Resume / Bổ Sung Chunks Mới (Giữ nguyên dữ liệu đã có)
Script tự động kiểm tra checkpoint (`migration_checkpoint.jsonl`), bỏ qua các `global_id` đã có và chỉ upsert các chunk chưa có:
```powershell
uv run python -m agent.scripts.migrate_dump_to_qdrant
```

### Trường Hợp C: Thử Nghiệm Với Số Lượng Giới Hạn (Limit)
Chạy thử nghiệm migration trên 10 document đầu tiên để kiểm tra pipeline:
```powershell
uv run python -m agent.scripts.migrate_dump_to_qdrant --limit 10 --recreate
```

---

## 4. Chạy Đánh Giá DeepEval Với NVIDIA NIM Model

### Bước 4.1: Cấu Hình Biến Môi Trường NVIDIA NIM
Mở file `.env` (hoặc tạo từ `template.env`) và điền API key của NVIDIA NIM:

```env
# === NVIDIA NIM eval LLM ===
NVIDIA_API_KEY=nvapi-your-actual-key-here
NVIDIA_EVAL_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_EVAL_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_EVAL_RPS=30
```

*Lưu ý: `NVIDIA_EVAL_RPS=30` đảm bảo không vượt quá giới hạn 30 requests/sec.*

### Bước 4.2: Định Vị Chunks Mẫu Sau Migration (Locator Step)
Sau khi migration hoàn tất, chạy script locator 1 lần để cập nhật `expected_chunk_locators` trong dataset `tests/golden_questions_v2.json`:
```powershell
uv run python tests/locate_expected_chunks.py
```

### Bước 4.3: Chạy Test DeepEval
Thực hiện chạy suite test retrieval quality với Pytest:
```powershell
$env:ALLOW_NETWORK_TESTS="1"; uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
```

---

## 5. Bảng Tóm Tắt Lệnh Nhanh (Cheat Sheet)

| Tác vụ | Lệnh thực hiện |
| ------ | -------------- |
| Bật Qdrant độc lập | `cd D:\Code\Python\Ami\RAG\qdrant_docker; docker compose up -d` |
| Bật RAG Backend Docker | `docker compose up --build -d` |
| Migrate (Ghi đè collection) | `uv run python -m agent.scripts.migrate_dump_to_qdrant --recreate` |
| Migrate (Thử 10 docs) | `uv run python -m agent.scripts.migrate_dump_to_qdrant --limit 10` |
| Định vị locators test | `uv run python tests/locate_expected_chunks.py` |
| Run DeepEval (NVIDIA NIM) | `$env:ALLOW_NETWORK_TESTS="1"; uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv` |
