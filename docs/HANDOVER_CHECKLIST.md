# Handover Checklist — Conversational Agent LangChain v6.0.0

> Tài liệu này xác nhận các thành phần đã sẵn sàng để bàn giao.
> Dùng checklist này cho buổi họp bàn giao (handover meeting).

## 1. Source Code

- [x] Repository `conversational-agent-langchain` (Python 3.13, uv)
- [x] Branch chính: `feature/retrieval-only-bge-qwen`
- [x] Pyproject.toml version: 6.0.0
- [x] `src/agent/` — API backend
- [x] `frontend/` — Streamlit GUI
- [x] `tests/` — Pytest suite (41/41 unit tests pass)

## 2. Container / Image

- [x] `Dockerfile` — base image `astral-sh/uv:python3.13-bookworm-slim`
- [x] `docker-compose.yml` — service API (:8001)
- [x] `qdrant_docker/docker-compose.yml` — Qdrant service riêng

## 3. Configuration

- [x] `template.env` — mẫu, copy sang `.env` trước khi chạy
- [x] `.env` — file làm việc thực tế (đã khớp template)
- [x] Tất cả env vars đều có default an toàn trong `config.py`

## 4. API Endpoints

- [x] `POST /rag/` — RetrievalResponse (query + documents)
- [x] `POST /rag/stream` — NDJSON stream
- [x] `POST /semantic/search` — direct hybrid search
- [x] `POST /collection/create/{name}` — Qdrant collection
- [x] `POST /embeddings/documents` — upload PDF/txt
- [x] `POST /embeddings/string/` — embed raw text
- [x] `DELETE /embeddings/delete/{source}` — delete by source

## 5. Eval / Testing

- [x] Unit tests: `make test` / `pytest tests/unit_tests`
- [x] VCR tests: `make test-vcr`
- [x] DeepEval suite (Qwen / NVIDIA NIM): `pytest tests/test_rag_deepeval_qwen.py`
- [x] Golden questions dataset: `tests/golden_questions_v2.json`
- [x] Chunk locator tool: `tests/locate_expected_chunks.py`

## 6. Scripts

- [x] `migrate_dump_to_qdrant.py` — E2E migration Mongo dump → Qdrant
- [x] `chunking.py` — text split + merge short + LLM enrich
- [x] `dump_reader.py` — read Extended JSON từ `input/`
- [x] `load_dummy_data.py` — quick upload test data

## 7. Documentation Handover

- [x] `docs/README.md` — Index
- [x] `docs/ARCHITECTURE.md` — Kiến trúc hệ thống
- [x] `docs/SETUP.md` — Hướng dẫn cài đặt
- [x] `docs/CONFIGURATION.md` — Cấu hình env vars
- [x] `docs/DEPLOYMENT.md` — Triển khai Docker
- [x] `docs/API_REFERENCE.md` — API chi tiết
- [x] `docs/DATA_INGESTION.md` — Nhập dữ liệu
- [x] `docs/EVALUATION.md` — Hướng dẫn đánh giá
- [x] `docs/OPERATIONS.md` — Runbook
- [x] `docs/TROUBLESHOOTING.md` — Xử lý lỗi
- [x] `docs/SECURITY.md` — Bảo mật
- [x] `docs/TESTING.md` — Hướng dẫn chạy test
- [x] `docs/DEVELOPMENT.md` — Phát triển
- [x] `docs/GLOSSARY.md` — Thuật ngữ
- [x] `docs/USER_GUIDE.md` — Hướng dẫn sử dụng API
- [x] `README.md` (root) — đã cập nhật link tới `docs/`

## 8. Known Issues / Gotchas

| # | Vấn đề | Workaround / Fix | Mức ảnh hưởng |
|---|---|---|---|
| 1 | `QDRANT_URL=http://localhost` trong template.env dễ gây lỗi khi chạy Docker vì localhost trong container chỉ container | Sửa thành `http://host.docker.internal` (Docker Desktop) hoặc service name nếu cùng network. Xem [DEPLOYMENT.md](DEPLOYMENT.md) | Critical |
| 2 | Lần đầu startup, model BGE-m3 (2.2GB) + reranker (2.2GB) phải download từ HuggingFace. `htpps://` lỗi cert ở 1 số môi trường internal | Set `HF_ENDPOINT=https://hf-mirror.com` ở China / proxy network | Medium |
| 3 | Qdrant collection cũ (pre-v6) dùng sparse `fast-sparse-bm25`, v6 dùng `bge-m3-sparse`. Collection cũ không tương thích. | `curl -X DELETE localhost:6333/collections/documents` → chạy lại migration với `--recreate` | Breaking |
| 4 | Embedding & reranker chạy CPU-only (không GPU) chậm hơn 3-5x, ngốn RAM ~8GB cho 2 model. | Không có giải pháp ngoài trang bị GPU NVIDIA. Giới hạn concurrent requests. | Performance |
| 5 | Frontend hiện có `process_rag_stream` xử lý event `content` nhưng API v6 không còn sinh content → frontend hiển thị trống. Frontend cần cập nhật nếu muốn dùng. | Xem [USER_GUIDE.md](USER_GUIDE.md) | Frontend breaking |

## 9. Thông Tin Liên Hệ (Điền khi bàn giao)

| Vai trò | Người | Email |
|---|---|---|
| Project Owner | | |
| Tech Lead | | |
| SRE / DevOps | | |
| Contact cho migration | | |

## 10. Trạng Thái Bàn Giao

- [ ] Tất cả mục 1-7 đã verify
- [ ] Team nhận đã chạy được `make test` (pass)
- [ ] Team nhận đã deploy thành công stack Docker
- [ ] Team nhận đã chạy được migration dataset mẫu
- [ ] Buổi họp handover hoàn tất
- [ ] Git tag release v6.0.0
- [ ] Documentation merge vào main
