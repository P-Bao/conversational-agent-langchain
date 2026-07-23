# Handover Checklist — Retrieval & Search API v7.0.0

> Tài liệu này xác nhận các thành phần đã sẵn sàng để bàn giao.
> Dùng checklist này cho buổi họp bàn giao (handover meeting).

## 1. Source Code

- [x] Repository `conversational-agent-langchain` (Python 3.13, uv)
- [x] Branch: `feature/retrieval-search-only` (rút gọn từ `feature/retrieval-only-bge`)
- [x] `pyproject.toml` version: 7.0.0
- [x] `src/agent/` — FastAPI backend (chỉ retrieval & search)
- [x] `tests/` — Pytest suite (61/61 unit + integration pass)
- [ ] **Ingestion repo (ngoài)** — phải được team ingestion cung cấp (xem [DATA_INGESTION.md](DATA_INGESTION.md))
- [ ] **Frontend repo (ngoài)** — phải được team frontend cung cấp (Streamlit)

## 2. Container / Image

- [x] `Dockerfile` — base image `astral-sh/uv:python3.13-bookworm-slim`
- [x] `docker-compose.yml` — service API (:8001) với healthcheck `/healthz`
- [x] `qdrant_docker/docker-compose.yml` — Qdrant service riêng

## 3. Configuration

- [x] `template.env` — mẫu, copy sang `.env` trước khi chạy
- [x] `.env` — file làm việc thực tế (đã khớp template)
- [x] Tất cả env vars đều có default an toàn trong `config.py`
- [x] Backward-compat: `AU_EMBED_MODEL_NAME` (→ `EMBEDDING_MODEL`), v.v.

## 4. API Endpoints (v7.0.0)

- [x] `GET /` — Welcome
- [x] `GET /healthz` — Liveness probe (process sống)
- [x] `GET /readyz` — Readiness probe (Qdrant + collection)
- [x] `POST /rag/` — `RetrievalResponse` (LangGraph + optional rerank)
- [x] `POST /rag/stream` — NDJSON stream
- [x] `POST /semantic/search` — `SearchResponse[]` (direct hybrid search)

### Endpoints đã loại (chuyển hệ ingestion ngoài)

- [x] `POST /collection/create/{name}` → **404** (repo này không tạo collection)
- [x] `POST /embeddings/documents` → **404**
- [x] `POST /embeddings/string/` → **404**
- [x] `DELETE /embeddings/delete/{source}` → **404**

## 5. Eval / Testing

- [x] Unit tests: `pytest tests/unit_tests` (61 tests pass)
- [x] Integration tests: `pytest tests/test_integration.py` (`/`, `/docs`, routes 404)
- [x] Contract tests: `pytest tests/vcr/test_contracts.py`
- [x] E2E tests: `RUN_LIVE_E2E=1 pytest tests/test_stream.py` (cần service running)
- [x] DeepEval suite: `ALLOW_NETWORK_TESTS=1 pytest tests/test_rag_deepeval_qwen.py` (Qwen / NVIDIA NIM)
- [x] Golden questions dataset: `tests/golden_questions_v2.json` (DeepEval goldens)
- [x] Health check tests: `tests/unit_tests/test_health.py` (6 tests pass)

## 6. Scripts

> **Không còn script migration/dump/chunking** trong repo này — thuộc về repo
> ingestion ngoài.

## 7. Documentation Handover (v7.0.0)

- [x] `docs/README.md` — Index v7.0.0 (cập nhật)
- [x] `docs/ARCHITECTURE.md` — Chỉ retrieval + search + health
- [x] `docs/SETUP.md` — Local dev + Docker
- [x] `docs/CONFIGURATION.md` — Env vars mới (bỏ migration/chunking)
- [x] `docs/DEPLOYMENT.md` — Docker + healthcheck config
- [x] `docs/API_REFERENCE.md` — Chỉ `/rag`, `/semantic`, `/healthz`, `/readyz` + danh sách endpoint đã bỏ
- [x] `docs/DATA_INGESTION.md` — Redirect sang repo ingestion ngoài
- [x] `docs/EVALUATION.md` — DeepEval chạy qua TestClient
- [x] `docs/OPERATIONS.md` — Runbook với `/healthz` + `/readyz`
- [x] `docs/TROUBLESHOOTING.md` — `/readyz` failure modes + 404 cho endpoints cũ
- [x] `docs/SECURITY.md` — Bỏ RFI section (upload file xoá)
- [x] `docs/TESTING.md` — Test files v7 + file đã loại
- [x] `docs/DEVELOPMENT.md` — Source tree mới
- [x] `docs/GLOSSARY.md` — Bỏ mục liên quan scripts/migration
- [x] `docs/USER_GUIDE.md` — Bỏ FAQ upload
- [x] `README.md` (root) — Link tới `docs/`

## 8. Known Issues / Gotchas

| # | Vấn đề | Workaround / Fix | Mức ảnh hưởng |
|---|---|---|---|
| 1 | `QDRANT_URL=http://qdrant` nhầm thành double `=` | Default `http://qdrant` trong template.env. Verify container network. | Critical |
| 2 | Lần đầu startup, model BGE-m3 (2.2GB) download từ HuggingFace | Set `HF_ENDPOINT=https://hf-mirror.com` ở China. Cache volume `bge_hf_cache` qua restart. | Medium |
| 3 | Qdrant collection mới cần recreate khi `EMBEDDING_SIZE` đổi | Ingestion repo ngoài chịu. Repo này chỉ `/readyz` 503 → có method detect. | Breaking |
| 4 | `[Timing]` API chậm lần đầu (model download + load) | Healthcheck `start_period=60s` trong compose. | Performance |
| 5 | `RERANK_PROVIDER=bge` tốn thêm ~2GB RAM + ~200ms/request | Default `none`. Bật chỉ khi cần precision cao. | Performance |
| 6 | Frontend cũ (v6) gọi `/embeddings/documents` → 404 | Frontend v7 phải dùng repo frontend ngoài (đã tách). | Migration |
| 7 | `tests/test_embedding_and_reranker_requests.py` (Cohere VCR) đã xoá | Cohere reranker đã bỏ ở v7. | Cleanup |

## 9. Thông Tin Liên Hệ (Điền khi bàn giao)

| Vai trò | Người | Email |
|---|---|---|
| Project Owner | | |
| Tech Lead | | |
| SRE / DevOps | | |
| Team ingestion (collection, embed, migration) | | |
| Team frontend (Streamlit, etc.) | | |
| Team eval (DeepEval, golden dataset) | | |

## 10. Trạng Thái Bàn Giao

- [x] Tất cả mục 1-7 đã verify (code + docs)
- [x] Team nhận đã chạy được `make test` (61/61 pass)
- [ ] Team nhận đã deploy thành công stack Docker
- [ ] Team nhận đã xác nhận `/readyz` trả 200 (Qdrant + collection OK)
- [ ] Team ingestion đã xác nhận pipeline ingestion ngoài đang chạy
- [ ] Buổi họp handover hoàn tất
- [ ] Git tag release v7.0.0
- [ ] Documentation merge vào main
