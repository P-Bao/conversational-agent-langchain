# Handover Checklist — Retrieval & Search API v7.1.0

> Tài liệu này xác nhận các thành phần đã sẵn sàng để bàn giao.
> Dùng checklist này cho buổi họp bàn giao (handover meeting).

## 1. Source Code

- [x] Repository `conversational-agent-langchain` (Python 3.10+, uv)
- [x] Branch: `feat/qwen-query-transform-nim-eval` (từ `feature/retrieval-search-only`)
- [x] `pyproject.toml` version: 7.1.0
- [x] `src/agent/` — FastAPI backend (retrieval & search, optional query transform)
- [x] `tests/` — Pytest suite (70+ unit + integration + DeepEval NIM)
- [ ] **Ingestion repo (ngoài)** — phải được team ingestion cung cấp (xem [DATA_INGESTION.md](DATA_INGESTION.md))
- [ ] **Frontend repo (ngoài)** — phải được team frontend cung cấp (Streamlit)

## 2. Container / Image

- [x] `Dockerfile` — base image `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` (GPU cho local reranker)
- [x] `docker-compose.yml` — service API (:8005) healthcheck `/healthz`, GPU reservation, volume `hf-cache`, network `ami-network`
- [x] `qdrant_docker/docker-compose.yml` — Qdrant service riêng
- [x] `embedding-server/` (repo ngoài) — BGE-m3 server (port 8008)
- [ ] **Qwen server (optional)** — self-host LLM cho query transform (port 8000)

## 3. Configuration

- [x] `template.env` — mẫu v8 (có `QUERY_TRANSFORM_ENABLED`, `QWEN_*`, `EMBEDDING_API_KEY`, `RERANK_MODEL`, v.v.)
- [x] `.env` — file làm việc thực tế (đã khớp template)
- [x] Tất cả env vars có default an toàn trong `config.py`
- [x] Backward-compat: `AU_EMBED_BASE_URL` (→ `EMBEDDING_BASE_URL`), `AU_EMBED_API_KEY` (→ `EMBEDDING_API_KEY`), `AU_RERANK_MODEL_NAME` (→ `RERANK_MODEL`), `QDRANT_COLLECTION_NAME` (→ `QDRANT_COLLECTION`, `qdrant_collection`)

## 4. API Endpoints (v7.1.0)

- [x] `GET /` — Welcome
- [x] `GET /healthz` — Liveness probe (process sống)
- [x] `GET /readyz` — Readiness probe (Qdrant + collection)
- [x] `POST /rag/` — `RetrievalResponse` (LangGraph: query_transform? → retriever + rerank)
- [x] `POST /rag/stream` — NDJSON stream
- [x] `POST /semantic/search` — `SearchResponse[]` (direct hybrid search, no rerank)

### Endpoints đã loại (chuyển hệ ingestion ngoài)

- [x] `POST /collection/create/{name}` → **404**
- [x] `POST /embeddings/documents` → **404**
- [x] `POST /embeddings/string/` → **404**
- [x] `DELETE /embeddings/delete/{source}` → **404**

## 5. Eval / Testing

- [x] Unit tests: `pytest tests/unit_tests` (70+ tests pass)
- [x] Integration tests: `pytest tests/test_integration.py` (`/`, `/docs`, routes 404)
- [x] Contract tests: `pytest tests/vcr/test_contracts.py`
- [x] E2E tests: `RUN_LIVE_E2E=1 pytest tests/test_stream.py` (cần service running)
- [x] DeepEval suite: `ALLOW_NETWORK_TESTS=1 NVIDIA_API_KEY=xxx pytest tests/test_rag_deepeval_nim.py -m qwen -vv`
  - 5 metrics: Correctness (GEval), Faithfulness, ContextualRelevancy, ContextualPrecision, ContextualRecall
  - Judge: NVIDIA NIM `meta/llama-3.3-70b-instruct` (rate limit 30 RPS)
  - Auto generate answer từ retrieved context
- [x] Golden questions dataset: `tests/golden_questions_v2.json` (14 câu)
- [x] Health check tests: `tests/unit_tests/test_health.py` (6 tests pass)

## 6. Scripts

> **Không còn script migration/dump/chunking** trong repo này — thuộc về repo ingestion ngoài.

## 7. Documentation Handover (v7.1.0)

- [x] `docs/README.md` — Index v7.1.0 (cập nhật)
- [x] `docs/ARCHITECTURE.md` — Graph conditional, hybrid retrieval, local reranker
- [x] `docs/SETUP.md` — Local dev + Docker (port 8005, GPU, Qwen, ami-network)
- [x] `docs/CONFIGURATION.md` — Env vars (embedding hybrid, reranker local bge, Qwen query transform, NIM eval)
- [x] `docs/DEPLOYMENT.md` — Docker (PyTorch CUDA base, requirements.txt, GPU reservation)
- [x] `docs/API_REFERENCE.md` — Chỉ `/rag`, `/semantic`, `/healthz`, `/readyz` + endpoint đã bỏ
- [x] `docs/DATA_INGESTION.md` — Redirect sang repo ingestion ngoài
- [x] `docs/EVALUATION.md` — DeepEval NIM 5 metrics, evaluate() batch, answer generation
- [x] `docs/OPERATIONS.md` — Runbook (port 8005, model cache, query transform latency)
- [x] `docs/TROUBLESHOOTING.md` — Lỗi query transform, GPU, auth embedding, fallback
- [x] `docs/SECURITY.md` — Secret management, model provenance, dependency scan
- [x] `docs/TESTING.md` — Test files v7.1 + marker `qwen` (file NIM)
- [x] `docs/DEVELOPMENT.md` — Source tree mới, query_transform node, test rename
- [x] `docs/GLOSSARY.md` — Query Transformation, Hybrid Retrieval, NIM-only DeepEval
- [x] `docs/USER_GUIDE.md` — Bỏ FAQ upload
- [x] `README.md` (root) — Link tới `docs/`, version v7.1.0, lệnh DeepEval NIM
- [x] `CHANGELOG.md` — v7.1.0 + nhóm changes nhánh hiện tại

## 8. Known Issues / Gotchas

| # | Vấn đề | Workaround / Fix | Mức ảnh hưởng |
|---|---|---|---|
| 1 | `QDRANT_URL=http://qdrant` nhầm thành double `=` | Default `http://qdrant` trong template.env. Verify container network. | Critical |
| 2 | Embedding server (repo ngoài) down | Restart `embedding-server` stack, check `EMBEDDING_BASE_URL`/`EMBEDDING_API_KEY` | Medium |
| 3 | Qdrant collection mới cần recreate khi dense size đổi | Ingestion repo ngoài chịu. Repo này `/readyz` 503 detect. | Breaking |
| 4 | `[Timing]` API chậm lần đầu (reranker load model 1.1GB + embedding server warm) | Khác local — image Docker không tải embedding model. Healthcheck `start_period=60s`. | Performance |
| 5 | `RERANK_PROVIDER=bge` thêm ~1.4GB VRAM | Default `bge` cho precision. Không có GPU → `RERANK_PROVIDER=none`. | Performance |
| 6 | Frontend cũ (v6) gọi `/embeddings/documents` → 404 | Frontend v7 phải dùng repo frontend ngoài. | Migration |
| 7 | Query transform fail (Qwen down) → vẫn trả docs (fallback) | By design. Kiểm tra Qwen server nếu cần recall cao. | Info |
| 8 | Docker image nặng ~6GB | PyTorch base + FlagEmbedding + transformers. Multi-stage build có thể giảm. | Future improvement |
| 9 | Network `ami-network` external — phải tạo trước | `docker network create ami-network` (1 lần). | Setup |

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
- [x] Team nhận đã chạy được `uv run pytest tests/unit_tests -q` (70+ pass)
- [ ] Team nhận đã deploy thành công stack Docker (Qdrant + embedding-server + API)
- [ ] Team nhận đã xác nhận `/readyz` trả 200 (Qdrant + collection OK)
- [ ] Team ingestion đã xác nhận pipeline ingestion ngoài đang chạy
- [ ] Buổi họp handover hoàn tất
- [ ] Git tag release v7.1.0
- [ ] Documentation merge vào main