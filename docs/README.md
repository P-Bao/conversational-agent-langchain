# Bộ Tài Liệu Bàn Giao — Retrieval & Search API v8.1.0

> **Project**: Retrieval-Only RAG API (FastAPI + Remote BGE-m3 + Qdrant + LangGraph)
>
> **Purpose**: API nhận câu hỏi, truy xuất document chunks liên quan nhất từ Qdrant
> bằng hybrid search (dense + sparse, gọi remote BGE-m3 qua HTTP) và remote
> reranker (default), trả về danh sách documents cho downstream LLM. Collection /
> embedding / delete thuộc về hệ thống quản lý Qdrant bên ngoài — repo này
> **chỉ retrieval & search**.

## Mục Lục Tài Liệu

### Nền Tảng

| File | Đối tượng | Mô tả |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Engineer + BA/PM | Kiến trúc hệ thống, data flow, component, design decisions |
| [GLOSSARY.md](GLOSSARY.md) | BA/PM, non-tech | Thuật ngữ: embedding, retrieval, rerank, remote, ... |

### Cài Đặt & Triển Khai

| File | Đối tượng | Mô tả |
|---|---|---|
| [SETUP.md](SETUP.md) | Engineer mới | Local dev setup: uv, .env, docker |
| [DEPLOYMENT.md](DEPLOYMENT.md) | DevOps / SRE | Triển khai Docker (Qdrant + API), healthcheck, restart |
| [CONFIGURATION.md](CONFIGURATION.md) | Engineer + Ops | Tất cả env vars cho retrieval/search, defaults, gợi ý chỉnh |

### API & Dữ Liệu

| File | Đối tượng | Mô tả |
|---|---|---|
| [API_REFERENCE.md](API_REFERENCE.md) | Engineer, backend client | Endpoints: `/rag`, `/rag/stream`, `/semantic/search`, `/healthz`, `/readyz`, curl/Python examples |
| [USER_GUIDE.md](USER_GUIDE.md) | End-user, BA | Cách sử dụng API, đọc response, FAQs |
| [DATA_INGESTION.md](DATA_INGESTION.md) | Engineer muốn nạp dữ liệu | Repo **không** làm ingestion — xem hệ thống quản lý Qdrant ngoài |

### Chất Lượng & Kiểm Thử

| File | Đối tượng | Mô tả |
|---|---|---|
| [TESTING.md](TESTING.md) | Engineer | Test markers, commands, fixtures, coverage |
| [EVALUATION.md](EVALUATION.md) | Engineer, QA | DeepEval (Qwen/NVIDIA), golden dataset, metrics |

### Vận Hành & Bảo Trì

| File | Đối tượng | Mô tả |
|---|---|---|
| [OPERATIONS.md](OPERATIONS.md) | SRE / On-call | Runbook: start/stop/logs/backup/health-check/capacity |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Engineer, SRE | Diagnostic checklist cho các lỗi thường gặp |
| [SECURITY.md](SECURITY.md) | Security, Ops | Secret management, network exposure, dependency security |

### Bàn Giao

| File | Đối tượng | Mô tả |
|---|---|---|
| [DEVELOPMENT.md](DEVELOPMENT.md) | Dev mới | Code structure, conventions, patterns, dev workflow |
| [HANDOVER_CHECKLIST.md](HANDOVER_CHECKLIST.md) | PM, Tech Lead | Checklist xác nhận bàn giao hoàn tất |

## Quick Links

| Cần gì? | Đọc ở đâu |
|---|---|
| "API bị 500, không start được" | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) #1 |
| "Muốn xem health của API & Qdrant" | [OPERATIONS.md](OPERATIONS.md) §6 (health check) |
| "Scores retrieval có tốt không?" | [EVALUATION.md](EVALUATION.md) |
| "Muốn thay embedding model" | [CONFIGURATION.md](CONFIGURATION.md) §1 + [DEVELOPMENT.md](DEVELOPMENT.md) §4 |
| "Deploy lên Docker" | [DEPLOYMENT.md](DEPLOYMENT.md) |
| "Đổi reranker default" | [CONFIGURATION.md](CONFIGURATION.md) §3 |

## Lưu Ý Quan Trọng (v8.1.0)

1. **Ingestion tách khỏi retrieval.** Các endpoint `/collection/create`,
   `/embeddings/documents`, `/embeddings/string`, `/embeddings/delete` đã
   được **loại bỏ khỏi repo này** — thuộc về hệ thống ngoài quản lý Qdrant.
2. **Migration Mongo dump → Qdrant (`migrate_dump_to_qdrant.py`) đã được
   loại bỏ** — thuộc về repo ingestion ngoài.
3. **Default reranker = `remote`** (HTTP server). Cần `RERANK_BASE_URL` trỏ tới
   server chạy BGE-reranker; lọc theo `RERANK_MIN_SCORE`. Alternatives: `bge`
   (local FlagEmbedding), `none` (passthrough).
4. **Health check**: `/healthz` (liveness) + `/readyz` (Qdrant connectivity).
5. **Retrieval hybrid** (dense + sparse fusion) từ remote BGE-m3; `top_k` 1-40.
