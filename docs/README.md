# Bộ Tài Liệu Bàn Giao — Conversational Agent LangChain v6.0.0

> **Project**: Retrieval-Only RAG API (FastAPI + BGE-m3 + Qdrant + LangChain)
>
> **Purpose**: API nhận câu hỏi, truy xuất document chunks liên quan nhất từ Qdrant
> bằng hybrid search (dense + sparse) và reranker BGE, trả về danh sách documents
> cho downstream LLM.

## Mục Lục Tài Liệu

### Nền Tảng

| File | Đối tượng | Mô tả |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Engineer + BA/PM | Kiến trúc hệ thống, data flow, component, design decisions |
| [GLOSSARY.md](GLOSSARY.md) | BA/PM, non-tech | Thuật ngữ: embedding, hybrid, rerank, RRF, chunking... |

### Cài Đặt & Triển Khai

| File | Đối tượng | Mô tả |
|---|---|---|
| [SETUP.md](SETUP.md) | Engineer mới | Local dev setup: uv, .env, docker |
| [DEPLOYMENT.md](DEPLOYMENT.md) | DevOps / SRE | Triển khai Docker (Qdrant + API), network, production config |
| [CONFIGURATION.md](CONFIGURATION.md) | Engineer + Ops | Tất cả env vars, defaults, gợi ý chỉnh |

### API & Dữ Liệu

| File | Đối tượng | Mô tả |
|---|---|---|
| [API_REFERENCE.md](API_REFERENCE.md) | Engineer, backend client | Endpoints: request/response schema, curl/Python examples |
| [USER_GUIDE.md](USER_GUIDE.md) | End-user, BA | Cách sử dụng API, đọc response, FAQs |
| [DATA_INGESTION.md](DATA_INGESTION.md) | Engineer, Ops | Nạp dữ liệu: API upload + Mongo dump migration script |

### Chất Lượng & Kiểm Thử

| File | Đối tượng | Mô tả |
|---|---|---|
| [TESTING.md](TESTING.md) | Engineer | Test markers, commands, fixtures, coverage |
| [EVALUATION.md](EVALUATION.md) | Engineer, QA | DeepEval (Qwen/NVIDIA), golden dataset, metrics |

### Vận Hành & Bảo Trì

| File | Đối tượng | Mô tả |
|---|---|---|
| [OPERATIONS.md](OPERATIONS.md) | SRE / On-call | Runbook: start/stop/logs/backup/monitoring/capacity |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Engineer, SRE | Diagnostic checklist cho các lỗi thường gặp |
| [SECURITY.md](SECURITY.md) | Security, Ops | Secret management, network exposure, model security |

### Bàn Giao

| File | Đối tượng | Mô tả |
|---|---|---|
| [DEVELOPMENT.md](DEVELOPMENT.md) | Dev mới | Code structure, conventions, patterns, dev workflow |
| [HANDOVER_CHECKLIST.md](HANDOVER_CHECKLIST.md) | PM, Tech Lead | Checklist xác nhận bàn giao hoàn tất |

## Quick Links

| Cần gì? | Đọc ở đâu |
|---|---|
| "API bị 500, không start được" | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) #1 |
| "Muốn thêm collection mới" | [API_REFERENCE.md](API_REFERENCE.md) #5 |
| "Cần nạp dữ liệu mới từ dump" | [DATA_INGESTION.md](DATA_INGESTION.md) #2 |
| "Scores retrieval có tốt không?" | [EVALUATION.md](EVALUATION.md) #5 |
| "Muốn thay embedding model" | [CONFIGURATION.md](CONFIGURATION.md) #1 + [DEVELOPMENT.md](DEVELOPMENT.md) #4 |
