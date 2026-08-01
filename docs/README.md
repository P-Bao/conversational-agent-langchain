# Bộ Tài Liệu Bàn Giao — Retrieval & Search API v7.1.0

> **Project**: Retrieval-Only RAG API (FastAPI + Remote BGE-m3 hybrid + Local BGE-reranker + LangGraph + optional Qwen Query Transform)
>
> **Purpose**: API nhận câu hỏi, (tuỳ chọn) biến đổi query qua Qwen LLM, truy xuất document chunks từ Qdrant bằng **hybrid retrieval** (gọi remote BGE-m3 dense+sparse qua HTTP) và **local reranker** (BGE-reranker-v2-m3), trả về danh sách documents cho downstream LLM. Collection / embedding / delete thuộc về hệ thống quản lý Qdrant bên ngoài — repo này **chỉ retrieval & search**.

## Mục Lục Tài Liệu

### Nền Tảng

| File | Đối tượng | Mô tả |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Engineer + BA/PM | Kiến trúc hệ thống, data flow, component, design decisions (query transform, hybrid retrieval, local reranker) |
| [GLOSSARY.md](GLOSSARY.md) | BA/PM, non-tech | Thuật ngữ: query transformation, hybrid retrieval, reranking, DeepEval NIM, ... |

### Cài Đặt & Triển Khai

| File | Đối tượng | Mô tả |
|---|---|---|
| [SETUP.md](SETUP.md) | Engineer mới | Local dev setup: uv, .env, docker (port 8005, ami-network, GPU, Qwen server) |
| [DEPLOYMENT.md](DEPLOYMENT.md) | DevOps / SRE | Triển khai Docker (PyTorch CUDA base, requirements.txt, GPU reservation, multi-stack workflow) |
| [CONFIGURATION.md](CONFIGURATION.md) | Engineer + Ops | Tất cả env vars (embedding hybrid, reranker local bge, Qwen query transform, NIM eval) |

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
| [EVALUATION.md](EVALUATION.md) | Engineer, QA | DeepEval NIM (5 metrics, evaluate() batch, answer generation, golden dataset) |

### Vận Hành & Bảo Trì

| File | Đối tượng | Mô tả |
|---|---|---|
| [OPERATIONS.md](OPERATIONS.md) | SRE / On-call | Runbook: start/stop/logs/backup/health-check/capacity (port 8005, model cache) |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Engineer, SRE | Diagnostic checklist (query transform, GPU, auth embedding, fallback) |
| [SECURITY.md](SECURITY.md) | Security, Ops | Secret management, network exposure, dependency security |

### Bàn Giao

| File | Đối tượng | Mô tả |
|---|---|---|
| [DEVELOPMENT.md](DEVELOPMENT.md) | Dev mới | Code structure, conventions, patterns, dev workflow (query_transform node, test rename) |
| [HANDOVER_CHECKLIST.md](HANDOVER_CHECKLIST.md) | PM, Tech Lead | Checklist xác nhận bàn giao hoàn tất (v7.1.0) |

## Quick Links

| Cần gì? | Đọc ở đâu |
|---|---|
| "API bị 500, không start được" | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) #1 |
| "Muốn xem health của API & Qdrant" | [OPERATIONS.md](OPERATIONS.md) §6 (health check) |
| "Scores retrieval có tốt không?" | [EVALUATION.md](EVALUATION.md) (5 metrics, NIM judge) |
| "Muốn bật query transform" | [CONFIGURATION.md](CONFIGURATION.md) §3 + [SETUP.md](SETUP.md) §4 |
| "Deploy lên Docker" | [DEPLOYMENT.md](DEPLOYMENT.md) (CUDA base, GPU, ami-network) |
| "Đổi reranker default" | [CONFIGURATION.md](CONFIGURATION.md) §4 (bge / none / remote) |
| "Bật query transformation" | [CONFIGURATION.md](CONFIGURATION.md) §3 (`QUERY_TRANSFORM_ENABLED`, `QWEN_*`) |

## Lưu Ý Quan Trọng (v7.1.0)

1. **Ingestion tách khỏi retrieval.** Các endpoint `/collection/create`,
   `/embeddings/documents`, `/embeddings/string`, `/embeddings/delete` đã
   được **loại bỏ khỏi repo này** — thuộc về hệ thống ngoài quản lý Qdrant.
2. **Migration Mongo dump → Qdrant (`migrate_dump_to_qdrant.py`) đã được
   loại bỏ** — thuộc về repo ingestion ngoài.
3. **Port & Network:** API port `8005` (thay `8001`), Docker network `ami-network`
   (external, thay `test_network`).
4. **Hybrid Retrieval:** BGE-m3 trả cả dense + sparse → Qdrant `RetrievalMode.HYBRID`
   (thay dense-only v7.0).
5. **Reranker Local Default:** `RERANK_PROVIDER=bge` (FlagEmbedding, cần GPU).
   Tắt: `RERANK_PROVIDER=none`. Legacy remote vẫn hỗ trợ.
6. **Query Transformation (Optional):** Bật `QUERY_TRANSFORM_ENABLED=true` +
   Qwen server → thêm node `query_transform` (rewrite + step-back + decompose).
   Default `false` → pipeline giữ nguyên v7.0.
6. **DeepEval NIM Only:** File `test_rag_deepeval_nim.py` — 5 metrics (Correctness
   GEval, Faithfulness, ContextualRelevancy, Precision, Recall), judge NVIDIA
   NIM `meta/llama-3.3-70b-instruct`, `evaluate()` batch, auto generate answer.
7. **Docker Image:** Base `pytorch:2.7.1-cuda12.6-cudnn9-runtime` (GPU cho reranker).
   `requirements.txt` chính thức, `pip install` trong Dockerfile.
8. **Health check:** `/healthz` (liveness) + `/readyz` (Qdrant + collection).