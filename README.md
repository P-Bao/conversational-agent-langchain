# Conversational RAG Agent (v6 Retrieval-Only)

Backend RAG retrieval service: trả về context (documents) cho downstream LLMs bằng model BGE-m3 (dense + sparse lexical weights) và BGE-reranker-v2-m3.

## Features (v6.0.0)
- **Retrieval-Only API**: `/rag` và `/rag/stream` trả về danh sách các document chunks đã qua hybrid retrieval + rerank (không sinh answer ở backend RAG).
- **BGE-m3 Multi-functional Embedding**: Một model `BAAI/bge-m3` cho cả dense (1024-dim) và sparse (lexical weights) qua named vectors Qdrant (`bge-m3-sparse`).
- **BGE Reranker v2-m3**: Multilingual reranker cho tiếng Việt và đa ngôn ngữ.
- **DeepEval với Qwen Self-host**: Suite đánh giá `ContextualPrecision` và `ContextualRecall` cùng custom locator verification dùng Qwen OpenAI-compatible API.

## Documentation

Bộ tài liệu bàn giao đầy đủ tại [`docs/`](docs/README.md):

| Lĩnh vực | File |
|---|---|
| Kien trúc | [ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Cài đặt | [SETUP.md](docs/SETUP.md) |
| Triển khai Docker | [DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| API Reference | [API_REFERENCE.md](docs/API_REFERENCE.md) |
| Cấu hình env | [CONFIGURATION.md](docs/CONFIGURATION.md) |
| Vận hành | [OPERATIONS.md](docs/OPERATIONS.md) |
| Troubleshooting | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| Bảo mật | [SECURITY.md](docs/SECURITY.md) |
| Testing | [TESTING.md](docs/TESTING.md) |
| Phát triển | [DEVELOPMENT.md](docs/DEVELOPMENT.md) |
| Đánh giá DeepEval | [EVALUATION.md](docs/EVALUATION.md) |
| Nhập dữ liệu | [DATA_INGESTION.md](docs/DATA_INGESTION.md) |
| Thuật ngữ | [GLOSSARY.md](docs/GLOSSARY.md) |
| User Guide | [USER_GUIDE.md](docs/USER_GUIDE.md) |
| Handover Checklist | [HANDOVER_CHECKLIST.md](docs/HANDOVER_CHECKLIST.md) |

## Quickstart

1. Sao chép `template.env` thành `.env` và thiết lập các biến môi trường:
   ```bash
   cp template.env .env
   ```

2. Chạy Qdrant & backend API bằng `uv`:
   ```bash
   uv sync
   uv run uvicorn agent.api:app --reload --port 8001
   ```

## Architecture

```
User Query -> REST /rag -> Qdrant Hybrid Search (BGE-m3 dense + sparse) -> BGE Reranker v2-m3 -> RetrievalResponse
```

| Route | Method | Description |
| ----- | ------ | ----------- |
| `/rag/` | POST | Trả về `RetrievalResponse(query, documents)` |
| `/rag/stream` | POST | Stream NDJSON các sự kiện `status` và `documents` |
| `/semantic/search` | POST | Direct semantic search endpoint |

## Testing & Evaluation

- **Unit tests**:
  ```bash
  uv run pytest tests/unit_tests -q
  ```

- **Qwen DeepEval**:
  ```bash
  $env:ALLOW_NETWORK_TESTS="1"
  uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
  ```
