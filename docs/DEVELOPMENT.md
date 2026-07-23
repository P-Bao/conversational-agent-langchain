# Hướng Dẫn Phát Triển — Development Guide

## 1. Project Structure

```
conversational-agent-langchain/
├── src/agent/
│   ├── __init__.py
│   ├── api.py                      # FastAPI app entry
│   ├── backend/
│   │   ├── graph.py                # LangGraph StateGraph pipeline
│   │   ├── state.py                # AgentState TypedDict
│   │   ├── nodes/retrieval.py      # retrieve_documents node
│   │   └── services/embedding_management.py
│   ├── data_model/
│   │   ├── request_data_model.py   # SearchParams, RAGRequest, EmbeddTextRequest
│   │   ├── response_data_model.py  # SearchResponse, RetrievalResponse, RetrievedDoc
│   │   └── internal_model.py
│   ├── routes/
│   │   ├── rag.py                  # POST /rag/, /rag/stream
│   │   ├── search.py               # POST /semantic/search
│   │   ├── collection.py           # POST /collection/create/{name}
│   │   ├── embeddings.py           # POST /embeddings/documents, /embeddings/string/
│   │   └── delete.py               # DELETE /embeddings/delete/{source}
│   ├── scripts/
│   │   ├── migrate_dump_to_qdrant.py
│   │   ├── chunking.py
│   │   ├── dump_reader.py
│   │   └── load_dummy_data.py
│   └── utils/
│       ├── config.py               # Pydantic Settings
│       ├── embeddings.py           # BGE-m3 dense + sparse
│       ├── vdb.py                  # Qdrant client singleton
│       ├── retriever.py            # Hybrid retriever (RFF/DBSF)
│       ├── reranker.py             # BGE / Cohere / FlashRank
│       └── utility.py              # Helpers
├── frontend/
│   ├── assistant.py                # Streamlit app
│   ├── client.py                   # API client
│   └── pyproject.toml
├── tests/
│   ├── conftest.py
│   ├── unit_tests/
│   ├── vcr/
│   ├── e2e_tests/
│   └── test_rag_deepeval_qwen.py
├── config/
│   └── qdrant.yaml
├── docs/                           # Tài liệu (bàn giao)
├── .env                            # Secrets (gitignored)
├── template.env                    # Mẫu env
├── docker-compose.yml              # API service
├── Dockerfile
├── pyproject.toml                  # Dependencies + tool config
├── Makefile                        # Dev shortcuts
└── ruff.toml                       # Linter config
```

## 2. Code Style & Conventions

### Python:

- Python 3.13+ (`src/agent/`)
- Type hints bắt buộc (mypy: `disallow_untyped_defs = true`)
- Docstring kiểu Google/NumPy (ví dụ: `"""Short description.\n\nArgs:\n----\nReturns:\n-------\n"""`)
- `ruff` cho lint + format (config in `ruff.toml`)
- `ruff format` để format code (tương đương black)
- Không thêm comment thừa — chỉ giữ docstring module-level + function-level

### Import order (theo ruff):

1. Standard library
2. Third-party
3. Local modules (`agent.*`)

### Naming:

| Pattern | Ví dụ |
|---|---|
| Module | `embeddings.py`, `vdb.py` |
| Class | `BGE3Embeddings`, `QdrantClient` (singleton module-level) |
| Function | `get_embedding_model`, `initialize_vector_db` |
| Private | `_get_bge3_model`, `_embeddings_cache` |
| Config field | `embedding_model`, `qdrant_url` |
| Env var | `AU_EMBED_MODEL_NAME`, `QDRANT_URL` |

### No circular imports:

- `config.py` không import từ `agent.*` (độc lập)
- `embeddings.py` import `Config` (đọc model name)
- `vdb.py` import `Config` + `get_sparse_embedding`
- `retriever.py` import `Config` + `embeddings` + `vdb`
- Routes import từ `utils.*`, `backend.*`, `data_model.*`

## 3. Adding a New Endpoint

1. Tạo route module trong `src/agent/routes/` hoặc thêm vào module có sẵn
2. Định nghĩa request/response model (nếu mới) trong `data_model/`
3. Register router trong `api.py`:
   ```python
   app.include_router(router=new_router.router, prefix="/new")
   ```
4. Viết unit test cho logic + VCR cassette cho external calls
5. Thêm endpoint vào [API_REFERENCE.md](API_REFERENCE.md)

## 4. Adding a New Embedding Provider

Theo pattern có sẵn trong `embeddings.py`:

```python
# 1. Thêm case trong get_embedding_model() tại embeddings.py:101-118
match provider:
    case "bge" | "flagembedding":
        return BGE3Embeddings(_get_bge3_model(cfg))
    case "my_new_provider":
        return MyNewEmbeddings(...)

# 2. Thêm env var trong config.py
my_provider_base_url: str = ""
my_provider_api_key: str = ""

# 3. Thêm mapping trong template.env
# MY_PROVIDER_BASE_URL=
# MY_PROVIDER_API_KEY=
```

## 5. Modifying the Retriever / Reranker Flow

Pipeline nằm trong `src/agent/backend/nodes/retrieval.py`.
Thay đổi ở đây ảnh hưởng tới cả `/rag/`, `/rag/stream`, `/semantic/search`.

Để thay đổi behavior:

| Behavior | File cần sửa |
|---|---|
| Thay đổi K | `.env`: `RETRIEVAL_K` |
| Thay đổi fusion | `.env`: `FUSION_ALGORITHM` |
| Thay đổi rerank provider | `.env`: `RERANK_PROVIDER` |
| Thay đổi số doc sau rerank | `.env`: `RERANK_TOP_K` |
| Thêm bước sau rerank | `src/agent/backend/nodes/retrieval.py` |

## 6. Git / Branch Workflow

```powershell
# Feature branch
git checkout -b feature/my-feature

# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Unit tests
uv run pytest tests/unit_tests -q

# Commit
git add -p
git commit -m "feat: description"
```

## 7. Common Dev Tasks

| Task | Lệnh |
|---|---|
| Add dependency | `uv add <package>` |
| Add dev dependency | `uv add --dev <package>` |
| Run linter | `ruff check src/` |
| Format code | `ruff format src/` |
| Run mypy | `uv run mypy src/` |
| Update lock | `uv lock` |
| Sync env | `uv sync` |
| Run typecheck | `uv run pyright src/` |

## 8. Known Dev Gotchas

- `load_dotenv()` được gọi ở `api.py:14` (trước Config) — env vars từ `.env` có sẵn cho Pydantic Settings.
- `initialize_all_vector_dbs()` chạy ở module-level của `api.py` — nếu Qdrant down, app crash ngay.
- `QdrantVectorStore` trong `embeddings.py` viết `local_mode` và `sparse_embedding` param phải match
  collection schema.
- Các test dùng `conftest.py` patching `initialize_all_vector_dbs` — nếu thêm Qdrant call ở module-level,
  nhớ patch trong conftest.
