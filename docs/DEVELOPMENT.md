# Hướng Dẫn Phát Triển — Development Guide (v7.0.0)

## 1. Project Structure

```
conversational-agent-langchain/
├── src/agent/
│   ├── __init__.py
│   ├── api.py                      # FastAPI app entry — only includes /rag, /semantic, /healthz, /readyz
│   ├── backend/
│   │   ├── graph.py                # LangGraph StateGraph pipeline (goc giu nguyen)
│   │   ├── state.py                # AgentState TypedDict
│   │   └── nodes/retrieval.py      # retrieve_documents node (gọi get_retriever + get_reranker)
│   ├── data_model/
│   │   ├── request_data_model.py   # SearchParams, ChatMessages, RAGRequest
│   │   └── response_data_model.py  # SearchResponse, RetrievalResponse, RetrievedDoc, Status
│   ├── routes/
│   │   ├── rag.py                  # POST /rag/, /rag/stream (LangGraph)
│   │   ├── search.py               # POST /semantic/search
│   │   └── health.py               # GET /healthz, /readyz
│   └── utils/
│       ├── config.py               # Pydantic Settings (rerank_provider default = "none")
│       ├── embeddings.py           # BGE-m3 dense + sparse
│       ├── vdb.py                  # Qdrant client (sync + async) — No collection mgmt
│       ├── retriever.py            # Hybrid retriever (RRF/DBSF)
│       └── reranker.py             # get_reranker() — providers: none / bge
├── tests/
│   ├── conftest.py
│   ├── unit_tests/
│   ├── vcr/
│   ├── e2e_tests/
│   ├── fakes/
│   ├── test_integration.py
│   ├── test_stream.py
│   └── test_rag_deepeval_qwen.py
├── ConvAgentBruno/                 # Bruno API test collection (chi giữ RAG + Search)
│   ├── RAG/{Chat,Stream}.bru
│   └── Search/Search.bru
├── docs/                           # Tai lieu (ban giao v7.0.0)
├── .env                            # Secrets (gitignored)
├── template.env                    # Mau env
├── docker-compose.yml              # API service
├── Dockerfile
├── pyproject.toml                  # Dependencies + tool config
├── Makefile                        # Dev shortcuts
└── ruff.toml                       # Linter config
```

> **Đã xoá ở v7** (cho cả branch này lẫn đã thuộc hệ ngoài quản lý Qdrant):
> - `backend/services/`, `scripts/`, `utils/utility.py`, `data_model/internal_model.py`
> - `routes/{collection,delete,embeddings}.py`
> - `frontend/` (Streamlit repo đã tách)
> - `Dockerfile.frontend`
> - `config/qdrant.yaml` (Qdrant quản lý ngoài repo)
> - `resources/` (legacy demo PDFs + diagrams)
> - `ConvAgentBruno/Embeddings/` (endpoints `/embeddings/*` đã bỏ ở v7)

## 2. Code Style & Conventions

### Python:

- Python 3.13+ (`src/agent/`)
- Type hints bắt buộc (mypy: `disallow_untyped_defs = true`)
- Docstring ngắn gọn (Google/NumPy style)
- `ruff` cho lint + format (config in `ruff.toml`)
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
| Function | `get_embedding_model`, `get_retriever` |
| Private | `_get_bge3_model`, `_embeddings_cache` |
| Config field | `embedding_model`, `qdrant_url` |
| Env var | `EMBEDDING_MODEL`, `QDRANT_URL` |

### No circular imports:

- `config.py` không import từ `agent.*` (độc lập)
- `embeddings.py` import `Config` (đọc model name)
- `vdb.py` import `Config`
- `retriever.py` import `Config` + `embeddings` + `vdb`
- `health.py` import `Config` + `vdb`
- Routes import từ `utils.*`, `backend.*`, `data_model.*`

## 3. Adding a New Endpoint

1. Tạo route module trong `src/agent/routes/` hoặc thêm vào module có sẵn
2. Định nghĩa request/response model (nếu mới) trong `data_model/`
3. Register router trong `api.py`:
   ```python
   app.include_router(router=new_router.router, prefix="/new")
   ```
4. Viết unit test cho logic + integration test qua `TestClient`
5. Cập nhật [API_REFERENCE.md](API_REFERENCE.md)

## 4. Adding a New Embedding Provider

Theo pattern có sẵn trong `embeddings.py`:

```python
# 1. Thêm case trong get_embedding_model()
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

Pipeline graph: `src/agent/backend/nodes/retrieval.py`.
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

```bash
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

- `load_dotenv()` được gọi ở `api.py` (trước `Config`) — env vars từ `.env` có sẵn cho Pydantic Settings.
- **Không có module-level side effect** ở v7 (`vdb.py`, `api.py`). `QdrantClient` được build từ Config nhưng exception nếu Qdrant down sẽ không crash import — chỉ fail khi gọi API.
- `get_retriever` cache `QdrantVectorStore` theo `collection_name`. Cần `retriever_module._vector_store_cache.clear()` giữa các test nếu muốn kiểm tra `QdrantVectorStore(...)` được tạo lại.
- Conftest default `RERANK_PROVIDER=none`. Nếu test cần rerank → set env trước `Config(...)` hoặc explicit override.
- Test deepeval (`test_rag_deepeval_qwen.py`) chạy qua `TestClient.post('/rag/', ...)`, **không gọi `Graph().invoke()` trực tiếp** — dễ test hơn vì không cần mock graph internals.
