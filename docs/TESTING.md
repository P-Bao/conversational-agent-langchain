# Hướng Dẫn Chạy Test — Testing Guide (v7.0.0)

## 1. Test Categories

Pyproject markers: `pyproject.toml`

| Marker | Mô tả | Ví dụ |
|---|---|---|
| `unit` | Pure unit tests, no external dependencies | `test_reranker`, `test_vdb`, `test_config`, `test_health` |
| `integration` | Integration tests with in-process deps | `test_integration.py`, `tests/vcr/test_contracts.py` |
| `contract` | Endpoint contract tests (no external HTTP) | `tests/vcr/test_contracts.py` |
| `e2e` | End-to-end tests (cần services thật) | `test_stream.py` |
| `vcr` | Tests backed by recorded HTTP cassettes | `tests/vcr/test_contracts.py` |
| `qwen` | DeepEval evaluation suite | `test_rag_deepeval_qwen.py` |
| `slow` | Long-running tests | — |

## 2. Quick Reference

| Lệnh | Mô tả |
|---|---|
| `make test` | Unit + integration, loại bỏ vcr/e2e, với coverage |
| `make test-vcr` | Run VCR-recorded tests (no external calls) |
| `make update-vcr-tests` | Re-record VCR cassettes (cần network) |
| `make test-e2e` | E2E tests (cần external services running) |
| `uv run pytest tests/unit_tests -q` | Chỉ unit tests |
| `uv run pytest tests/vcr -q` | VCR tests |
| `uv run pytest -m qwen -vv` | DeepEval eval suite (cần `ALLOW_NETWORK_TESTS=1`) |

## 3. Chi Tiết Các Lệnh

### Unit Tests (61/61 pass ở v7.0.0):

```bash
uv run pytest tests/unit_tests -q
```

Các test file trong `tests/unit_tests/` (v7.0.0):

| File | Mục đích |
|---|---|
| `test_search.py` | `POST /semantic/search` route, `get_retriever` dense-only/cache (không còn RRF/DBSF/fusion) |
| `test_rag_route.py` | `POST /rag/` + `POST /rag/stream`, NDJSON events |
| `test_retrieval_node.py` | `retrieve_documents` qua LangGraph: K, retry, rerank skip/apply |
| `test_reranker.py` | `get_reranker(provider=...)`: none / remote (bge/cohere/flashrank đã bỏ) |
| `test_vdb.py` | `qdrant_client` & `async_qdrant_client` singleton |
| `test_config.py` | Pydantic `Config` defaults (`embedding_provider=remote`, `embedding_base_url=""`, `rerank_provider=none`, qdrant, retrieval, rerank) |
| `test_embeddings_bge.py` | `BGEM3RemoteEmbeddings` wrapper (test qua mocked httpx, không load model thật) |
| `test_health.py` | `/healthz` + `/readyz` (collection_exists + error branches) |

### Contract Tests (VCR):

```bash
uv run pytest tests/vcr/test_contracts.py -v
```

Kiểm tra schema `/semantic/search` request/response không thay đổi so với
recorded contract.

### E2E Tests:

```bash
RUN_LIVE_E2E=1 uv run pytest -m "e2e" -v tests/
```

Yêu cầu: Qdrant + API services đang chạy.

### DeepEval (Qwen / NVIDIA):

```bash
ALLOW_NETWORK_TESTS=1 uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
```

Chi tiết xem [EVALUATION.md](EVALUATION.md).

## 4. Test Configuration

### `tests/conftest.py`:

- `test_env_defaults`: set mặc định env vars cho test (`EMBEDDING_PROVIDER=remote`,
  `EMBEDDING_BASE_URL=http://mock` mock, `RERANK_PROVIDER=none`, `QDRANT_URL=...`).
- `block_external_http`: mặc định chặn HTTP ra ngoài (trừ localhost). Bỏ qua
  nếu `ALLOW_NETWORK_TESTS=1` hoặc test có marker `vcr` / `qwen`.
- `app` fixture: import FastAPI app (v7 không có module-level side effects nên
  không cần patch).
- `client` fixture: FastAPI `TestClient`.

### Environment cho tests:

```bash
EMBEDDING_PROVIDER=remote
EMBEDDING_BASE_URL=http://mock-embedding
EMBEDDING_TIMEOUT=60
RERANK_PROVIDER=none
RERANK_BASE_URL=http://mock-reranker
RERANK_TIMEOUT=60
QDRANT_URL=http://localhost
QDRANT_PORT=6333
QDRANT_API_KEY=test_api_key
QDRANT_COLLECTION_NAME=documents
```

## 5. Fixtures

| Fixture | Scope | Mô tả |
|---|---|---|
| `app` | session | FastAPI app instance |
| `client` | function | `TestClient` cho HTTP tests |
| `resources_path` | session | Path đến `tests/resources/` |
| `vcr_config` | module | VCR config (filter headers, record mode) |
| `golden_questions` | module | Load `tests/golden_questions_v2.json` (DeepEval) |
| `eval_llm` | module | `NvidiaEvalLLM` hoặc `QwenEvalLLM` (DeepEval) — chọn qua `TEST_EVAL_BACKEND` hoặc auto-detect |
| `rag_api_url` | module | Base URL API Docker thật (default `http://localhost:8001`, override qua `RAG_API_URL`) |

## 6. Coverage

```bash
uv run pytest -n auto -m "not vcr and not e2e" -v \
  --cov=src/agent --cov-report=term --cov-report=html tests/
```

Output:
- Terminal: summary table
- `htmlcov/`: detailed HTML report
- Omit: `tests/*`

## 7. Best Practices

- Viết test mới → match marker với loại test
- Dùng VCR cassette cho external API calls (tránh network dependency)
- Set `ALLOW_NETWORK_TESTS=1` chỉ cho test có network thật (e2e, qwen)
- Trước commit: `uv run pytest tests/unit_tests -q` (~30s)

## 8. Test Files Đã Loại Bỏ (v7)

Các file sau đã được **xoá** vì nội dung ingestion / CRUD đã chuyển sang hệ ngoài:

- `tests/unit_tests/test_search_delete.py` (chứa test cho route `/embeddings/delete` đã bỏ)
- `tests/unit_tests/test_rag_graph.py` (graph không bị xoá nhưng đã viết lại test ở `test_retrieval_node.py` + `test_rag_route.py`)
- `tests/unit_tests/test_routes_embeddings.py` (routes/embeddings.py đã bỏ)
- `tests/unit_tests/test_embedding_management.py` (EmbeddingManagement đã bỏ)
- `tests/unit_tests/test_utility.py` (`utils/utility.py` đã bỏ)
- `tests/vcr/test_embedding_and_reranker_requests.py` (Cohere reranker đã bỏ)
