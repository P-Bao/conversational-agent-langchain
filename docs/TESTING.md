# Hướng Dẫn Chạy Test — Testing Guide

## 1. Test Categories

Pyproject markers: `tests/pyproject.toml:91-98`

| Marker | Mô tả | Ví dụ |
|---|---|---|
| `unit` | Pure unit tests, no external dependencies | test_reranker, test_vdb |
| `integration` | Integration tests with in-process deps | test_rag_graph |
| `contract` | Endpoint contract tests (no external HTTP) | test_contracts |
| `e2e` | End-to-end tests (cần services thật) | test_api |
| `vcr` | Tests backed by recorded HTTP cassettes | test_cohere, test_langsmith |
| `qwen` | DeepEval evaluation suite | test_rag_deepeval_qwen |
| `slow` | Long-running tests | — |

## 2. Quick Reference

| Lệnh | Mô tả |
|---|---|
| `make test` | Unit + integration, loại bỏ vcr/e2e, với coverage |
| `make test-vcr` | Run VCR-recorded tests (no external calls) |
| `make update-vcr-tests` | Re-record VCR cassettes (cần network) |
| `make test-e2e` | E2E tests (cần external services running) |
| `uv run pytest tests/unit_tests -q` | Chỉ unit tests (nhanh, ~30s warm) |
| `uv run pytest tests/vcr -q` | VCR tests (nhanh, ~10s) |
| `uv run pytest -m qwen -vv` | DeepEval eval suite (cần ALLOW_NETWORK_TESTS=1) |

## 3. Chi Tiết Các Lệnh

### Unit Tests (41/41 pass):

```powershell
uv run pytest tests/unit_tests -q
```

Các test file trong `tests/unit_tests/`:

| File | Test | Mô tả |
|---|---|---|
| `test_search_delete.py` | test_search, test_dbsf_fusion, test_bad_fusion | Retriever + Fusion logic |
| `test_embedding_management.py` | test_embedding_management | Upload service |
| `test_reranker.py` | test_rerank_* | Reranker providers |
| `test_rag_graph.py` | test_retrieval_graph | LangGraph pipeline |
| `test_routes_embeddings.py` | test_* | Route handlers |
| `test_utility.py` | test_* | Utility functions |
| `test_vdb.py` | test_* | Vector DB operations |

### VCR Tests:

```powershell
uv run pytest -m "vcr" -v tests/
```

Các cassette được record từ lần chạy đầu, replay cho các lần sau.
Nếu API response của external service thay đổi, cần rewrite:

```powershell
uv run pytest --record-mode=rewrite -m "vcr" -v tests/
```

### E2E Tests:

```powershell
$env:RUN_LIVE_E2E = "1"
uv run pytest -m "e2e" -v tests/
```

Yêu cầu: Qdrant + API services đang chạy.

### Contract Tests:

```powershell
uv run pytest tests/vcr/test_contracts.py -v
```

Kiểm tra schema request/response không thay đổi so với recorded contract.

### DeepEval (Qwen / NVIDIA):

```powershell
# Bắt buộc: cho phép network tới eval LLM
$env:ALLOW_NETWORK_TESTS = "1"
uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
```

Chi tiết xem [EVALUATION.md](EVALUATION.md).

## 4. Test Configuration

### conftest.py (`tests/conftest.py`):

- `test_env_defaults`: set mặc định env vars cho test (bge-m3 models, Qdrant localhost)
- `block_external_http`: mặc định chặn HTTP ra ngoài (trừ localhost). Bỏ qua nếu
  `ALLOW_NETWORK_TESTS=1` hoặc test có marker `vcr` / `qwen`
- `app` fixture: import FastAPI app với `initialize_all_vector_dbs` patched (tránh
  cần Qdrant thật khi test routes)
- `client` fixture: FastAPI TestClient

### Environment for tests:

```powershell
# Default (từ conftest.py, override được nếu set trước)
AU_EMBED_MODEL_NAME=BAAI/bge-m3
AU_EMBED_DIMENSION=1024
AU_SPARSE_MODEL_NAME=BAAI/bge-m3
AU_RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANK_PROVIDER=bge
QDRANT_URL=http://localhost
QDRANT_PORT=6333
QDRANT_API_KEY=test_api_key
```

## 5. Fixtures

| Fixture | Scope | Mô tả |
|---|---|---|
| `app` | session | FastAPI app instance (patched Qdrant) |
| `client` | function | TestClient cho HTTP tests |
| `resources_path` | session | Path đến `tests/resources/` (PDF mẫu) |
| `vcr_config` | module | VCR config (filter headers, record mode) |

## 6. Coverage

```powershell
uv run pytest -n auto -m "not vcr and not e2e" -v --cov=src/agent --cov-report=term --cov-report=html tests/
```

Output:
- Terminal: summary table
- `htmlcov/`: detailed HTML report
- Omit: `tests/*`

## 7. Best Practices

- Viết test mới → match marker với loại test
- Dùng VCR cassette cho external API calls (tránh network dependency)
- Set `ALLOW_NETWORK_TESTS=1` chỉ cho test có network thật (e2e, qwen)
- Trước commit: `uv run pytest tests/unit_tests -q` (41 tests, ~30s)
