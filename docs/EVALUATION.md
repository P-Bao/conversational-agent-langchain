# Đánh Giá — Evaluation Guide (v8.1.0)

> Lưu ý v7.1: `tests/test_rag_deepeval_qwen.py` gọi **API Docker container đang chạy
> thật** qua HTTP (`httpx` tới `RAG_API_URL` mặc định `http://localhost:8005`), thay
> vì dùng in-process `TestClient`. Lý do:
> - Tránh tự ý lấy endpoint mặc định / mock embedding — test chỉ pass khi service thật
>   (Docker + Colab notebook embedding + Qdrant có data) thực sự hoạt động.
> - Đồng nhất với cách caller thực sự dùng API (qua HTTP).
> - Cuối test assert tỷ lệ pass `≥ TEST_MIN_PASS_RATIO` (default 0.7) — không còn
>   "PASSED" giả khi 0/14 câu hỏi fail.

## 1. Tổng Quan

Hệ thống dùng **DeepEval** để đánh giá chất lượng retrieval:
- `ContextualPrecisionMetric`: đo tỷ lệ document relevant trong top-K
- `ContextualRecallMetric`: đo tỷ lệ expected context được retrieve thành công

Metric này dùng một **Eval LLM** (judge) — hỗ trợ 2 backend:

| Backend | Env prefix | Rate limit | Chi phí |
|---|---|---|---|
| NVIDIA NIM | `NVIDIA_*` | 30 req/s (NVIDIA_EVAL_RPS) | Tính phí theo token |
| Qwen self-host | `QWEN_*` | Không | Local |

Chọn backend nào tuỳ theo tài nguyên:
- **Qwen**: cần GPU host model Qwen local, chạy unlimited request.
- **NVIDIA NIM**: API cloud, có rate limit, tốn phí nhưng không cần GPU local.

## 2. Dataset: Golden Questions

File: `tests/golden_questions_v2.json`

Format:

```json
[
  {
    "question": "Attention trong Transformer hoạt động thế nào?",
    "expected_context": ["The attention mechanism allows..."],
    "expected_chunk_locators": [{"global_id": "550e8400-..."}]
  }
]
```

| Field | Mô tả |
|---|---|
| `question` | Query test |
| `expected_context` | Mảng text fragment mong đợi xuất hiện trong retrieved documents |
| `expected_chunk_locators` | List global_id mong đợi (định danh chính xác) |

Locator được đánh giá qua `assert_chunk_locators()`:
1. Nếu global_id match → pass
2. Nếu không, fallback content check (expected_context substring match)
3. Nếu cả 2 không match → fail

## 3. Locator Step (Sau Migration)

Sau mỗi lần migration dữ liệu (bên ingestion repo ngoài), `global_id` có thể
thay đổi nếu có thay đổi chunking. Cần chạy script locator để cập nhật
`expected_chunk_locators`:

```bash
uv run python tests/locate_expected_chunks.py
```

> Script làm việc với Qdrant trực tiếp — có thể chạy ở bất kỳ agent nào
> có Qdrant access. Repo này không đổi `tests/locate_expected_chunks.py`.

## 4. Run Evaluation

> **Yêu cầu v7.1**: Test gọi `POST /rag/` qua TestClient, route này lại gọi remote
> BGE-m3 embedding (và remote reranker). Trước khi chạy eval, **phải**:
> 1. Chạy remote embedding server (GPU) và lấy public URL / base URL.
> 2. Set `EMBEDDING_BASE_URL` (và `RERANK_BASE_URL` nếu `RERANK_PROVIDER=remote`,
>    mặc định là `remote`) tới URL đó. Nếu không, mọi câu hỏi sẽ fail ở bước
>    embedding hoặc rerank (fail-fast khi remote không khả dụng).
> 3. Qdrant collection (`TEST_QDRANT_COLLECTION_NAME`, default `documents`) phải
>    đã được dựng bởi hệ thống ingestion ngoài và có data (`/readyz` trả 200).

### Với Qwen self-host:

```bash
# Set env (PowerShell)
$env:ALLOW_NETWORK_TESTS="1"
$env:TEST_EVAL_BACKEND="qwen"            # hoặc để trống + set QWEN_EVAL_BASE_URL
$env:QWEN_EVAL_BASE_URL="http://localhost:8000/v1"
$env:QWEN_EVAL_MODEL="qwen"
$env:EMBEDDING_BASE_URL="http://localhost:8008"   # remote embed server
$env:RERANK_BASE_URL="http://localhost:8010"      # remote rerank server
$env:RERANK_PROVIDER="remote"                     # default "remote"

# Run full eval
uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv

# Output:
# test_qwen_deepeval_retrieval ... PASSED
```

### Với NVIDIA NIM:

```bash
# Set NVIDIA API key (ưu tiên hơn Qwen nếu cùng set)
$env:ALLOW_NETWORK_TESTS="1"
$env:TEST_EVAL_BACKEND="nvidia"          # hoặc set NVIDIA_API_KEY để auto-detect
$env:NVIDIA_API_KEY="nvapi-your-key"
$env:NVIDIA_EVAL_MODEL="meta/llama-3.3-70b-instruct"
$env:EMBEDDING_BASE_URL="http://localhost:8008"   # remote embed server

uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
```

Auto-detection: nếu `TEST_EVAL_BACKEND` set → dùng trực tiếp; ngược lại nếu
`NVIDIA_API_KEY` hoặc `NVIDIA_EVAL_API_KEY` set → `NvidiaEvalLLM`; nếu
`QWEN_EVAL_BASE_URL` set → `QwenEvalLLM`; còn lại → default `QwenEvalLLM`.

### Eval LLM Config:

```env
# NVIDIA (rate-limited)
NVIDIA_API_KEY=nvapi-your-key
NVIDIA_EVAL_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_EVAL_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_EVAL_RPS=30

# Qwen (no rate limit)
QWEN_EVAL_BASE_URL=http://localhost:8000/v1
QWEN_EVAL_API_KEY=
QWEN_EVAL_MODEL=qwen
```

### Tuning env (optional):

| Env | Default | Mô tả |
|---|---|---|
| `RAG_API_URL` | `http://localhost:8005` | Base URL API Docker container (test gọi `POST {RAG_API_URL}/rag/`) |
| `TEST_QDRANT_COLLECTION_NAME` | `documents` | Collection Qdrant để eval |
| `TEST_LOCATOR_STRICT` | `0` | `1` = fail test khi locator/context mismatch (mặc định lenient) |
| `TEST_SKIP_DEEPEVAL` | `0` | `1` = bỏ qua metrics DeepEval, chỉ kiểm tra retrieval/locator (nhanh) |
| `TEST_DEEPEVAL_TOP_K` | `5` | Số top-K context đưa vào `LLMTestCase.retrieval_context` |
| `TEST_MIN_PASS_RATIO` | `0.7` | Tỷ lệ câu hỏi tối thiểu phải pass để test assert pass (0.0-1.0) |

## 5. Interpret Results

DeepEval output:

```
test_qwen_deepeval_retrieval (test_rag_deepeval_qwen.py) ...
  ✓ Contextual Precision: 0.85 (threshold=0.5)
  ✓ Contextual Recall: 0.78 (threshold=0.5)
  ✓ Chunk Locators: all expected chunks found
```

| Metric | Range | Good | Acceptable | Poor |
|---|---|---|---|---|
| Contextual Precision | 0-1 | > 0.8 | 0.5-0.8 | < 0.5 |
| Contextual Recall | 0-1 | > 0.7 | 0.5-0.7 | < 0.5 |

Nếu fail → kiểm tra:
- Data có trong collection không? (`curl /readyz` phải 200)
- `RETRIEVAL_K` đủ lớn không?
- Eval LLM online không? (rate limit, model name)

Xem log DeepEval để debug từng test case.

## 6. Add New Test Cases

1. Thêm entry vào `tests/golden_questions_v2.json`:
   - `question`: câu hỏi thật (Vietnamese nếu data gốc tiếng Việt)
   - `expected_context`: trích dẫn chính xác đoạn trong tài liệu gốc
2. Chạy locator để cập nhật global_id:
   ```bash
   uv run python tests/locate_expected_chunks.py
   ```
3. Commit + push cả `golden_questions_v2.json`

## 7. Test Markers (Chạy Một Phần)

```bash
# Chỉ eval (mất ~5-10 phút)
uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv --durations=0

# Kết hợp unit tests trước (mất ~1-2 phút)
uv run pytest tests/unit_tests -q && \
  uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv

# Chỉ custom locator assertion, skip DeepEval metrics (nhanh) — chưa support
```

> `ALLOW_NETWORK_TESTS=1` là bắt buộc vì DeepEval gọi eval LLM qua HTTP.

## 8. Mock Retrieval For Evaluation (khi chưa có data thật)

Nếu cần chạy eval mà Qdrant chưa có data (dev/test), có thể patch
`graph.with_config(...)().ainvoke`:

```python
# tests/fakes/deepeval_graph.py
from langchain_core.documents import Document

async def fake_ainvoke(state, **_):
    return {
        "query": state["messages"][-1]["content"],
        "documents": [
            Document(page_content="...", metadata={"global_id": "...", "source": "..."})
        ],
    }

# Patch in test
with patch("agent.routes.rag.graph") as g:
    g.with_config.return_value.ainvoke = fake_ainvoke
    response = rag_client.post("/rag/", json={...})
```

Cách này cho phép test golden questions với ground truth đã biết (assertion trước)
trước khi đẩy lên Qdrant thật.
