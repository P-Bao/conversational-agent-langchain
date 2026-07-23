# Đánh Giá — Evaluation Guide (v7.0.0)

> Lưu ý v7: `tests/test_rag_deepeval_qwen.py` chạy qua **TestClient.post("/rag/", ...)**
> thay vì gọi `Graph().invoke(...)` trực tiếp. Lý do:
> - Đồng nhất với cách caller thực sự dùng API (qua HTTP).
> - Giảm số mock cần thiết — chỉ mock `graph.with_config(...)().ainvoke`.
> - Dễ test edge cases (404, schema thay đổi).

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

### Với Qwen self-host:

```bash
# Set env
export ALLOW_NETWORK_TESTS=1

# Run full eval
uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv

# Output:
# test_qwen_deepeval_retrieval ... PASSED
```

### Với NVIDIA NIM:

```bash
# Set NVIDIA API key (ưu tiên hơn Qwen)
export NVIDIA_API_KEY=nvapi-your-key
export ALLOW_NETWORK_TESTS=1

uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
```

Auto-detection: Nếu `NVIDIA_API_KEY` hoặc `NVIDIA_EVAL_API_KEY` set → dùng
`NvidiaEvalLLM`; ngược lại → dùng `QwenEvalLLM`.

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
QWEN_EVAL_THINKING=false
```

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
