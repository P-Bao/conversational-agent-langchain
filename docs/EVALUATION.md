# Đánh Giá — Evaluation Guide (v7.1.0)

> **Lưu ý v7.1:**
> - File test đổi từ `test_rag_deepeval_qwen.py` → **`test_rag_deepeval_nim.py`**
>   (NVIDIA NIM-only, không còn song song Qwen backend).
> - Số metric tăng từ 2 → **5** (thêm `GEval` Correctness, `Faithfulness`,
>   `ContextualRelevancy` theo reference notebook `example/evaluation_deep_eval.ipynb`).
> - Dùng `evaluate()` batch thay vì assert từng câu (theo pattern notebook).
> - Test gọi **API Docker đang chạy thật** qua HTTP (`httpx` tới `RAG_API_URL`
>   mặc định `http://localhost:8005`) — không dùng in-process `TestClient`.
>   Lý do:
>   - Tránh mock embedding — test chỉ pass khi service thật (Docker +
>     embedding-server + Qdrant có data) thực sự hoạt động.
>   - Đồng nhất với cách caller thật dùng API (qua HTTP).
> - Test tự sinh `actual_output` bằng NIM từ retrieved contexts → mới đo được
>   Correctness/Faithfulness (vì route `/rag/` chỉ trả documents, không có
>   generation node).

## 1. Tổng Quan

DeepEval đánh giá chất lượng RAG qua **5 metric**, tất cả dùng NVIDIA NIM
`meta/llama-3.3-70b-instruct` làm judge:

| Metric | đánh giá gì | Threshold | theo notebook? |
|---|---|---|---|
| `GEval` (Correctness) | actual output có fact-correct so với expected output | 0.5 | ✅ |
| `FaithfulnessMetric` | actual output có bị hallucinate khỏi retrieval context | 0.7 | ✅ |
| `ContextualRelevancyMetric` | retrieval context có relevant với input query | 0.5 | ✅ |
| `ContextualPrecisionMetric` | top-K có nhiều relevant docs (sắp xếp đúng) | 0.5 | extra |
| `ContextualRecallMetric` | có retrieve đủ expected context | 0.5 | extra |

## 2. Dataset: Golden Questions

File: `tests/golden_questions_v2.json` (14 câu hỏi).

```json
[
  {
    "id": 1,
    "topic": "organization",
    "question": "Địa chỉ của Hội đồng Học viện là gì?",
    "expected_context": ["Hội đồng Học viện có địa chỉ tại..."],
    "expected_chunk_locators": [{"global_id": "..."}]
  }
]
```

| Field | Mô tả |
|---|---|
| `id`, `topic` | Phân loại câu hỏi |
| `question` | Query test |
| `expected_context` | List text fragment mong đợi xuất hiện trong retrieved documents. `expected_context[0]` được dùng làm `expected_output` cho Correctness |
| `expected_chunk_locators` | List `global_id` mong đợi (định danh chính xác chunk) |

Locator check (outside DeepEval) qua `assert_chunk_locators()`:
1. Nếu `global_id` match → pass
2. Fallback: content substring match với `expected_context`
3. Không match → fail (lenient mode chỉ warn, strict mode fail test)

## 3. Locator Step (Sau Migration)

Sau mỗi lần migration (bên ingestion repo ngoài), `global_id` có thể đổi nếu
chunking thay. Cập nhật `expected_chunk_locators`:

```bash
uv run python tests/locate_expected_chunks.py
```

> Script làm việc với Qdrant trực tiếp — có thể chạy ở bất kỳ agent nào có
> Qdrant access. Repo này không đổi `tests/locate_expected_chunks.py`.

## 4. Run Evaluation

> **Yêu cầu v7.1**: Trước khi chạy eval, **phải**:
> 1. Chạy Docker stack: Qdrant + embedding-server (repo ngoài) + API container.
> 2. Set `EMBEDDING_BASE_URL` trỏ tới embedding-server (`http://bge-m3-embed:8008`
>    trong Docker network, hoặc `http://localhost:8008` nếu host).
> 3. Qdrant collection (`TEST_QDRANT_COLLECTION_NAME`, default `documents`) phải
>    đã được dựng bởi hệ thống ingestion ngoài và có data (`/readyz` trả 200).
> 4. Set `NVIDIA_API_KEY` (lấy từ https://build.nvidia.com).

### Lệnh chạy:

```bash
# Set env (PowerShell)
$env:ALLOW_NETWORK_TESTS="1"
$env:NVIDIA_API_KEY="nvapi-your-key"
# Tùy chọn: 
$env:EMBEDDING_BASE_URL="http://bge-m3-embed:8008"
$env:RAG_API_URL="http://localhost:8005"
$env:TEST_QDRANT_COLLECTION_NAME="documents"

# Run full eval
uv run pytest tests/test_rag_deepeval_nim.py -m qwen -vv
```

### Bash (Linux/macOS):

```bash
ALLOW_NETWORK_TESTS=1 NVIDIA_API_KEY=nvapi-xxx \
  uv run pytest tests/test_rag_deepeval_nim.py -m qwen -vv
```

### Eval LLM Config (.env hoặc shell):

```env
# NVIDIA NIM (judge + answer generator)
NVIDIA_API_KEY=nvapi-your-key
NVIDIA_EVAL_MODEL=meta/llama-3.3-70b-instruct
NVIDIA_EVAL_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_EVAL_RPS=30
```

### Tuning env (optional):

| Env | Default | Mô tả |
|---|---|---|
| `RAG_API_URL` | `http://localhost:8005` | Base URL API Docker container |
| `TEST_QDRANT_COLLECTION_NAME` | `documents` | Collection Qdrant để eval |
| `TEST_LOCATOR_STRICT` | `0` | `1` = fail test khi locator/context mismatch (mặc định lenient) |
| `TEST_SKIP_DEEPEVAL` | `0` | `1` = bỏ qua DeepEval metrics, chỉ check retrieval/locator (nhanh) |
| `TEST_SKIP_ANSWER_GEN` | `0` | `1` = bỏ NIM answer generation, dùng top-1 chunk làm `actual_output` |
| `TEST_DEEPEVAL_TOP_K` | `5` | Số top-K context đưa vào `LLMTestCase.retrieval_context` |
| `TEST_MIN_PASS_RATIO` | `0.7` | Tỷ lệ câu hỏi tối thiểu phải pass để test assert pass (0.0-1.0) |

## 5. DeepEval Workflow

Test thực hiện 2 bước cho mỗi câu hỏi:

### Step 1: Retrieve + Generate Answer

```
POST {RAG_API_URL}/rag/  body: {"messages": [{"role":"user","content":"<question>"}]}
  -> retrieved_docs: [{"text","score","metadata"}, ...]
  -> retrieved_contexts = [d["text"] for d in retrieved_docs][:TEST_DEEPEVAL_TOP_K]

# Generate actual_output (NIM):
prompt = "You are a helpful assistant. Answer using ONLY the context below..."
actual_output = NvidiaEvalLLM.generate(prompt_with(context=retrieved_contexts, question=question))
```

### Step 2: Build Test Case + Run 5 Metrics

```python
test_cases = create_deep_eval_test_cases(
    questions=[...],
    gt_answers=[expected_context[0] for each question],   # expected_output
    generated_answers=[actual_output for each question],
    retrieved_documents=[retrieved_contexts for each question],
)

metrics = [
    GEval(name="Correctness", model=eval_llm, ...),
    FaithfulnessMetric(threshold=0.7, model=eval_llm),
    ContextualRelevancyMetric(threshold=0.5, model=eval_llm),
    ContextualPrecisionMetric(threshold=0.5, model=eval_llm),
    ContextualRecallMetric(threshold=0.5, model=eval_llm),
]

results = evaluate(test_cases=test_cases, metrics=metrics)
```

### Helper function (theo notebook)

```python
def create_deep_eval_test_cases(questions, gt_answers, generated_answers, retrieved_documents):
    """Build list of LLMTestCase from 4 parallel lists."""
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document,
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]
```

## 6. Interpret Results

DeepEval output:

```
test_nim_deepeval_evaluation (test_rag_deepeval_nim.py) ...
  ✓ Correctness: 0.85 (threshold=0.5)
  ✓ Faithfulness: 0.92 (threshold=0.7)
  ✓ Contextual Relevancy: 0.78 (threshold=0.5)
  ✓ Contextual Precision: 0.85 (threshold=0.5)
  ✓ Contextual Recall: 0.78 (threshold=0.5)
  ✓ Chunk Locators: all expected chunks found
[done] 12/14 passed all 5 metrics (86%, threshold=70%)
```

| Metric | Range | Good | Acceptable | Poor |
|---|---|---|---|---|
| Correctness (GEval) | 0-1 | > 0.8 | 0.5-0.8 | < 0.5 |
| Faithfulness | 0-1 | > 0.8 | 0.7-0.8 | < 0.7 |
| Contextual Relevancy | 0-1 | > 0.8 | 0.5-0.8 | < 0.5 |
| Contextual Precision | 0-1 | > 0.8 | 0.5-0.8 | < 0.5 |
| Contextual Recall | 0-1 | > 0.7 | 0.5-0.7 | < 0.5 |

Cuối test:

```
[done] N/total passed all 5 metrics (X%, locators Y/total, api_failures=Z, threshold=70%)
```

Nếu `ratio < min_pass` → `pytest.fail` với hướng dẫn debug.

### Nếu fail → kiểm tra:

- Data trong collection không? (`curl /readyz` phải 200)
- `RETRIEVAL_K` đủ lớn không?
- `EMBEDDING_BASE_URL` đúng không? (embedding-server reachable)
- `RERANK_PROVIDER` đúng? (bge cần GPU)
- `QUERY_TRANSFORM_ENABLED` quá chậm? (tắt nếu chỉ test retrieval)
- NVIDIA NIM key/quota còn hạn không? ( nhìn logs `RateLimiter.wait()`)
- Answer generation fail? → test tự fallback dùng top-1 chunk làm `actual_output`
- `RAG_API_URL` đúng port? (port mới `8005`)

## 7. Add New Test Cases

1. Thêm entry vào `tests/golden_questions_v2.json`:
   - `question`: câu hỏi thật (Vietnamese nếu data gốc tiếng Việt)
   - `expected_context`: trích dẫn chính xác đoạn trong tài liệu gốc
2. Chạy locator để cập nhật `global_id`:
   ```bash
   uv run python tests/locate_expected_chunks.py
   ```
3. Commit + push cả `golden_questions_v2.json` và locator output.

## 8. Test Markers (Chạy Một Phần)

```bash
# Chỉ eval (mất ~5-15 phút tuỳ NIM rate limit)
uv run pytest tests/test_rag_deepeval_nim.py -m qwen -vv --durations=0

# Kết hợp unit tests trước
uv run pytest tests/unit_tests -q && \
  uv run pytest tests/test_rag_deepeval_nim.py -m qwen -vv

# Skip NIM answer generation (dùng top-1 chunk — nhanh, nhưng Correctness/Faithfulness không đo đầy đủ)
TEST_SKIP_ANSWER_GEN=1 uv run pytest tests/test_rag_deepeval_nim.py -m qwen -vv

# Chỉ locator check, skip DeepEval
TEST_SKIP_DEEPEVAL=1 uv run pytest tests/test_rag_deepeval_nim.py -m qwen -vv
```

> `ALLOW_NETWORK_TESTS=1` bắt buộc vì DeepEval gọi NIM qua HTTPS và API
> Docker qua HTTP.

## 9. Mock Retrieval For Evaluation (khi chưa có data thật)

Nếu cần chạy eval mà Qdrant chưa có data (dev/test), patch graph:

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

# Patch in test:
with patch("agent.routes.rag.graph") as g:
    g.with_config.return_value.ainvoke = fake_ainvoke
    response = rag_client.post("/rag/", json={...})
```

Cách này cho phép test golden questions với ground truth đã biết trước khi
push lên Qdrant thật.

## 10. So sánh với notebook `evaluation_deep_eval.ipynb`

`example/evaluation_deep_eval.ipynb` là tài liệu reference. `test_rag_deepeval_nim.py`
implement đầy đủ các pattern của notebook:

| Notebook cell | Triển khai trong `test_rag_deepeval_nim.py` |
|---|---|
| Cell 1: `GEval(name="Correctness", ...)` | ✅ `correctness_metric` (cùng `evaluation_steps`) |
| Cell 2: `FaithfulnessMetric(threshold=0.7, ...)` | ✅ `faithfulness_metric` |
| Cell 3: `ContextualRelevancyMetric(threshold=1, ...)` | ✅ `relevancy_metric` (threshold 0.5 cho practical) |
| Cell 4: `LLMTestCase(input, expected_output, actual_output, retrieval_context)` | ✅ mỗi câu hỏi |
| Cell 5: `create_deep_eval_test_cases()` helper | ✅ cùng tên hàm |
| Cell 6: `evaluate(test_cases=[...], metrics=[...])` | ✅ gọi batch |
