# Agent Instructions — Retrieval-only BGE + Qwen DeepEval Refactor

>File này dành cho **phiên agent kế tiếp** thực thi kế hoạch trong `plan.md`.
>Ngày lập: 2026-07-22. Repo: `conversational-agent-langchain`.
>Branch nguồn: `feature/gemini-deepeval-integration`. Branch đích: `feature/retrieval-only-bge-qwen`.

## 1. Nguyên tắc làm việc

- **Luôn đọc `plan.md` trước khi bắt đầu** — toàn bộ ngữ cảnh ở đó.
- **Không tự ý commit/push.** Chỉ khi user yêu cầu rõ ràng "commit đi" / "tạo PR". Xem mục 7.
- Tuân thủ style code đã có trong repo:
  - `ruff` (config `ruff.toml`), `ruff format`.
  - Type hints đầy đủ (pydantic 2, `from __future__ import annotations` không bắt buộc nhưng khuyến nghị theo file mới).
  - Docstring kiểu Google/NumPy theo file xung quanh.
- **KHÔNG thêm comment** trừ khi user yêu cầu (theo system prompt opencode). Chỉ giữ docstring mô tả hàm module-level.
- Khi gặp điểm chờ user confirm (§10 trong plan.md) → dùng tool `question`, không tự quyết.

## 2. Thứ tự task (làm theo số, cập nhật todo qua todowrite)

> Mỗi task: mark `in_progress` trước, `completed` khi xong + verify bước own task. Không batch completion.

### Task 1 — Nhánh & sync
```powershell
git status
git checkout feature/gemini-deepeval-integration
git pull
git checkout -b feature/retrieval-only-bge-qwen
```
- Verify: `git branch --show-current` == `feature/retrieval-only-bge-qwen`.

### Task 2 — Config (`src/agent/utils/config.py`)
- Thêm `validation_alias=AliasChoices(...)` cho `embedding_model`, `embedding_size`, `sparse_model` (mới), `rerank_model`:
  - `AU_EMBED_MODEL_NAME` → `embedding_model` (dense)
  - `AU_EMBED_DIMENSION` → `embedding_size`
  - `AU_SPARSE_MODEL_NAME` → `sparse_model` (sparse — **tách env** với dense, mặc định cũng = `BAAI/bge-m3` để sau này swap linh hoạt)
  - `AU_RERANK_MODEL_NAME` → `rerank_model`
- Default: `embedding_model="BAAI/bge-m3"`, `embedding_size=1024`, `sparse_model="BAAI/bge-m3"`, `rerank_model="BAAI/bge-reranker-v2-m3"`, `rerank_provider="bge"`.
- Xóa field `generation_model`, `llm_base_url`, `llm_api_key`, `llm_model_name`, `gemini_api_key`, `model_name` (backward-compat). Chỉ giữ nếu có caller khác — grep trước.
  - `grep -rn "generation_model\|llm_base_url\|llm_api_key\|llm_model_name\|\.gemini_api_key\|\.model_name" src tests`
  - Nếu chỉ graph.py + test ref → safely xóa.
- Giữ `embedding_provider` (default `"bge"`).
- Verify: `uv run python -c "from agent.utils.config import Config; c=Config(); print(c.embedding_model, c.embedding_size, c.rerank_model, c.rerank_provider)"` (cần env hoặc .env set) → in ra mặc định BGE.

### Task 3 — Embeddings (`src/agent/utils/embeddings.py`) + Sparse từ BGE-m3 (env tách)
- Thêm `case "bge" | "fastembed":` dùng **BGE-m3**, **tách 2 instance riêng** (dense + sparse, cùng model string nhưng env tách):
  - Dense: `fastembed.TextEmbedding(model_name=cfg.embedding_model)` → wrapper `Embeddings`.
  - Sparse: `fastembed.SparseTextEmbedding(model_name=cfg.sparse_model)` → wrapper interface tương thích `langchain_qdrant.FastEmbedSparse` (return `SparseEmbedding` có `indices`+`values`).
  - **Không** chia sẻ instance dense+sparse — mỗi `TextEmbedding` / `SparseTextEmbedding` instantiate riêng. Nếu sau user set `AU_SPARSE_MODEL_NAME` khác (e.g. `Qdrant/bm25`) thì chỉ đổi một instance.
  - Nếu fastembed không support BGE-m3 sparse → fallback `FlagEmbedding.BGEM3FlagModel` (thêm dep `FlagEmbedding`).
- Xuất 2 helper đọc env riêng:
  - `get_embedding_model(cfg)` → dense instance (đọc `cfg.embedding_model`).
  - `get_sparse_embedding(cfg)` → sparse instance (đọc `cfg.sparse_model`).
- **Bỏ BM25**: trong `vdb.py`, gỡ `sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")` → thay bằng `get_sparse_embedding(Config())`.
- Bỏ `case "google"` (xem §10.c trong plan — mặc định **bỏ**).
- Giữ `case "openai" | "openai-compatible" | "custom"` để phục vụ BGE serve qua API nếu chọn (dense-only; sparse từ local `cfg.sparse_model`).
- Verify: smoke import `from agent.utils.embeddings import get_embedding_model, get_sparse_embedding` + instantiate không raise.

### Task 3b — vdb.py + retriever.py (sparse đổi tên)
- `vdb.py`:
  - Đổi module-level `sparse_embeddings` sang instance BGE-m3 sparse (từ `embeddings.get_sparse_embedding(Config())`).
  - `generate_collection`: `client.set_sparse_model(embedding_model_name=cfg.sparse_model)` (KHÔNG hardcode `"BAAI/bge-m3"` — đọc env `AU_SPARSE_MODEL_NAME`); `sparse_vectors_config` dùng vector name `"bge-m3-sparse"`.
  - `QdrantVectorStore` init trong `init_vdb`: `sparse_vector_name="bge-m3-sparse"`.
- `retriever.py`:
  - `_get_cached_vector_store`: `sparse_vector_name="bge-m3-sparse"`, `sparse_embedding=_get_cached_sparse()` — **cache key tách**: dense cache `(provider, embedding_model)`, sparse cache `(provider, sparse_model)`. Tách helper `_get_cached_sparse()` riêng khỏi `_get_cached_embedding()`.
- **Breaking**: collection `documents` cũ có sparse `fast-sparse-bm25` → phải **xóa/recreate** (`curl -X DELETE http://localhost:6333/collections/documents` hoặc skip check tồn tại) trước khi migration lại. Note trong CHANGELOG + README.

### Task 4 — Migration (`src/agent/scripts/migrate_dump_to_qdrant.py`)
- Trong `load_migration_config()`:
  - Đọc `AU_EMBED_MODEL_NAME` (fallback `EMBEDDING_MODEL`), `AU_EMBED_DIMENSION` (mới — nếu có thì dùng, bỏ bước embed test để đo size).
  - Đọc `AU_SPARSE_MODEL_NAME` (mới — tách với dense, mặc định cũng `BAAI/bge-m3`).
  - Đọc `EMBEDDING_PROVIDER` (mới) → `"bge"` default.
  - **Bỏ đọc `EMBEDDING_RPM/TPM/RPD`** (rate limit local) hoặc giữ backward-compat nhưng default 0.
- Trong `get_dense_embedding(cfg)` + `get_sparse_embedding(cfg)` (mới):
  - If `cfg["embedding_provider"] == "bge"` → trả về BGE-m3 dense+sparse local (tái dùng wrapper shared từ `embeddings.py` để DRY — import, không copy code). Dense đọc `cfg["embedding_model"]`, sparse đọc `cfg["sparse_model"]` riêng.
  - Else (có base_url) → giữ `OpenAICompatibleEmbeddings` cho dense; sparse vẫn BGE-m3 local từ `cfg["sparse_model"]` (sparse thường không có qua OpenAI API).
- Trong `run_migration()`:
  - `dense_vector_size = cfg.get("embedding_dimension") or len(test_embedding)` (default 1024).
  - `sparse_vector_name = "bge-m3-sparse"`.
  - `ensure_collection_exists`: `set_sparse_model(cfg["sparse_model"])` (KHÔNG hardcode); tạo collection với sparse vector name mới.
  - **Bỏ sleep rate limit**: gỡ khối tính `rpm_delay`/`tpm_delay`/`rpd_delay` và `time.sleep(sleep_time)`. Để comment 1 dòng `# rate-limit disabled (local model)` để git blame rõ.
  - Log rõ model + size + sparse name.
- Verify: `uv run python -m agent.scripts.migrate_dump_to_qdrant --help` (CHỈ help — mục 7 chặn chạy thật trừ khi user approve).

### Task 5 — Reranker (`src/agent/utils/reranker.py`) + wire + thứ tự
- Thêm `rerank_with_bge`:
  - Dùng fastembed `TextCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")`. Nếu fastembed không có model này → fallback `sentence_transformers.CrossEncoder` (thêm dep `sentence-transformers` vào pyproject — ưu tiên tránh thêm dep).
- Trong `get_reranker`: thêm `case "bge"` → trả về `lambda docs, q: rerank_with_bge(docs, q, top_k, cfg_model)`.
- **Wire vào pipeline và đảm bảo thứ tự trả về theo rerank**:
  - Trong `src/agent/backend/nodes/retrieval.py` `retrieve_documents`, sau `retriever.invoke(query)`, nếu `cfg.rerank_provider != "none"`:
    ```python
    rerank_fn = get_reranker(provider=cfg.rerank_provider, top_k=cfg.rerank_top_k, model=cfg.rerank_model)
    relevant_documents = rerank_fn(relevant_documents, query)
    ```
  - **Không** sort lại list sau rerank (reranker đã trả theo score giảm dần). Thứ tự `relevant_documents` chính là thứ tự top-1→top-K rerank.
  - Trả `relevant_documents` trực tiếp trong `state["documents"]` — route `/rag` serialize theo thứ tự này. Pydantic list preserves order; chỉ verify không có `sorted(...)` ngầm.
- Test: `tests/unit_tests/test_reranker.py`:
  - `test_rerank_with_bge` (mock TextCrossEncoder).
  - `test_rerank_preserves_order` — assert output list theo thứ tự score giảm dần, không shuffle.
  - `test_get_reranker_bge` default provider.
- Verify: `uv run pytest tests/unit_tests/test_reranker.py -q` xanh.

### Task 6 — Graph refactor (`src/agent/backend/graph.py`)
- Xóa import `ChatLiteLLM`, `END` vẫn giữ.
- `Graph.__init__`: bỏ `self.llm`. Chỉ giữ `self.cfg = settings`.
- `route_to_retriever`: bỏ nhánh chat history (mặc định), trả luôn `"retriever"`. Hoặc xóa hàm, set entry point thẳng `retriever`.
- `build_graph`:
  ```python
  workflow = StateGraph(AgentState)
  workflow.add_node("retriever", functools.partial(retrieve_documents, cfg=self.cfg))
  workflow.set_entry_point("retriever")
  workflow.add_edge("retriever", END)
  return workflow.compile()
  ```
- Xóa import `generate_response_default, grade_documents, retrieve_documents_with_chat_history, rewrite_query`.
- Verify: `uv run python -c "from agent.backend.graph import Graph; g=Graph().build_graph(); print(g)"`.

### Task 7 — Node cleanup
- Xóa file: `src/agent/backend/nodes/grading.py`, `rewrite.py`, `generation.py`.
- `src/agent/backend/nodes/retrieval.py`: gỡ `retrieve_documents_with_chat_history`; gỡ `get_chat_history` ở đây (chỉ một nơi cần — generation đã xóa). Rút gọn `retrieve_documents` theo plan §2.3.
- `src/agent/backend/prompts.py`: xóa file.
- `src/agent/backend/state.py`: xóa `Grade`; `AgentState` giữ `query`, `documents`, `messages`. Bỏ `retry_count` (không còn loop) — nhưng nếu còn ref ở test thì giữ optional. Grep trước.
- Verify: `uv run ruff check src/agent/backend` không có lỗi unused import.

### Task 8 — Routes & data models
- `response_data_model.py`: thêm `RetrievedDoc`, `RetrievalResponse` (plan §2.4). Xóa `QAResponse`, `ExplainQAResponse` (grep frontend/test trước — nếu ref thì giữ deprecated alias).
- `request_data_model.py`: rename class `RAGRequest` → `RetrievalRequest` (giữ `RAGRequest` alias nếu cần compat). Hoặc chỉ update docstring. Quyết định: **giữ tên `RAGRequest`** để ít churn, chỉ cập nhật doc.
- `routes/rag.py`:
  - `question_answer`: trả `RetrievalResponse(query=..., documents=[RetrievedDoc(...) for d in chain_result["documents"]])`.
  - `question_answer_stream`: bỏ event `content`; giữ `status` + đổi `citation` → `documents`.
- Grep test ref `QAResponse`/`.answer`/`citation` để cập nhật.
- Verify: `uv run pytest tests/unit_tests/test_search_delete.py tests/unit_tests/test_rag_graph.py -q` (cập nhật assertion theo contract mới).

### Task 9 — api.py + misc
- `api.py`: update `description` (plan §2.6). Bỏ `Phoenix` instrument nếu graph không còn LLM? — Không, tracer vẫn hợp lệ cho retriever/embedding → giữ.
- `frontend/assistant.py`: chỉ đọc + note trong plan PR (không sửa trừ khi nhỏ). Thêm TODO comment trong file nếu được phép (hỏi user).
- `ConvAgentBruno/`: thêm file Bruno mới cho contract `/rag` (trả documents). Out of scope nếu nhỏ — note trong PR description.

### Task 10 — DeepEval Qwen (`tests/test_rag_deepeval_qwen.py`) — **kiểm tra tài liệu trả về**
- Copy skeleton từ `test_rag_deepeval.py`, sửa:
  - `CustomEvalLLM` → `QwenEvalLLM` (plan §4.2): dùng `openai.OpenAI(base_url, api_key)`, `extra_body={"thinking": False}`.
  - Env: `QWEN_EVAL_BASE_URL`, `QWEN_EVAL_API_KEY`, `QWEN_EVAL_MODEL`, `QWEN_EVAL_THINKING`.
  - **Bỏ rate limit RAG** (`apply_rag`/`apply_embed`) vì chạy local BGE. **Giữ `apply_eval`** cho eval LLM (vì có thể chuyển NVIDIA model khi đủ tài nguyên).
  - **Mục tiêu test đổi**: kiểm tra **tài liệu trả về** (chunk chứa dữ liệu cần thiết), không có answer LLM từ RAG.
    - `chain_result["documents"]` giờ là list chunk đã rerank. `retrieval_context = [doc.page_content for doc in chain_result["documents"]]`.
    - `actual_output = ""` (hoặc `"retrieval-only"`) — không dùng cho metric context-based.
    - **Metric dùng** (vì không có answer):
      - `ContextualPrecisionMetric` — top-K có context match không.
      - `ContextualRecallMetric` — recall context.
      - (Optional) `ContextualRelevancyMetric` nếu deepeval support.
    - **Bỏ**: `AnswerRelevancyMetric`, `FaithfulnessMetric` (cần `actual_output` = answer, không còn).
  - **Custom assertion ngoài DeepEval** — kiểm tra chunk trả về nằm đúng phần tài liệu gốc (mục tiêu chính của test):
    - Hàm `assert_chunk_locators(retrieved_docs, expected_locators)` trong test file, không phụ thuộc LLM:
      - Lấy `retrieved_docs` = `chain_result["documents"]` (đã rerank).
      - Với mỗi chunk, đọc `d.metadata` (migration đã set `document_id`, `chunk_index`, `global_id`).
      - Pass nếu **≥1** chunk retrieved có `global_id` match một trong `expected_locators[*].global_id`.
      - Nếu corpus bị re-chunk (chunk_index khác → global_id khác do hash `doc_id::chunk_idx`) → fallback kiểm tra content match: chunk retrieved chứa fragment trong `expected_context` thì vẫn pass, nhưng flag `locator_drift=true` trong log để phân tích sau.
    - Gọi `assert_chunk_locators(...)` trong test case (sau `assert_test` metric context-based). Đây là **primary assertion**; metric DeepEval context-based là secondary confirmation slot-match với LLM eval.
  - Marker `@pytest.mark.qwen` để bypass network guard.
- `golden_questions_v2.json`: **tái sử dụng câu hỏi từ `golden_questions.json` cũ** (14 câu) — không viết câu mới. Schema:
  ```json
  [
    {
      "id": 1,
      "topic": "aiot",
      "question": "...",
      "expected_context": ["..."],

      "expected_chunk_locators": [
        {"document_id": "<oid>", "chunk_index": 3, "source": "...", "heading_path": "h1>h2", "global_id": "<uuid>"}
      ]
    }
  ]
  ```
  - `expected_context` lấy nguyên từ file cũ (14 câu).
  - `expected_chunk_locators` **không điền tay** — do bước locator (Task 10a) điền自動 sau khi migration + index xong. Ban đầu file có field rỗng `[]`, sau khi chạy locator sẽ điền đầy đủ.
  - **Bỏ** `expected_answer` (nếu user muốn giữ làm reference thì để optional, metric không dùng).
- Verify: `uv run pytest tests/test_rag_deepeval_qwen.py -m qwen --collect-only` (không gọi LLM thực) phải collect thành công.

### Task 10a — Locator script (`tests/locate_expected_chunks.py`)
Mục đích: điền `expected_chunk_locators` vào `golden_questions_v2.json` bằng cách quét corpus đã index, tìm chunk nào chứa fragment trong `expected_context`.
- Script dev (không phải test auto), chạy **một lần sau migration** (Task 4 xong) và **trước khi chạy deepeval** (Task 10/13). User approve chạy (plan §10.h).
- Logic:
  1. Load `golden_questions_v2.json` (lúc này `expected_chunk_locators` có thể rỗng).
  2. Kết nối Qdrant (`qdrant_client` từ `agent.utils.vdb`), `scroll` toàn bộ points collection `documents`.
  3. Với mỗi câu hỏi, với mỗi fragment trong `expected_context`:
     - Kiểm tra point nào có `page_content` chứa fragment (substring match — case-sensitive, có thể lower-case cả 2 nếu cần).
     - Nếu match → ghi lại `{document_id, chunk_index, source, heading_path, global_id}` (lấy từ payload metadata đã set trong migration `build_payload` — đã có `document_id`, `chunk_index`, `global_id`). `heading_path` xây từ metadata `h1`/`h2`/`h3` (Markdown header từ chunking) — concatenate "h1>h2>h3".
  4. Ghi ngược lại `golden_questions_v2.json` field `expected_chunk_locators` (đảm bảo stable JSON, indent 2, ensure_ascii=False).
  5. Mỗi câu hỏi có thể match nhiều chunk (chunk overlap hoặc info lặp) → lưu hết list. Test sau kiểm tra **≥1** match.
- **Anchor document**: chọn 1 document thật làm reference (e.g. ATTT curriculum doc) — log ra riêng để user verify bộ golden khớp tài liệu thật, không phải đoán.
- Không tốn quota eval LLM (chỉ quét Qdrant local). Mark script `@pytest.mark.skip` nếu chạy qua pytest, hoặc để ngoài `tests/` runner (`python tests/locate_expected_chunks.py`).
- Verify: chạy script, mở `golden_questions_v2.json` kiểm tra `expected_chunk_locators` đã điền (≠ `[]`) cho tất cả 14 câu.

### Task 11 — conftest
- `tests/conftest.py`: thêm marker `qwen` vào `block_external_http` bypass (plan §4.5). Đăng ký marker trong `pyproject.toml` `[tool.pytest.ini_options].markers`.
- Verify: `uv run pytest --markers | findstr qwen`.

### Task 12 — Env, deps, docs
- `template.env`: cập nhật (plan §5). Bỏ `GEMINI_API_KEY`, `GENERATION_MODEL`, `LLM_*`, `EVAL_LLM_MODEL` (Gemini), `RAG_GEN_*`. Thêm `AU_*`, `QWEN_EVAL_*`.
- `pyproject.toml`: review deps (plan §6). Mark pyproject markers `[tool.pytest]` update. Run `uv lock`.
- `README.md`: thêm section "Retrieval-only mode (v6)" mô tả contract mới + BGE + Qwen eval. Cập nhật "Quickstart".
- `CHANGELOG.md`:entry `[NEW] v6.0.0 — breaking: /rag returns documents only, BGE-m3 embed+rerank, Qwen eval`.

### Task 13 — Lint/type/test tổng
```powershell
uv sync
uv run ruff check .
uv run ruff format .
uv run ruff format --check .
uv run pytest tests/unit_tests -q
# Nếu có Qdrant up + env thật:
$env:ALLOW_NETWORK_TESTS="1"
uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
```
- Fix tới khi xanh. Không nắn test để pass (chỉ khi test cũ exit vì contract break → cập nhật assertion theo contract mới + commit reason trong message PR).

### Task 14 — Dọn file legacy (plan §11)
Vì nhánh mới tinh, xóa triệt để file không phục vụ pipeline retrieval-only BGE. **Thứ tự**: làm sau các task refactor (6/7/8/10) để tránh break mid-way.

**Source backend — `git rm`:**
- `src/agent/backend/nodes/generation.py`
- `src/agent/backend/nodes/grading.py`
- `src/agent/backend/nodes/rewrite.py`
- `src/agent/backend/prompts.py`
- `src/agent/scripts/visualize_graph.py`
- `src/agent/scripts/generate_diagrams.py`
- `src/agent/scripts/resources/` (toàn bộ png: gemini/cohere/openai/langchain/streamlit/Architecture)
- `graph.png`

**Source backend — refactor trong file:**
- `src/agent/backend/state.py`: xóa class `Grade`.
- `src/agent/data_model/internal_model.py`: xóa `RetrievalResults` nếu `convert_qdrant_result_to_retrieval_results` không caller ngoài test (grep trước: `grep -rn "RetrievalResults\|convert_qdrant_result\|load_prompt_template\|combine_text_from_list\|format_docs_for_citations" src tests`).
- `src/agent/utils/utility.py`: xóa `load_prompt_template`, `convert_qdrant_result_to_retrieval_results` (và `RetrievalResults` import). Giữ `format_docs_for_citations` nếu còn dùng ở trace/dedupe; nếu chỉ generation dùng → xóa. `combine_text_from_list` giữ nếu có caller.
- `src/agent/data_model/response_data_model.py`: xóa `QAResponse`, `ExplainQAResponse`.
- `src/agent/backend/services/embedding_management.py`: gỡ `if __name__ == "__main__"` block legacy "Was ist Attention?".

**Test + VCR — `git rm`:**
- `tests/test_rag_deepeval.py` (Gemini legacy — thay bằng Qwen version).
- `tests/golden_questions.json` (thay bằng `golden_questions_v2.json`).
- `tests/vcr/test_litellm_requests.py`
- `tests/vcr/cassettes/test_litellm_requests/test_litellm_gemini_chat_vcr.yaml`
- `tests/vcr/test_embedding_and_reranker_requests.py`
- `tests/vcr/cassettes/test_embedding_and_reranker_requests/test_cohere_reranker_vcr.yaml`
- `tests/vcr/cassettes/test_embedding_and_reranker_requests/test_langchain_cohere_embedding_vcr.yaml`
- `tests/resources/1706.03762v5.pdf` `tests/resources/1912.01703v1.pdf` `tests/resources/albert.txt`

**Test — cập nhật (không xóa):**
- `tests/unit_tests/test_utility.py`: bỏ test `test_load_prompt_template*`, `test_convert_qdrant_result_to_retrieval_results` (không còn source). Giữ test `create_tmp_folder`, `format_docs_for_citations` (nếu còn), `combine_text_from_list`.
- `tests/unit_tests/test_rag_graph.py`: assert documents (không answer).
- `tests/unit_tests/test_reranker.py`: thêm test bge + order, bỏ test cohere/flashrank nếu xóa provider.
- `tests/unit_tests/test_embedding_management.py`: assert dim 1024, BGE-m3.
- `tests/unit_tests/test_vdb.py`: sparse name `bge-m3-sparse`, model `BAAI/bge-m3`.
- `tests/test_stream.py`: bỏ assert event `content`, thêm assert event `documents`.
- `tests/test_integration.py`: giữ.
- `tests/e2e_tests/test_api.py`: giữ.
- `tests/vcr/test_contracts.py`: giữ.
- `tests/fakes/rag.py`: giữ.

**Resources / docs — `git rm`:**
- `resources/1706.03762v5.pdf` `resources/1912.01703v1.pdf` `resources/My_Neighbor_Totoro.pdf` `resources/albert.txt`
- `resources/Architecture.png` `resources/search_flow.png` `resources/cohere.png` `resources/ollama.png` `resources/openai.png` `resources/ui.png` `resources/tracing.png` `resources/research.png`
- (Nếu §10.g user muốn giữ `resources/` cho demo thì skip — mặc định xóa.)

**Docs — cập nhật (không xóa):**
- `README.md`: viết lại (Architecture, Quickstart, LLM section, Reranking).
- `CHANGELOG.md`: thêm entry v6.0.0.
- `ConvAgentBruno/RAG/Chat.bru` `ConvAgentBruno/RAG/Stream.bru`: cập nhật expected body (không `answer`, có `documents`).
- `Makefile`: review `test-e2e` (cần Qwen env) — giữ cấu trúc.
- `docker-compose.yml`: gỡ env Gemini/Cohere, thêm `AU_*` nếu cần.
- `.github/workflows/test.yml`: gỡ env dummy Gemini/Cohere; CI không chạy deepeval (local only).

**Giữ nguyên (không xóa):**
- `.devcontainer/` `.idx/` `Dockerfile` `Dockerfile.frontend` `.dockerignore` `.pre-commit-config.yaml` `.markdown-link-check.json` `.python-version`
- `config/qdrant.yaml` `Makefile` `ruff.toml` `pyproject.toml` `uv.lock` `template.env`
- `frontend/` (xử lý riêng theo §10.e)
- `src/agent/scripts/chunking.py` `dump_reader.py` `migrate_dump_to_qdrant.py` `load_dummy_data.py` (migration pipeline)
- `tests/conftest.py` `tests/fakes/` `tests/vcr/test_contracts.py` `tests/e2e_tests/` `tests/test_integration.py`

**Verify sau cleanup:**
- `git status` review staged deletes.
- `uv run ruff check .` không có lỗi "import not found".
- `uv run pytest tests/unit_tests -q` pass.
- `grep -rn "Generation\|grading\|rewrite_query\|QAResponse\|gemini.*embed\|FastEmbedSparse.*bm25\|load_prompt_template" src tests` → không có match (legacy gone).

## 3. Cách verify từng bước (cheat sheet)

| Task | Verify cmd | Kỳ vọng |
|------|------------|---------|
| 2 | `uv run python -c "from agent.utils.config import Config; print(Config().embedding_size, Config().sparse_model)"` | `1024 BAAI/bge-m3` |
| 3 | import + instant `get_embedding_model` + `get_sparse_embedding` (Config()) không raise | OK |
| 3b | `grep -n bge-m3-sparse src/agent/utils/vdb.py src/agent/utils/retriever.py` | match; không còn `Qdrant/bm25` |
| 4 | `uv run python -m agent.scripts.migrate_dump_to_qdrant --help` | in usage, không sleep rate |
| 5 | `uv run pytest tests/unit_tests/test_reranker.py -q` | pass + có test order |
| 6 | `uv run python -c "from agent.backend.graph import Graph; Graph().build_graph()"` | no error |
| 7 | `uv run ruff check src/agent/backend` | clean |
| 8 | `uv run pytest tests/unit_tests/test_rag_graph.py -q` | pass (no `answer` field) |
| 10 | `uv run pytest tests/test_rag_deepeval_qwen.py -m qwen --collect-only` | collect OK, metric context |
| 10a | chạy `python tests/locate_expected_chunks.py` → mở `golden_questions_v2.json` | `expected_chunk_locators` ≠ `[]` |
| 12 | `uv lock && uv sync` | no error |
| 13 | `uv run ruff check . && uv run pytest tests/unit_tests -q` | clean + pass |
| 14 | `grep -rn "generation\|grading\|rewrite_query\|QAResponse\|FastEmbedSparse.*bm25" src tests` | no match |

## 4. Quy tắc PR / commit

- **Không commit** trừ khi user yêu cầu rõ.
- Khi được yêu cầu: stage riêng từng file logic, message theo Conventional Commits (repo dùng commitizen — `feat(scope): ...`, `refactor(scope): ...`, `test(scope): ...`).
- Không `--no-verify`, không `--force`, không skip pre-commit (`prek`).
- Mỗi commit ≤ ~400 dòng diff nếu có thể tách.

## 5. Khi bị block

- Nếu gặp điểm chờ user confirm (plan §10) → dùng `question` tool, không đoán mò cho a/b/c/d/e.
- Nếu test đỏ do dep thiếu (sentence-transformers) → ưu tiên giải pháp trong dep hiện có (fastembed) trước khi thêm dep.
- Nếu frontend/VCR test break do contract change → cập nhật; không rollback contract.

## 6. Self-check trước khi báo "xong"

- [ ] `ruff check .` passed
- [ ] `ruff format --check .` passed
- [ ] `pytest tests/unit_tests` passed
- [ ] Graph build thành công, không import LLM
- [ ] `/rag` route trả `RetrievalResponse` (không có field `answer`), documents theo thứ tự rerank
- [ ] Default env dùng BGE-m3 / dim 1024 / reranker v2-m3
- [ ] Sparse dùng BGE-m3 (tên `bge-m3-sparse`), không còn `Qdrant/bm25`
- [ ] Dense + sparse **env tách** (`AU_EMBED_MODEL_NAME`, `AU_SPARSE_MODEL_NAME`) — mặc định cùng `BAAI/bge-m3` nhưng 2 instance/cached riêng
- [ ] Rate limit RAG đã bỏ (embed/gen), eval LLM giữ `apply_eval`
- [ ] `test_rag_deepeval_qwen.py` collect được, marker `qwen` đăng ký, metric chỉ dùng context-based + custom `assert_chunk_locators`
- [ ] `golden_questions_v2.json` tái sử dụng 14 câu hỏi cũ, có `expected_chunk_locators` điền bởi locator script (Task 10a)
- [ ] Migration script: không còn sleep rate-limit, dense+sparse từ BGE-m3 (env tách)
- [ ] `template.env` phản ánh config mới, không còn `GEMINI_API_KEY`/`GENERATION_MODEL`/`RAG_GEN_*`
- [ ] **Legacy file đã xóa** (nodes gen/grading/rewrite, prompts.py, visualize_graph, generate_diagrams, VCR Gemini/Cohere, `golden_questions.json`, resources legacy) — `grep` không còn match
- [ ] Test VCR legacy đã xóa, test contract/unit đã cập nhật theo contract mới
- [ ] `plan.md` mục §10 (điểm chờ a-g) được làm rõ trước khi close phiên

## 7. Ràng buộc "KHÔNG TỰ CHẠY migration/eval thật"

Kế thừa quy tắc đã tồn tại trong `migrate_dump_to_qdrant.py` (dòng 16): 
> "CHỈ VIẾT CODE — KHÔNG TỰ CHẠY"

- **Không tự ý chạy** `python -m agent.scripts.migrate_dump_to_qdrant`, `uv run pytest tests/test_rag_deepeval_qwen.py` (gọi Qwen thật), `docker compose up` để rồi tiêu thụ quota/ 网络.
- Chỉ chạy khi user yêu cầu rõ "chạy thử migration", "chạy test eval Qwen đi".
- **Ngoại lệ — được phép chạy**: `python tests/locate_expected_chunks.py` (Task 10a) chỉ quét Qdrant local, không gọi LLM, không tốn eval quota — nhưng vẫn cần user approve 1 lần để đảm bảo migration đã xong (plan §10.h).
- Verify tĩnh (`--help`, `--collect-only`, import smoke) vẫn OK.
