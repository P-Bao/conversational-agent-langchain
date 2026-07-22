# Plan — Refactor sang "Retrieval-only API" + BGE models + DeepEval (Qwen tự host)

>Phiên bàn thực hiện: phiên kế tiếp (agent tự làm theo `agent.md`).
>Repo: `conversational-agent-langchain`.
>Nhánh gốc (khi lập plan): `feature/gemini-deepeval-integration`.
>Nhánh làm việc (đã tạo + đang refactor): `feature/retrieval-only-bge-qwen`.
>Ngày lập: 2026-07-22. Tiến trình thực hiện: xem §A ở đầu file.

## A. TIẾN TRÌNH THỰC HIỆN (bản cập nhật)

> Cập nhật lần cuối: 2026-07-22 (hoàn thành 100% tất cả 16 task).
> Nhánh làm việc: `feature/retrieval-only-bge-qwen`.

### A.1 Trạng thái task

| Task | Trạng thái | Ghi chú |
|------|-----------|---------|
| 1 — Tạo nhánh | ✅ DONE | Đã tạo nhánh `feature/retrieval-only-bge-qwen`. |
| 2 — Config | ✅ DONE | `src/agent/utils/config.py` đã rewrite: alias `AU_EMBED_MODEL_NAME`, `AU_EMBED_DIMENSION`, `AU_SPARSE_MODEL_NAME`, `AU_RERANK_MODEL_NAME`, `sparse_vector_name="bge-m3-sparse"`. |
| 3 — embeddings.py | ✅ DONE | `BGE3Embeddings` (dense 1024-dim) + `BGE3SparseEmbeddings` (sparse `SparseVector(indices, values)`) dùng `FlagEmbedding.BGEM3FlagModel`. Đã fix CPU FP16 hang bằng `use_fp16 = torch.cuda.is_available()`. |
| 3b — VDB & Retriever | ✅ DONE | `vdb.py` & `retriever.py` cập nhật dùng BGE-m3 sparse embeddings & `"bge-m3-sparse"` vector name. |
| 4 — Migration Script | ✅ DONE | `src/agent/scripts/migrate_dump_to_qdrant.py` dùng BGE-m3 models, `"bge-m3-sparse"`, và bỏ rate-limit delays cho local execution. |
| 5 — Reranker | ✅ DONE | `src/agent/utils/reranker.py` bổ sung provider `bge` dùng `FlagEmbedding.FlagReranker` (`BAAI/bge-reranker-v2-m3`). Wired trực tiếp vào `retrieve_documents`. |
| 6 & 7 — Graph & Nodes Cleanup | ✅ DONE | `src/agent/backend/graph.py` đơn giản hóa thành pipeline `entry -> retriever -> END`. Loại bỏ `ChatLiteLLM`, `generate_response_default`, `grade_documents`, `rewrite_query`. |
| 8 — Data Models & Routes | ✅ DONE | `src/agent/data_model/response_data_model.py` thêm `RetrievedDoc` & `RetrievalResponse`. `/rag/` và `/rag/stream` trong `rag.py` cập nhật trả về `RetrievalResponse` (không LLM answer). |
| 9 — API & OpenAPI Schema | ✅ DONE | `src/agent/api.py` cập nhật OpenAPI schema version 6.0.0 retrieval-only. |
| 10 & 10a — DeepEval & Golden Dataset | ✅ DONE | Tạo `tests/golden_questions_v2.json`, `tests/locate_expected_chunks.py`, và `tests/test_rag_deepeval_qwen.py` dùng `QwenEvalLLM`. |
| 11 & 12 — Conftest, Pyproject & Env | ✅ DONE | `pyproject.toml` (version 6.0.0, registered `qwen` marker), `template.env`, `conftest.py` đã cập nhật. |
| 13 — Unit Tests Verification | ✅ DONE | Đã chạy `uv run pytest tests/unit_tests -q` → **41/41 unit tests PASSED** (`41 passed, 6 warnings in 62.78s`). |
| 14 — Legacy Cleanup | ✅ DONE | Đã xóa legacy files (`generation.py`, `grading.py`, `rewrite.py`, `prompts.py`, `visualize_graph.py`, `generate_diagrams.py`, `graph.png`, `test_rag_deepeval.py`, `golden_questions.json`, legacy VCR files, `load_prompt_template`, `convert_qdrant_result_to_retrieval_results`, `Grade` model). |
| 15 — Docs & Changelog | ✅ DONE | Update `README.md` & `CHANGELOG.md` cho v6.0.0. |
| 16 — GitNexus Verification | ✅ DONE | Chạy `gitnexus detect-changes -r conversational-agent-langchain` kiểm tra blast radius và affected processes. |

### A.2 Chướng ngại & Giải pháp đã thực hiện

1. **FastEmbed không hỗ trợ BGE-m3**:
   - Đã chuyển sang **`FlagEmbedding`** (`BGEM3FlagModel` và `FlagReranker`).
2. **CPU FP16 Hang**:
   - `BGEM3FlagModel` và `FlagReranker` trên CPU bị đơ khi `use_fp16=True`. Đã tự động chọn `use_fp16 = torch.cuda.is_available()`.
3. **Thứ tự Document sau Rerank**:
   - Đã đảm bảo document trả về được sắp xếp theo điểm số rerank giảm dần và bảo toàn thứ tự nguyên vẹn khi serialize trong endpoint API `/rag`.

---



## 0. Tổng quan mục tiêu

1. **Tạo nhánh mới** từ `feature/gemini-deepeval-integration` (hoặc từ `main` nếu cần base sạch — quyết định ở §1).
2. **Loại bỏ LLM generation" ở cuối**: hệ thống chỉ còn là một **API trả về các tài liệu (context)** phục vụ RAG downstream. Xóa node `response_synthesizer`, bỏ grading/rewrite dùng LLM (nếu giữ lại grading thì phải chuyển sang dạng không-LLM hoặc bỏ hẳn — mặc định **bỏ hẳn** graph LLM).
3. **Đổi embedding + migration + rerank** sang self-host-compatible, dùng **một model BGE-m3 cho cả dense + sparse**:
   - `AU_EMBED_MODEL_NAME=BAAI/bge-m3` (dim 1024)
   - `AU_EMBED_DIMENSION=1024`
   - `AU_RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3`
   - Áp dụng cho cả: service runtime (`src/agent/utils/embeddings.py`), migration (`src/agent/scripts/migrate_dump_to_qdrant.py`), reranker (`src/agent/utils/reranker.py`).
   - BGE-m3 cung cấp 3 chế độ (dense, sparse, colbert) → tận dụng cho hybrid: **dense + sparse từ cùng một forward pass** (FastEmbed hoặc FlagEmbedding), bỏ BM25 FastEmbed `Qdrant/bm25`. Kết quả documents trả về phải **giữ thứ tự sau rerank** (reranker là bước cuối trước khi trả về client).
4. **Bỏ rate limit cho RAG system** (embed + gen — giờ chỉ embed) vì chạy local model. **Giữ rate limit cho eval LLM** (DeepEval) vì có thể chuyển sang NVIDIA model khi đủ tài nguyên.
5. **Tạo một phiên bản test khác** bằng DeepEval với **bộ câu hỏi mới**, dùng **model Qwen tự host** (OpenAI-compatible) làm eval LLM. Mục tiêu test đổi: kiểm tra **tài liệu trả về đúng** (chunk chứa dữ liệu cần thiết) chứ không kiểm tra answer.
   ```
   POST http://A.B.C.D:E/v1/chat/completions
   Authorization: Bearer <token>
   body: { messages, stream:false, thinking:false }
   ```

## 1. Quyết định nhánh & base

- **Mặc định**: tạo nhánh `feature/retrieval-only-bge-qwen` từ `feature/gemini-deepeval-integration` (giữ scripts migration/chunking đã có).
- Lý do giữ base này: đã có `migrate_dump_to_qdrant.py`, `chunking.py`, `dump_reader.py`, `golden_questions.json`, bộ test deepeval. Refactor trên đó nhanh hơn.
- Nếu phát hiện conflict lớn với history Gemini-only → fallback nhánh từ `main` + cherry-pick scripts.

Thao tác:
```powershell
git checkout feature/gemini-deepeval-integration
git pull
git checkout -b feature/retrieval-only-bge-qwen
```

## 2. Loại bỏ LLM generation ở cuối — biến hệ thống thành "retrieval-only API"

### 2.1. Mục tiêu API mới
- `/rag` và `/rag/stream` hiện trả `answer` (LLM sinh) + `meta_data` (documents). Sau refactor:
  - **Bỏ `answer`**, trả **chỉ documents** (context) + metadata.
  - Đổi `QAResponse` → `RetrievalResponse` (xem §2.4).
  - `/rag/stream`: bỏ sự kiện `content` (sinh text), chỉ giữ `status` + `citation` (tên field đặt lại `documents` cho rõ ràng).

### 2.2. Sửa `src/agent/backend/graph.py`
- Bỏ `ChatLiteLLM` (không còn LLM).
- Bỏ nodes `rewrite_query`, `response_synthesizer`.
- Bỏ `grade_documents` (cần LLM). Quyết định: **bỏ grading** — trả thẳng documents retriever. Lý do: hệ thống giờ là "retrieval-only"; downstream LLM sẽ tự quyết relevance. (Ghi chú A/B: nếu sau cần giữ simple dedup/length filter không-LLM thì thêm vào node `post_filter` — không nằm trong scope lần này.)
- Graph mới đơn giản:
  ```
  entry -> route_to_retriever -> [retriever | retriever_with_chat_history] -> END
  ```
  - `retriever_with_chat_history` hiện dùng LLM để "condense question". Vì không còn LLM → **bỏ luôn nhánh chat history**; chỉ giữ `retriever`. Nếu muốn giữ multi-turn, có thể thay condenser bằng heuristic (ghép last user message + previous answer) — **out of scope**, mặc định chỉ hỗ trợ single-turn.
- `Graph.__init__` không nhận/tạo LLM nữa. `build_graph()` trả graph không cần LLM.

### 2.3. Sửa nodes
- `src/agent/backend/nodes/retrieval.py`: giữ `retrieve_documents` (bỏ phần retry_count/rewrite vì không còn loop). Gỡ `retrieve_documents_with_chat_history`.
- `src/agent/backend/nodes/grading.py`: **xóa file** (hoặc giữ nhưng unused — khuyến nghị xóa để ruff/mypy sạch).
- `src/agent/backend/nodes/rewrite.py`: **xóa file**.
- `src/agent/backend/nodes/generation.py`: **xóa file**.
- `src/agent/backend/prompts.py`: giữ `RESPONSE_TEMPLATE`? → **xóa toàn file** vì không còn sinh answer. (Nếu downstream test vẫn ref prompt → bỏ ref.)
- `src/agent/backend/state.py`: `AgentState` giữ `query`, `documents`, `messages`, `retry_count`(optional xóa). `Grade` model → **xóa**.

### 2.4. Data models — `src/agent/data_model/`
- `response_data_model.py`:
  - Thêm `RetrievalResponse`:
    ```python
    class RetrievedDoc(BaseModel):
        text: str
        page: int | None = None
        source: str | None = None
        score: float | None = None
        metadata: dict = {}
    class RetrievalResponse(BaseModel):
        query: str
        documents: list[RetrievedDoc]
    ```
  - Có thể giữ `SearchResponse`, `EmbeddingResponse`. Xóa `QAResponse`, `ExplainQAResponse`.
- `request_data_model.py`: `RAGRequest` → đổi tên/đặt alias `RetrievalRequest` (giữ `messages`, `collection_name`). Giữ `SearchParams`, `EmbeddTextRequest`.

### 2.5. Routes — `src/agent/routes/`
- `rag.py`:
  - `POST /rag/` → trả `RetrievalResponse` (chỉ documents).
  - `POST /rag/stream` → chỉ stream `status` + `documents`.
  - Import `Graph` mới (không LLM).
- `search.py`: giữ như cũ (đã là retrieval-only).
- `collection.py`, `delete.py`, `embeddings.py`: không đổi nội dung, chỉ kiểm tra import sau khi xóa nodes.

### 2.6. `api.py`
- Giữ include routers. Update `description` trong `my_schema`: "Retrieval-only API: returns relevant documents for downstream LLM."
- Phoenix tracing: giữ (instrument vẫn hợp lệ vì retriever/embed dùng langchain).

### 2.7. Frontend (`frontend/assistant.py`)
- Đọc file, nếu nó gọi `/rag` rồi hiển thị `answer` → cần điều hướng người dùng (frontend giờ sẽ tự gọi downstream LLM). **Scope**: không sửa frontend trong PR này, chỉ note trong README/CODEOWNERS rằng frontend cần cập nhật riêng. (Nếu nhỏ thì sửa luôn.)

## 3. Đổi embedding/migration/rerank sang BGE (dense + sparse từ cùng một model)

### 3.1. Naming convention — dùng prefix `AU_`
Yêu cầu user: biến môi trường dạng `AU_EMBED_MODEL_NAME`, `AU_EMBED_DIMENSION`, `AU_RERANK_MODEL_NAME`.
→ Trong `Config` (`src/agent/utils/config.py`) thêm alias:

```python
# Dense (BGE-m3)
embedding_model: str = Field(
    default="BAAI/bge-m3",
    validation_alias=AliasChoices(
        "embedding_model", "EMBEDDING_MODEL",
        "AU_EMBED_MODEL_NAME", "AU_EMBED_MODEL",
    ),
)
embedding_size: int = Field(
    default=1024,
    validation_alias=AliasChoices(
        "embedding_size", "EMBEDDING_SIZE",
        "AU_EMBED_DIMENSION",
    ),
)
# Sparse (BGE-m3 — cùng model nhưng env tách để linh hoạt swap sau)
sparse_model: str = Field(
    default="BAAI/bge-m3",
    validation_alias=AliasChoices(
        "sparse_model", "SPARSE_MODEL",
        "AU_SPARSE_MODEL_NAME", "AU_SPARSE_MODEL",
    ),
)
# Reranker (BGE-reranker v2-m3)
rerank_model: str = Field(
    default="BAAI/bge-reranker-v2-m3",
    validation_alias=AliasChoices(
        "rerank_model", "RERANK_MODEL",
        "AU_RERANK_MODEL_NAME", "AU_RERANK_MODEL",
    ),
)
```

> Lưu ý: mặc định `AU_EMBED_MODEL_NAME` == `AU_SPARSE_MODEL_NAME` == `BAAI/bge-m3` (cùng model cho dense+sparse), nhưng **env tách biệt** để sau này swap sparse sang `Qdrant/bm25` hoặc dense-only endpoint mà không đổi dense, và ngược lại. Wrapper (§3.2) cũng **chia 2 instance** riêng (`TextEmbedding` cho dense, `SparseTextEmbedding` cho sparse, cùng model string), không instantiate chung để chuẩn bị cho khả năng swap.

Bỏ mặc định Gemini:
- `embedding_provider` default → `"bge"` (hoặc `"huggingface"`/`"fastembed"` — quyết định ở §3.2).
- `generation_model`, `llm_base_url`, `llm_api_key`, `llm_model_name` → **xóa** (không còn LLM). Giữ `gemini_api_key`? → không cần; bỏ luôn hoặc giữ làm dead var (khuyến nghị bỏ).
- Backward-compat aliases (`model_name`, `embedding_model_name`) → update default sang `BAAI/bge-m3`.

### 3.2. Embedding provider — `src/agent/utils/embeddings.py`
- Thêm `case "bge" | "fastembed":` dùng **BGE-m3 multi-function**, **tách 2 instance riêng** (dense + sparse, cùng model string nhưng env tách):
  - Dense: `fastembed.TextEmbedding(model_name=cfg.embedding_model)` → wrapper `Embeddings`.
  - Sparse: `fastembed.SparseTextEmbedding(model_name=cfg.sparse_model)` → wrapper interface tương thích `langchain_qdrant.FastEmbedSparse` (return `SparseEmbedding` có `indices`+`values`).
  - **Không** chia sẻ instance giữa dense + sparse (mỗi `TextEmbedding` / `SparseTextEmbedding` instantiate riêng) → nếu sau user set `AU_SPARSE_MODEL_NAME` khác (e.g. BM25) thì chỉ đổi một instance.
  - Nếu fastembed không support BGE-m3 sparse → fallback `FlagEmbedding.BGEM3FlagModel` (thêm dep `FlagEmbedding` vào pyproject — balance: tránh thêm torch nếu có thể, ưu tiên fastembed).
- Xuất 2 helper: `get_embedding_model(cfg)` (dense) + `get_sparse_embedding(cfg)` (sparse), đọc env riêng.
- **Bỏ `FastEmbedSparse("Qdrant/bm25")`** trong `vdb.py` và `retriever.py` — thay bằng instance BGE-m3 sparse từ `get_sparse_embedding`.
- Bỏ `case "google"` (xem §10.c trong plan — mặc định **bỏ**).
- Giữ `case "openai" | "openai-compatible" | "custom"` để phục vụ BGE serve qua API nếu chọn (dense-only; sparse lúc đó vẫn dùng local BGE-m3 từ `cfg.sparse_model`).
- Caching trong `retriever.py`: cache key tách riêng — dense cache `(provider, embedding_model)`, sparse cache `(provider, sparse_model)`.

### 3.3. Qdrant collection schema — sparse vector đổi tên
- `vdb.py` `generate_collection` / `generate_collection_async`:
  - `sparse_vector_name` đổi từ `"fast-sparse-bm25"` → `"bge-m3-sparse"` (hoặc giữ variable name nhưng model đổi — quyết định: **đổi luôn tên cho rõ**) → phải tạo collection mới hoặc migration lại (collection cũ BM25 incompatible). Note breaking.
- `QdrantVectorStore` init (`vdb.py`, `retriever.py`):
  - `sparse_embedding=sparse_embeddings` (instance BGE-m3 sparse)
  - `sparse_vector_name="bge-m3-sparse"`
- `migrate_dump_to_qdrant.py` đồng bộ tên sparse.

### 3.4. Reranker — `src/agent/utils/reranker.py`
- Reranker hiện không được graph gọi (grep xác nhận `get_reranker` không có caller trong `src`). Yêu cầu nêu rõ phải đổi rerank → wire vào pipeline.
- Thêm `case "bge":` dùng model `BAAI/bge-reranker-v2-m3`:
  ```python
  def rerank_with_bge(documents, query, top_k, model_name="BAAI/bge-reranker-v2-m3"):
      from fastembed import TextCrossEncoder  # fastembed hỗ trợ cross-encoder
      ranker = TextCrossEncoder(model_name=model_name)
      pairs = [(query, d.page_content) for d in documents]
      scores = ranker.rank(pairs)
      ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
      return [d for d, _ in ranked]
  ```
  (Nếu fastembed thiếu model → fallback `sentence-transformers.CrossEncoder`.)
- Wire reranker vào pipeline và **đảm bảo thứ tự trả về theo rerank**:
  - Trong `retrieve_documents` (retrieval node):
    1. `docs = retriever.invoke(query)` (hybrid fusion BGE-m3 dense+sparse)
    2. Nếu `cfg.rerank_provider != "none"`: `docs = rerank_fn(docs, query)`
    3. Trả `docs` đã rerank — **đây là thứ tự cuối cùng**, không sort lại theo score Qdrant.
  - **Đảm bảo** route `/rag` và `/rag/stream` preserve thứ tự list khi serialize (Pydantic list giữ thứ tự — OK; chỉ verify không có `sorted(...)` ngầm).
- `Config.rerank_provider` default → `"bge"` (sẵn sàng), `rerank_top_k=5`.
- Bỏ `case "cohere"` và `case "flashrank"`? → Khuyến nghị **giữ** flashrank (có test), **bỏ** cohere nếu muốn lean. Cân nhắc giữ cohere vì có VCR test — quyết định: giữ cả để không phá test, chỉ set default sang `bge`.

### 3.5. tests/unit_tests/test_reranker.py
- Thêm test `test_rerank_with_bge` (mock `TextCrossEncoder`), `test_get_reranker_bge`, và `test_rerank_preserves_order` (assert output list theo thứ tự score giảm dần).
- Cập nhật default `rerank_provider` ở các test cũ nếu chúng assert default `"none"`.

### 3.6. Bỏ rate limit (RAG system)
- `tests/test_rag_deepeval*.py`: gỡ `UnifiedRateLimiter.apply_rag`/`apply_embed` (gọi local model, không quota). **Chỉ giữ `apply_eval`** cho eval LLM (vì có thể chuyển NVIDIA model khi đủ tài nguyên).
- Migration script `migrate_dump_to_qdrant.py`: bocj phần tính `rpm_delay`/`tpm_delay`/`rpd_delay` sleep (chạy local). Để flag `EMBEDDING_RPM` env vẫn đọc được (backward-compat) nhưng nếu =0 hoặc unset → không sleep.

### 3.7. Migration — `src/agent/scripts/migrate_dump_to_qdrant.py`
- Đang đọc `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY` trực tiếp. Refactor:
  - Đọc `AU_EMBED_MODEL_NAME` (fallback `EMBEDDING_MODEL`), `AU_EMBED_DIMENSION` (bỏ bước embed test nếu set).
  - `get_dense_embedding(cfg)` + `get_sparse_embedding(cfg)`:
    - Nếu `embedding_provider=="bge"` → dùng BGE-m3 dense+sparse local (cùng model `BAAI/bge-m3` — tượng tự `embeddings.py`). Tái dùng class shared để DRY, không copy.
    - Nếu có base_url → giữ pattern OpenAI-compatible (phục vụ self-host serve dense-only; sparse lúc này fallback local BGE-m3 sparse).
  - `dense_vector_size` = `AU_EMBED_DIMENSION` env (mặc định 1024).
  - `sparse_vector_name` đổi sang `"bge-m3-sparse"`.
- Payload logic giữ nguyên; chỉ đổi nguồn dense + tên sparse vector.
- **Bỏ rate limit sleep**: xem §3.6.

## 4. DeepEval mới với Qwen self-host — **kiểm tra tài liệu trả về đúng**

> Mục tiêu test đổi: không đánh giá answer (hệ thống giờ không sinh answer). Đánh giá **retrieval_context** — chunk chứa dữ liệu cần thiết có mặt trong top-K sau rerank hay không.

### 4.1. File mới
- `tests/test_rag_deepeval_qwen.py`: copy từ `test_rag_deepeval.py`, đổi:
  - Eval LLM: `QwenEvalLLM(DeepEvalBaseLLM)` gọi `http://A.B.C.D:E/v1/chat/completions` qua OpenAI client (langchain_openai `ChatOpenAI` hoặc httpx trực tiếp). Body có `"thinking": false`.
  - Bộ câu hỏi: tách riêng `tests/golden_questions_v2.json` (bộ câu hỏi mới — user sẽ cung cấp nội dung cụ thể; phiên agent tạo skeleton + vài ví dụ để verify pipeline).
- Giữ `test_rag_deepeval.py` (Gemini) nguyên để so sánh? → Khuyến nghị **bỏ** (yêu cầu "tạo một phiên bản khác") hoặc rename `.bak`. Quyết định: giữ file cũ nhưng để `@pytest.mark.skip` kèm note, ưu tiên chạy Qwen. Hỏi lại user nếu cần. (Mặc định: giữ, thêm marker `@pytest.mark.skip(reason="legacy gemini eval — giữ để tham khảo, chạy test_qwen")`.)

### 4.2. `QwenEvalLLM` — cấu hình env
Thêm vào `template.env` + `Config` (scope test):
```
# Qwen self-host eval LLM
QWEN_EVAL_BASE_URL=http://A.B.C.D:E/v1
QWEN_EVAL_API_KEY=<token>
QWEN_EVAL_MODEL=qwen3-...  # tên model thực tế do user cung cấp
QWEN_EVAL_THINKING=false
```
`CustomEvalLLM.generate`:
```python
from openai import OpenAI
client = OpenAI(base_url=self.base_url, api_key=self.api_key)
resp = client.chat.completions.create(
    model=self.model,
    messages=[{"role":"user","content":prompt}],
    stream=False,
    extra_body={"thinking": False},  # qwen custom param
)
return resp.choices[0].message.content
```
(Lưu ý: `thinking` là param ngoài OpenAI spec → đẩy qua `extra_body`.)

### 4.3. Rate limiter — **bỏ RAG, giữ eval**
- Do RAG system chạy local (BGE in-process), không còn quota embed/gen → **bỏ `apply_rag` và `apply_embed`**.
- **Giữ `apply_eval`** nguyên logic (RPM + TPM sliding window) vì eval LLM có thể là:
  - Qwen self-host (RNA vendor) — không quota nhưng có giới hạn throughput/GPU.
  - NVIDIA model khi đủ tài nguyên — có quota cloud.
- `UnifiedRateLimiter` giờ chỉ cần track `eval_*`. Có thể giữ class nhưng default RAG slot = 0 hoặc xóa phương thức `apply_rag`/`apply_embed`. Khuyến nghị: **xóa `apply_rag`** + **giữ `apply_eval`** (clear intent).

### 4.4. Bộ câu hỏi mới `tests/golden_questions_v2.json`
- **Tái sử dụng bộ câu hỏi cũ** `golden_questions.json` (14 câu PTIT/AIoT/ATTT) — không viết câu mới. Schema cũ `{id, topic, question, expected_answer, expected_context}` được giữ nguyên về câu hỏi + expected, **bỏ** `expected_answer` (hệ thống giờ không sinh answer) hoặc giữ làm reference nhưng metric không dùng. Thêm field `expected_chunk_locators` (xem §4.4a).
- **Cơ chế kiểm tra mới — kiểm tra chunk trả về nằm đúng phần tài liệu gốc**:
  - Trước khi chạy eval, một bước "locator" quét corpus đã index, tìm chunk nào chứa fragment trong `expected_context` → ghi lại `{document_id, chunk_index, source, heading_path}` cho mỗi câu hỏi. Bước này dùng **một document thật** (chọn 1 doc từ `input/` làm anchor) để xác minh chunk được trích ra từ đúng document/section.
  - Khi RAG trả về `documents` cho câu hỏi, kiểm tra:
    1. **Content match**: page_content của chunk trả về có chứa fragment trong `expected_context` (substring/fuzzy match).
    2. **Locator match**: chunk trả về có `metadata.document_id` + `chunk_index` nằm trong `expected_chunk_locators` (kiểm tra chunk được lấy từ đúng phần tài liệu gốc). Nếu RAG trả chunk khác doc nhưng nội dung khớp (e.g. trùng thông tin ở nhiều doc) → vẫn pass content match, nhưng flag `locator_drift=true` trong log để phân tích.
- **Schema mới**:
  ```json
  [
    {
      "id": 1,
      "topic": "aiot",
      "question": "...",
      "expected_context": ["..."],
      "expected_chunk_locators": [
        {"document_id": "<oid>", "chunk_index": 3, "source": "ten_file_hoac_link", "heading_path": "h1>h2>..."}
      ]
    }
  ]
  ```
  - `expected_context` lấy nguyên từ `golden_questions.json` cũ.
  - `expected_chunk_locators` do bước "locator" điền自動 — không phải user viết tay.
- **Metric DeepEval** (vì giờ kiểm tra context, không có answer):
  - `ContextualPrecisionMetric` — top-K có context match không.
  - `ContextualRecallMetric` — recall context.
  - (Optional) `ContextualRelevancyMetric` nếu deepeval support.
  - **Bỏ**: `AnswerRelevancyMetric`, `FaithfulnessMetric`.
  - `LLMTestCase.actual_output = ""` (empty) — không dùng cho metric context-based.
  - `retrieval_context` = list page_content từ `chain_result["documents"]` (theo thứ tự rerank).
- **Custom assertion ngoài DeepEval** (kiểm tra locator):
  - Hàm `assert_chunk_locators(retrieved_docs, expected_locators)` trong test file, không phụ thuộc LLM — so sánh `d.metadata["global_id"]` (đã được Qdrant trả về nếu migration set id stable) với `expected_chunk_locators[*].global_id`. Pass nếu ≥1 chunk trả về match locator (hoặc content-only fallback nếu corpus bị re-chunk với size khác → so sánh nội dung).

### 4.4a. Bước "locator" — sinh chunk locator từ corpus đã index
- Script nhỏ `tests/locate_expected_chunks.py` (scriptukan dev, không phải test auto):
  - Đọc `golden_questions_v2.json` (đang còn `expected_context` rỗng/placeholder).
  - Với mỗi câu hỏi, scan toàn bộ collection Qdrant: `scroll` tất cả point, kiểm tra point nào có `page_content` chứa từng fragment trong `expected_context` (substring, không dùng query — quét thẳng).
  - Ghi lại `{document_id, chunk_index, source, heading_path, global_id}` (lấy từ payload metadata đã set trong migration: `src/agent/scripts/migrate_dump_to_qdrant.py` `build_payload` đã có `document_id`, `chunk_index`, `global_id`).
  - Ghi ngược lại `golden_questions_v2.json` field `expected_chunk_locators`.
- Chạy một lần sau migration, trước khi chạy test deepeval. Output: bộ golden hoàn chỉnh.
- **Anchor document**: chọn 1 document thật (e.g. doc có `document_type="curriculum"` ATTT) làm reference chính — test chỉ pass nếu chunk trả về từ doc không trùng anchor nhưng khác section → vẫn pass content match, nhưng nếu khác anchor doc hoàn toàn → report `locator_drift`.
- Lưu ý: script này chạy local, không cần LLM, không tốn quota eval.

### 4.5. conftest
- `block_external_http` chặn host ngoài localhost. Test Qwen cần gọi `A.B.C.D:E` → phải allow. Thêm marker `@pytest.mark.qwen` bypass guard (như marker `vcr`). Sửa `conftest.py`:
  ```python
  if os.getenv("ALLOW_NETWORK_TESTS")=="1" or request.node.get_closest_marker("vcr") or request.node.get_closest_marker("qwen"):
      return
  ```
- Hoặc đơn giản: runner set `ALLOW_NETWORK_TESTS=1` khi chạy test qwen.

## 5. template.env & docker-compose

- `template.env`: update default:
  ```
  # Dense embedding (BGE-m3)
  AU_EMBED_MODEL_NAME=BAAI/bge-m3
  AU_EMBED_DIMENSION=1024
  # Sparse embedding (cùng model nhưng env tách — swap linh hoạt sau)
  AU_SPARSE_MODEL_NAME=BAAI/bge-m3
  # Reranker
  AU_RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3
  RERANK_PROVIDER=bge
  # bỏ GEMINI_API_KEY, GENERATION_MODEL, LLM_*, EVAL_LLM_MODEL(Gemini)
  # thêm QWEN_EVAL_*
  QWEN_EVAL_BASE_URL=http://A.B.C.D:E/v1
  QWEN_EVAL_API_KEY=
  QWEN_EVAL_MODEL=
  QWEN_EVAL_THINKING=false
  ```
- `docker-compose.yml`: review service phụ thuộc Gemini (đã có diff giảm 74 dòng ở branch trước). Không cần đổi thêm ngoài env, trừ khi muốn thêm service serving BGE (ngoại scope — giả định BGE chạy qua fastembed in-process hoặc do user serve riêng).

## 6. Dependencies — `pyproject.toml`

- Bỏ (nếu không dùng): `langchain-cohere` (nếu bỏ cohere rerank), `langchain-google-genai` (bỏ Gemini embed), `langchain-litellm` (bỏ LLM gen). Cẩn thận: `langchain-litellm` cũng dùng ở test deepeval legacy — nếu skip file thì safe.
- Giữ/thêm: `fastembed` (đã có), có thể thêm `sentence-transformers` nếu fastembed thiếu BGE reranker.
- `deepeval>=4.1.1`, `openai` (đã có) cho Qwen eval client.
- Chạy `uv sync` + `uv lock` lại sau khi sửa.

## 7. Verification (chạy nhưng KHÔNG commit tự động)

```powershell
uv sync
uv run ruff check .
uv run ruff format --check .
# mypy/ty nếu có
uv run pytest tests/unit_tests -q          # fast
# Chạy test Qwen (cần env thật + Qdrant up):
$env:ALLOW_NETWORK_TESTS="1"
uv run pytest tests/test_rag_deepeval_qwen.py -m qwen -vv
```
- Kiểm tra bằng tay endpoint `/rag` (Trả về documents, không có `answer`).
- Nếu cần migration thật: `python -m agent.scripts.migrate_dump_to_qdrant --limit 5`, verify dense vector size = 1024 trong log.

## 8. Phân chia công việc (task list cho phiên agent kế tiếp)

Tham khảo chi tiết tại `agent.md` (task list đánh số). Tóm tắt:
1. Tạo nhánh + sync.
2. Sửa Config (env `AU_*`, bỏ LLM fields, xóa `gemini_api_key`).
3. Sửa `embeddings.py` (provider bge, dense+sparse từ BGE-m3).
4. Sửa `vdb.py` + `retriever.py` (sparse BGE-m3, đổi tên sparse vector).
5. Sửa migration script (BGE-m3, bỏ rate-limit sleep).
6. Sửa/Thêm reranker bge + wire vào retriever, đảm bảo thứ tự rerank.
7. Refactor graph: bỏ LLM + grading/rewrite/generation.
8. Sửa routes + data models (RetrievalResponse, không answer).
9. Sửa api.py + state/internal_model cleanup.
10. Tạo `test_rag_deepeval_qwen.py` (metric context-based) + `golden_questions_v2.json` (schema mới).
11. Bỏ rate limit RAG trong test, giữ `apply_eval` cho eval LLM.
12. Sửa conftest (marker `qwen` bypass network).
13. Update template.env, pyproject deps, README note.
14. **Dọn file legacy** (plan §11) — xóa nodes/prompts/scripts/diagrams/legacy resources/tests/VCR cũ.
15. Chạy lint/type/test, sửa tới khi xanh.
16. (KHÔNG commit/push trừ khi user yêu cầu — xem `agent.md` mục 7).

## 9. Risk / lưu ý

- **BGE-m3 multi-functional** (dense, sparse, colbert) — repo dùng **dense + sparse từ cùng một model** (bỏ BM25 fastembed `Qdrant/bm25`). Sparse vector đổi tên `"bge-m3-sparse"` → collection cũ incompatible, **phải migration lại** (gỡ collection `documents` rồi chạy migration). Lưu ý thứ tự các bước: stop API → recreate collection → re-index.
- **BGE reranker v2-m3** lingual (multilingual) — phù hợp corpus VN. Cần model download cache lần đầu (~2GB) — note trong README. Reranker là bước **cuối cùng** trước khi trả về — thứ tự list phải khớp score reranker giảm dần, không sort lại.
- **fastembed API**: verify `fastembed.SparseTextEmbedding` support `"BAAI/bge-m3"`; nếu không, fallback `FlagEmbedding` (`BGEM3FlagModel`) — thêm dep `FlagEmbedding` (kéo torch).
- **break contracts cũ**: ConvAgentBruno và frontend đang gọi `/rag` với `answer` → sẽ break. Ghi vào CHANGELOG: major version bump (`v6.0.0`), breaking change. Frontend giờ tự gọi downstream LLM để tổng hợp answer từ documents.
- **Phoenix tracing**: sau khi bỏ LLM, trace sẽ ngắn hơn (chỉ retriever + embed + rerank) — OK.
- **tests/vcr**: cassette cohere rerank có thể fail test nếu xóa provider cohere → giữ provider, chỉ đổi default sang `bge`.
- **rate limit**: bỏ hoàn toàn cho RAG system. Eval LLM (DeepEval) **giữ** vì có thể dùng NVIDIA model cloud sau này → quota cần throttle. Không accidentally gỡ `apply_eval`.
- **test mục tiêu đổi**: không còn đánh giá answer → phải cập nhật test_grading/test_generation cũ (nếu còn) hoặc xóa cùng file node.
- **dọn file nhánh mới**: vì nhánh mới tinh, thuận tiện xóa hết file legacy không dùng cho pipeline retrieval-only BGE — tránh dead code, ruff/mypy sạch, repo nhỏ. Danh sách cụ thể ở §11.

## 10. Điểm chờ user confirm

- a) Bộ câu hỏi mới `golden_questions_v2.json` — **tái sử dụng câu hỏi cũ** `golden_questions.json` (14 câu) — không cần user cung cấp nội dung. Confirm OK.
- b) Endpoint Qwen thực tế (`A.B.C.D:E`, token, model name chính xác) để test eval thật chạy được.
- c) Có giữ `case "google"`/`case "cohere"` trong `embeddings.py`/`reranker.py` hay bỏ hẳn (Note: sparse env giờ tách `AU_SPARSE_MODEL_NAME`, nếu user muốn swap sparse sang BM25 → giữ `case "fastembed"` cho sparse hỗ trợ `Qdrant/bm25`).
- d) Có giữ `test_rag_deepeval.py` (Gemini) hay xóa (mặc định xóa).
- e) Frontend có cần cập nhật trong cùng PR hay tách PR.
- f) Khi xóa legacy file (§11), giữ `.bak`/archive hay xóa thẳng? Mặc định: **xóa thẳng** (`git rm`), git history đủ để khôi phục.
- g) Bộ dummy data `resources/*.pdf`, `tests/resources/*.pdf`, `tests/resources/albert.txt` — giữ để test index/ingest, hay xóa vì corpus thật nằm ở `input/` (migration)? Mặc định: **xóa tests/resources/legacy PDF** (Attention/Albert/Totoro không liên quan domain PTIT), **giữ `resources/` nếu frontend cần demo**.
- h) Bước "locator" (§4.4a) chạy **sau** migration xong để điền `expected_chunk_locators` vào `golden_questions_v2.json` — confirm là user phê duyệt chạy `tests/locate_expected_chunks.py` 1 lần (chỉ quét Qdrant local, không tốn eval quota).

## 11. Dọn file nhánh mới (cleanup legacy)

Vì nhánh `feature/retrieval-only-bge-qwen` là nhánh mới tinh, xảy ra cơ hội xóa triệt để các file không phục vụ pipeline mới (retrieval-only + BGE-m3 + Qwen eval). Mục đích: giảm dead code, ruff/mypy pass sạch, repo gọn, người mới onboard dễ hiểu.

### 11.1. File backend — chắc chắn xóa

| File | Lý do |
|-----|-------|
| `src/agent/backend/nodes/generation.py` | Bỏ node generate (không còn LLM answer) |
| `src/agent/backend/nodes/grading.py` | Bỏ grading LLM |
| `src/agent/backend/nodes/rewrite.py` | Bỏ rewrite LLM |
| `src/agent/backend/prompts.py` | Không còn sinh answer, không load template |
| `src/agent/scripts/visualize_graph.py` | Sinh `graph.png` từ graph cũ có LLM — graph mới đơn giản không cần |
| `src/agent/scripts/generate_diagrams.py` | Diagrams vẽ Cohere/OpenAI/Gemini — sai kiến trúc mới |
| `src/agent/scripts/resources/gemini.png` `cohere.png` `openai.png` `langchain.png` `streamlit.png` `Architecture.png` | Không dùng trong diagram mới |
| `graph.png` | Output của `visualize_graph.py` đã xóa |
| `resources/Architecture.png` `search_flow.png` `cohere.png` `ollama.png` `openai.png` `ui.png` `tracing.png` `research.png` | Diagram/screenshot kiến trúc cũ — nếu cần thì regenerate |

### 11.2. File backend — refactor/xóa phần LLM

| File | Hành động |
|-----|-----------|
| `src/agent/backend/state.py` | Xóa `Grade` (pydantic BaseModel) — không còn grading |
| `src/agent/data_model/internal_model.py` | Xóa `RetrievalResults` nếu `utility.convert_qdrant_result_to_retrieval_results` không caller (grep: chỉ test ref) → xóa cả function trong `utility.py` |
| `src/agent/utils/utility.py` | Xóa `load_prompt_template` (không còn prompt file), `convert_qdrant_result_to_retrieval_results` (không caller ngoài test), `combine_text_from_list` (grep caller — nếu chỉ test thì xóa). Giữ `format_docs_for_citations` (dùng cho trace? grep — nếu chỉ generation đã xóa thì cũng xóa). |
| `src/agent/data_model/response_data_model.py` | Xóa `QAResponse`, `ExplainQAResponse` |
| `src/agent/backend/services/embedding_management.py` | Rename class `EmbeddingManagement` → `IndexingManagement` (clearer) — optional, không bắt buộc. Gỡ `__main__` block test "Was ist Attention?" (legacy German demo). |

### 11.3. Test + VCR — xóa legacy

| File | Hành động |
|-----|-----------|
| `tests/test_rag_deepeval.py` (Gemini) | **Xóa** (sẽ được thay bằng `test_rag_deepeval_qwen.py`). Hoặc giữ `.bak` nếu §10.f user muốn — mặc định xóa. |
| `tests/golden_questions.json` | **Xóa** — thay bằng `golden_questions_v2.json` (schema mới). |
| `tests/vcr/test_litellm_requests.py` + cassette `test_litellm_gemini_chat_vcr.yaml` | **Xóa** — test Gemini chat LLM, không còn LLM. |
| `tests/vcr/test_embedding_and_reranker_requests.py` + 2 cassettes `test_cohere_*.yaml` | **Xóa** — Cohere embed/rerank không dùng mặc định (giữ provider cohere trong code nếu §10.c, nhưng test VCR Cohere không fit pipeline mới — chuyển trọng tâm sang BGE). |
| `tests/vcr/test_contracts.py` | **Giữ** nhưng cập nhật — contract `/semantic/search` không đổi. Nếu giữ thì không xóa. |
| `tests/fakes/rag.py` | **Giữ** — `FakeAsyncRetriever`/`FakeDoc` vẫn dùng cho test contract/unit. |
| `tests/test_stream.py` | **Giữ**, cập nhật assert — `/rag/stream` giờ không có event `content`, chỉ `status`+`documents`. |
| `tests/test_integration.py` | **Giữ**, cập nhật assert (welcome message có thể đổi). |
| `tests/e2e_tests/test_api.py` | **Giữ** — test `/` và `/collection/create`. |
| `tests/unit_tests/test_utility.py` | **Cập nhật** — xóa test cho `load_prompt_template`, `convert_qdrant_result_to_retrieval_results` nếu xóa source. Giữ test `create_tmp_folder`, `format_docs_for_citations` (Nếu còn dùng). |
| `tests/unit_tests/test_rag_graph.py` | **Cập nhật** — graph mới không có LLM, không có answer. Assert documents thay vì answer. |
| `tests/unit_tests/test_reranker.py` | **Cập nhật** — thêm test bge, có thể bỏ test cohere/flashrank nếu xóa provider. |
| `tests/unit_tests/test_embedding_management.py` | **Cập nhật** — đổi test sang BGE embed dim 1024, bỏ Gemini reference. |
| `tests/unit_tests/test_routes_embeddings.py` | **Cập nhật** — route `/embeddings/*` giữ nguyên nhưng kiểm dep import mới. |
| `tests/unit_tests/test_vdb.py` | **Cập nhật** — sparse name `bge-m3-sparse`, model `BAAI/bge-m3`. |
| `tests/resources/1706.03762v5.pdf` (Attention) `1912.01703v1.pdf` (Albert) `albert.txt` | **Xóa** — legacy demo, không liên quan domain PTIT/AIoT/ATTT. |

### 11.4. Infrastructure / docs

| File | Hành động |
|-----|-----------|
| `Dockerfile.frontend` `frontend/` toàn bộ | **Giữ** nếu §10.e user muốn tách PR frontend. Mặc định: **giữ nguyên không sửa**, chỉ note README. |
| `ConvAgentBruno/` (Bruno API requests) | **Cập nhật** `RAG/Chat.bru` `RAG/Stream.bru` — body không cần `answer` expected; hoặc xóa luôn nếu không còn ai dùng Bruno. Mặc định: **giữ + cập nhật** (Bruno nhỏ, là contract test thủ công). |
| `resources/*.pdf` (Attention, Albert, Totoro) | **Xóa** — legacy demo. Giữ nếu làm starter example. Mặc định: **xóa**. |
| `docker-compose.yml` | **Cập nhật** — gỡ env Gemini/Cohere không dùng; thêm `AU_*` service env (optional, có thể dùng `.env` mount). |
| `config/qdrant.yaml` | **Giữ** — config Qdrant server (auth, storage). |
| `Makefile` | **Cập nhật** — note `test-e2e` giờ cần Qwen env; giữ cấu trúc. |
| `README.md` | **Cập nhật toàn bộ** — §2.x "LLMs and Backend Providers" bỏ, §Architecture regenerate, §Quickstart đổi env, thêm §6 "Retrieval-only mode (v6)". |
| `CHANGELOG.md` | **Thêm entry** v6.0.0 breaking change. |
| `.idx/dev.nix` `.devcontainer/` | **Giữ** — dev environment. |
| `.github/workflows/test.yml` | **Cập nhật** — gỡ env dummy Gemini/Cohere nếu có, thêm env Qwen nếu CI chạy deepeval (mặc định skip deepeval trong CI, chạy local). |
| `.markdown-link-check.json` `.markdown-link-check.json` `.pre-commit-config.yaml` `ruff.toml` `.python-version` `pyproject.toml` `uv.lock` | Giữ, chỉ update deps. |

### 11.5. Thứ tự xóa (tránh break mid-way)

1. **Task 6/7** (graph + node cleanup) → xóa source files `nodes/*.py`, `prompts.py`, `state.Grade`, `internal_model` nếu không caller.
2. **Task 8** (routes + data models) → xóa `QAResponse`, `ExplainQAResponse` từ `response_data_model.py`.
3. **Task utility** → xóa `load_prompt_template`, `convert_qdrant_result_to_retrieval_results` khỏi `utility.py`.
4. **Task test** → xóa `test_rag_deepeval.py` + cassettes Gemini/Cohere + `golden_questions.json` + `tests/resources/legacy`. Cập nhật test còn lại.
5. **Task scripts** → xóa `visualize_graph.py`, `generate_diagrams.py`, `scripts/resources/*.png`, `graph.png`, `resources/legacy`.
6. **Cuối**: `git rm` một lượt, commit riêng `"chore(cleanup): remove legacy gen-LLM files"` để git history rõ. Chỉ commit khi user approve.
