"""
DeepEval test cho Hybrid RAG dựa trên dữ liệu thật từ `../input/`.
Dùng `DeepEvalBaseLLM` bọc Google Gemini client theo `EVAL_LLM_*`.
Các câu hỏi vàng lấy từ `tests/golden_questions.json`.
"""
import os
import json
import sys
import time
import threading
from pathlib import Path

_env_test = Path(__file__).parent / ".env.test"
_env_main = Path(__file__).parent.parent / ".env"

# Load .env trước (nếu có), rồi .env.test override (nếu có)
from dotenv import load_dotenv
if _env_main.exists():
    load_dotenv(_env_main, override=False)
    print(f"[deepeval] Loaded base env from {_env_main}")
if _env_test.exists():
    load_dotenv(_env_test, override=True)
    print(f"[deepeval] Loaded test env from {_env_test} (override)")

src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent.backend.graph import Graph

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

GOLDEN_FILE = Path(__file__).parent / "golden_questions.json"

# ============================================================
# RATE LIMITER: Unified for RAG System + Eval LLM
# - RPM: chỉ cho phép tối đa `rpm` request trong cửa sổ 60s trượt
# - TPM: chỉ cho phép tối đa `tpm` tokens trong cửa sổ 60s trượt
# Mặc định nghiêm ngặt, override qua env
# ============================================================
# Generation: rpm=30, tpm=16k
# Embedding (3072-dim): rpm=100, tpm=30k
# Eval LLM: rpm=20, tpm=40k


class UnifiedRateLimiter:
    """Track and enforce RPM+TPM rate limits for RAG gen/embed AND eval LLM calls."""

    def __init__(
        self,
        gen_rpm: int = 30,
        gen_tpm: int = 16000,
        embed_rpm: int = 100,
        embed_tpm: int = 30000,
        eval_rpm: int = 20,
        eval_tpm: int = 40000,
    ):
        self.gen_rpm = gen_rpm
        self.gen_tpm = gen_tpm
        self.embed_rpm = embed_rpm
        self.embed_tpm = embed_tpm
        self.eval_rpm = eval_rpm
        self.eval_tpm = eval_tpm

        self._lock = threading.Lock()
        self._gen_timestamps: list[float] = []
        self._gen_token_log: list[tuple[float, int]] = []
        self._embed_timestamps: list[float] = []
        self._embed_token_log: list[tuple[float, int]] = []
        self._eval_timestamps: list[float] = []
        self._eval_token_log: list[tuple[float, int]] = []

    def _clean_old_timestamps(self, timestamps: list[float], window: float = 60.0) -> None:
        """Xóa timestamps cũ hơn `window` giây."""
        now = time.time()
        while timestamps and now - timestamps[0] > window:
            timestamps.pop(0)

    def _clean_old(self, log: list[tuple[float, int]], window: float = 60.0) -> None:
        now = time.time()
        while log and now - log[0][0] > window:
            log.pop(0)

    def _wait_rpm(self, timestamps: list[float], rpm: int) -> None:
        """Block cho đến khi số request trong 60s < rpm, rồi append 1 timestamp mới."""
        while True:
            self._clean_old_timestamps(timestamps)
            if len(timestamps) < rpm:
                break
            # Còn quá giới hạn → sleep đến khi entry cũ nhất hết window
            sleep_time = 60.0 - (time.time() - timestamps[0])
            if sleep_time > 0.01:
                time.sleep(sleep_time)
            else:
                # Entry cũ nhất đã > 60s nhưng chưa bị clean (race) → loop lại sẽ clean
                time.sleep(0.01)
        timestamps.append(time.time())

    def _wait_tpm(self, token_log: list[tuple[float, int]], tpm: int, tokens: int) -> None:
        now = time.time()
        self._clean_old(token_log)
        current_tokens = sum(t for _, t in token_log)
        while current_tokens + tokens > tpm:
            if not token_log:
                break
            sleep_time = 60.0 - (now - token_log[0][0])
            if sleep_time > 0.01:
                time.sleep(sleep_time)
                now = time.time()
                self._clean_old(token_log)
                current_tokens = sum(t for _, t in token_log)
            else:
                break
        token_log.append((time.time(), tokens))

    def _estimate_gen_tokens(self, question: str, answer: str, num_docs: int) -> tuple[int, int]:
        input_tokens = int(len(question.split()) * 1.3)
        context_tokens = num_docs * 200
        output_tokens = int(len(answer) / 4)
        return input_tokens + context_tokens, output_tokens

    def _estimate_embed_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def _estimate_eval_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def apply_rag(self, question: str, answer: str = "", num_docs: int = 0) -> None:
        with self._lock:
            gen_input, gen_output = self._estimate_gen_tokens(question, answer, num_docs)
            embed_tokens = self._estimate_embed_tokens(question) + num_docs * 100

            self._wait_rpm(self._gen_timestamps, self.gen_rpm)
            self._wait_tpm(self._gen_token_log, self.gen_tpm, gen_input + gen_output)

            self._wait_rpm(self._embed_timestamps, self.embed_rpm)
            self._wait_tpm(self._embed_token_log, self.embed_tpm, embed_tokens)

    def apply_eval(self, text: str, requests: int = 1) -> None:
        """Reserve `requests` slot cho eval LLM trong window 60s.
        Mỗi slot = 1 timestamp append vào _eval_timestamps → rate limit chính xác.
        """
        with self._lock:
            tokens = self._estimate_eval_tokens(text) * requests
            # Reserve đủ `requests` slot RPM (mỗi slot block riêng, append riêng)
            for _ in range(requests):
                self._wait_rpm(self._eval_timestamps, self.eval_rpm)
            self._wait_tpm(self._eval_token_log, self.eval_tpm, tokens)


limiter = UnifiedRateLimiter(
    gen_rpm=int(os.environ.get("RAG_GEN_RPM", 30)),
    gen_tpm=int(os.environ.get("RAG_GEN_TPM", 16000)),
    embed_rpm=int(os.environ.get("RAG_EMBED_RPM", 100)),
    embed_tpm=int(os.environ.get("RAG_EMBED_TPM", 30000)),
    eval_rpm=int(os.environ.get("EVAL_RPM", 20)),
    eval_tpm=int(os.environ.get("EVAL_TPM", 40000)),
)
print(
    f"[deepeval] Rate limits: "
    f"gen(rpm={limiter.gen_rpm}, tpm={limiter.gen_tpm}) "
    f"embed(rpm={limiter.embed_rpm}, tpm={limiter.embed_tpm}) "
    f"eval(rpm={limiter.eval_rpm}, tpm={limiter.eval_tpm})"
)


class CustomEvalLLM(DeepEvalBaseLLM):
    """Wrapper cho Google Gemini làm evaluation LLM (DeepEval).
    Dùng ChatLiteLLM với prefix `gemini/` — giống graph.py, đảm bảo output dạng string
    (DeepEval cần string thô để tự parse JSON schema).

    Cấu hình qua env:
      - EVAL_LLM_MODEL: tên model, mặc định "gemini/gemini-2.5-flash" (必须有 prefix `gemini/`)
      - EVAL_LLM_API_KEY hoặc GEMINI_API_KEY
    """

    def __init__(self):
        model = os.environ.get("EVAL_LLM_MODEL", "gemini/gemini-2.5-flash")
        # Đảm bảo có prefix gemini/ cho LiteLLM
        if not model.startswith("gemini/") and not model.startswith("gemini-"):
            model = f"gemini/{model}" if "/" not in model else model
        elif model.startswith("gemini-"):
            model = f"gemini/{model}"

        api_key = os.environ.get("EVAL_LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("EVAL_LLM_API_KEY hoặc GEMINI_API_KEY phải được set")

        self._model_name = model
        self._api_key = api_key

        super().__init__(model=model)
        print(f"[CustomEvalLLM] model={self._model_name}")

    def load_model(self):
        from langchain_litellm import ChatLiteLLM
        return ChatLiteLLM(
            model=self._model_name,
            api_key=self._api_key,
            temperature=0,
        )

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.load_model().ainvoke(prompt)).content

    def get_model_name(self):
        return self._model_name


def load_golden_cases() -> list:
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


rag_graph = Graph().build_graph()
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", os.environ.get("QDRANT_COLLECTION_NAME", "documents"))
CHECKPOINT_FILE = Path(os.environ.get("CHUNK_CHECKPOINT_FILE", "./chunk_checkpoint.jsonl"))

from agent.utils.vdb import qdrant_client

# if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
#     pytest.skip(f"Qdrant collection '{COLLECTION_NAME}' does not exist. Run indexing first.", allow_module_level=True)
# if not CHECKPOINT_FILE.exists() or CHECKPOINT_FILE.stat().st_size == 0:
#     pytest.skip(f"Checkpoint file '{CHECKPOINT_FILE}' is empty or missing. Run indexing first.", allow_module_level=True)

golden_cases = load_golden_cases()


@pytest.mark.parametrize("case", golden_cases)
def test_rag_evaluation(case):
    """Mỗi test case:
    1. Block sync cho RAG (gen + embed) theo RPM
    2. Invoke graph → lấy answer + contexts
    3. Block sync cho 4 metric eval slots (reserve trước khi DeepEval asyncio.gather)
    4. assert_test → 4 metric chạy song song, mỗi metric đã reserve 1 slot
    """
    eval_llm = CustomEvalLLM()

    # ---- RATE LIMIT: RAG System (1 gen + 1 embed) ----
    limiter.apply_rag(question=case["question"])

    messages = [{"role": "user", "content": case["question"]}]

    chain_result = rag_graph.invoke(
        {"messages": messages},
        config={"metadata": {"collection_name": COLLECTION_NAME}},
    )

    actual_answer = chain_result["messages"][-1].content
    num_docs = len(chain_result.get("documents", []))
    actual_contexts = [doc.page_content for doc in chain_result.get("documents", [])]

    # ---- RATE LIMIT: Eval LLM (4 metrics = 4 slots, reserve ĐỒNG BỘ trước khi a_measure) ----
    # Reserve xong mới DeepEval chạy song song — đảm bảo mỗi metric chỉ trigger 1 slot đã được reserve
    limiter.apply_eval(text=case["question"] + actual_answer, requests=4)

    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=eval_llm, include_reason=True),
        FaithfulnessMetric(threshold=0.7, model=eval_llm, include_reason=True),
        ContextualPrecisionMetric(threshold=0.7, model=eval_llm, include_reason=True),
        ContextualRecallMetric(threshold=0.7, model=eval_llm, include_reason=True),
    ]

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_answer,
        expected_output=case.get("expected_answer", ""),
        retrieval_context=actual_contexts,
    )

    assert_test(test_case, metrics)

    # ---- Cooldown giữa các test case: đảm bảo RPM recover ----
    # 14 test × (1 RAG + 4 eval) = 70 RPM cho eval model → cần ~6 phút để hoàn tất với RPM=13
    # Sleep ngắn giúp tránh burst khi pytest chạy lưu niệm liên tiếp
    time.sleep(float(os.environ.get("TEST_CASE_COOLDOWN", "0")))
