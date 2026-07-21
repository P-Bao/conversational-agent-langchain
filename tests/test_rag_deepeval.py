"""
DeepEval test cho Hybrid RAG dựa trên dữ liệu thật từ `../input/`.
Dùng `DeepEvalBaseLLM` bọc client theo `EVAL_LLM_*`.
Các câu hỏi vàng lấy từ `tests/golden_questions.json`.
"""
import os
import json
import sys
import time
import threading
from pathlib import Path

_env_test = Path(__file__).parent / ".env.test"
if _env_test.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_test, override=True)
    print(f"[deepeval] Loaded env from {_env_test}")

import litellm

import sys
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

    def _clean_old(self, log: list[tuple[float, int]], window: float = 60.0) -> None:
        now = time.time()
        while log and now - log[0][0] > window:
            log.pop(0)

    def _wait_rpm(self, timestamps: list[float], rpm: int) -> None:
        now = time.time()
        while timestamps and now - timestamps[0] < 60.0:
            sleep_time = 60.0 - (now - timestamps[0])
            if sleep_time > 0.01:
                time.sleep(sleep_time)
                now = time.time()
            else:
                break
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
        with self._lock:
            tokens = self._estimate_eval_tokens(text) * requests
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


class CustomEvalLLM(DeepEvalBaseLLM):
    """Bọc OpenAI-compatible LLM cho DeepEval evaluation (OpenRouter / NVIDIA NIM)."""

    def __init__(self):
        base_url = os.environ.get("EVAL_LLM_BASE_URL", "").rstrip("/")
        api_key = os.environ.get("EVAL_LLM_API_KEY", "")
        model = os.environ.get("EVAL_LLM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")

        if not base_url or not api_key:
            raise ValueError(
                f"EVAL_LLM_BASE_URL and EVAL_LLM_API_KEY must be set. "
                f"Got base_url='{base_url}', api_key={'SET' if api_key else 'EMPTY'}"
            )

        self._model_name = model
        self._base_url = base_url
        self._api_key = api_key

        super().__init__(model=model)

        print(f"[CustomEvalLLM] base_url={self._base_url}, model={self.model}, api_key={'SET' if self._api_key else 'EMPTY'}")

    def load_model(self):
        from langchain_openai import ChatOpenAI

        if not self._base_url:
            raise ValueError("base_url is empty in load_model()")
        if not self._api_key:
            raise ValueError("api_key is empty in load_model()")

        return ChatOpenAI(
            model=self._model_name,
            api_key=self._api_key,
            base_url=self._base_url,
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

if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    pytest.skip(f"Qdrant collection '{COLLECTION_NAME}' does not exist. Run indexing first.")
if not CHECKPOINT_FILE.exists() or CHECKPOINT_FILE.stat().st_size == 0:
    pytest.skip(f"Checkpoint file '{CHECKPOINT_FILE}' is empty or missing. Run indexing first.")

golden_cases = load_golden_cases()


@pytest.mark.parametrize("case", golden_cases)
def test_rag_evaluation(case):
    eval_llm = CustomEvalLLM()

    # ---- RATE LIMIT: RAG System ----
    limiter.apply_rag(question=case["question"])

    messages = [{"role": "user", "content": case["question"]}]

    chain_result = rag_graph.invoke(
        {"messages": messages},
        config={"metadata": {"collection_name": COLLECTION_NAME}},
    )

    actual_answer = chain_result["messages"][-1].content
    num_docs = len(chain_result.get("documents", []))
    actual_contexts = [doc.page_content for doc in chain_result.get("documents", [])]

    # ---- RATE LIMIT: Eval LLM (4 metrics x calls) ----
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
