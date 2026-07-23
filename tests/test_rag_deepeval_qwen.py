"""DeepEval test suite hỗ trợ cả Qwen self-host và NVIDIA NIM API làm eval LLM.

Đánh giá chất lượng retrieval_context (contextual precision, contextual recall)
qua route ``/rag/`` của API. Dùng TestClient nên không cần server live.

Yêu cầu env (NVIDIA NIM):
  NVIDIA_API_KEY (hoặc NVIDIA_EVAL_API_KEY)
  NVIDIA_EVAL_MODEL (default: "meta/llama-3.3-70b-instruct")
  NVIDIA_EVAL_BASE_URL (default: "https://integrate.api.nvidia.com/v1")
  NVIDIA_EVAL_RPS (default: "30")

Yêu cầu env (Qwen self-host fallback):
  QWEN_EVAL_BASE_URL
  QWEN_EVAL_API_KEY
  QWEN_EVAL_MODEL
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, List

import pytest
from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from fastapi.testclient import TestClient
from openai import OpenAI

pytestmark = [pytest.mark.qwen]


class RateLimiter:
    """Thread-safe rate limiter cho API requests (ví dụ 30 req/s cho NVIDIA NIM)."""

    def __init__(self, max_rps: float = 30.0) -> None:
        self.max_rps = max_rps
        self.min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_time = now


class NvidiaEvalLLM(DeepEvalBaseLLM):
    """Custom DeepEval LLM wrapper cho NVIDIA NIM API với rate limit."""

    def __init__(self) -> None:
        self.base_url = os.environ.get("NVIDIA_EVAL_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_EVAL_API_KEY", "")
        self.model = os.environ.get("NVIDIA_EVAL_MODEL", "meta/llama-3.3-70b-instruct")
        max_rps = float(os.environ.get("NVIDIA_EVAL_RPS", "30.0"))
        self.rate_limiter = RateLimiter(max_rps=max_rps)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def load_model(self) -> OpenAI:
        return self.client

    def generate(self, prompt: str) -> str:
        self.rate_limiter.wait()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model


class QwenEvalLLM(DeepEvalBaseLLM):
    """Custom DeepEval LLM wrapper cho Qwen OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        self.base_url = os.environ.get("QWEN_EVAL_BASE_URL", "http://localhost:8000/v1")
        self.api_key = os.environ.get("QWEN_EVAL_API_KEY", "dummy")
        self.model = os.environ.get("QWEN_EVAL_MODEL", "qwen")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def load_model(self) -> OpenAI:
        return self.client

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            extra_body={"thinking": False},
        )
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model


def assert_chunk_locators(retrieved_docs: List[Any], expected_locators: List[dict], expected_context: List[str]) -> None:
    """Assertion ngoài DeepEval: kiểm tra chunk trả về nằm đúng phần tài liệu gốc."""
    if not expected_locators and not expected_context:
        return

    retrieved_gids = {doc.metadata.get("global_id") for doc in retrieved_docs if doc.metadata.get("global_id")}
    expected_gids = {loc["global_id"] for loc in expected_locators if "global_id" in loc}

    if retrieved_gids & expected_gids:
        return

    retrieved_text = " ".join([getattr(doc, "page_content", "") for doc in retrieved_docs]).lower()
    for fragment in expected_context:
        if fragment.strip().lower() in retrieved_text:
            return

    pytest.fail(
        f"None of the retrieved chunks matched expected locators or context. Retrieved gids: {retrieved_gids}"
    )


@pytest.fixture(scope="module")
def eval_llm() -> DeepEvalBaseLLM:
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_EVAL_API_KEY"):
        return NvidiaEvalLLM()
    return QwenEvalLLM()


@pytest.fixture(scope="module")
def golden_questions() -> list[dict]:
    path = Path("tests/golden_questions_v2.json")
    if not path.exists():
        pytest.skip("golden_questions_v2.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rag_client(app) -> TestClient:
    """In-process TestClient đã được patch expensive startups bởi fixture `app`."""
    return TestClient(app)


def test_qwen_deepeval_retrieval(
    golden_questions: list[dict],
    eval_llm: DeepEvalBaseLLM,
    rag_client: TestClient,
) -> None:
    """Test full retrieval quality across golden questions (via /rag route)."""
    for item in golden_questions:
        question = item["question"]
        expected_context = item.get("expected_context", [])
        expected_locators = item.get("expected_chunk_locators", [])

        response = rag_client.post(
            "/rag/",
            json={
                "messages": [{"role": "user", "content": question}],
                "collection_name": item.get("collection_name", "default"),
            },
        )
        response.raise_for_status()
        payload = response.json()

        retrieved_docs = payload.get("documents", [])
        retrieved_contexts = [d["text"] for d in retrieved_docs]

        assert_chunk_locators(
            [
                type("Doc", (), {"page_content": d["text"], "metadata": d.get("metadata", {})})()
                for d in retrieved_docs
            ],
            expected_locators,
            expected_context,
        )

        test_case = LLMTestCase(
            input=question,
            actual_output="",
            expected_output=expected_context[0] if expected_context else "",
            retrieval_context=retrieved_contexts,
        )

        precision_metric = ContextualPrecisionMetric(threshold=0.5, model=eval_llm)
        recall_metric = ContextualRecallMetric(threshold=0.5, model=eval_llm)

        assert_test(test_case, [precision_metric, recall_metric])
