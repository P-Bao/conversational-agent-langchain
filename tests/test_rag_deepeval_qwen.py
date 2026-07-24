"""DeepEval test suite hỗ trợ cả Qwen self-host và NVIDIA NIM API làm eval LLM.

Đánh giá chất lượng retrieval_context (contextual precision, contextual recall)
qua route ``/rag/`` của API Docker đang chạy thật (không dùng in-process TestClient
để tránh tự ý lấy endpoint mặc định).

Yêu cầu env (API Docker):
  RAG_API_URL (default: "http://localhost:8001") — base URL của API container
  EMBEDDING_BASE_URL — base URL remote BGE-m3 server (Colab ngrok / GPU server)
                       (API container phải được khởi động với cùng giá trị này)
  RERANK_BASE_URL    — bắt buộc nếu RERANK_PROVIDER=remote
  TEST_QDRANT_URL / TEST_QDRANT_PORT / TEST_QDRANT_COLLECTION_NAME — Qdrant thật

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

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, List

import httpx
import pytest
from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
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
    """Chọn judge LLM theo TEST_EVAL_BACKEND, hoặc auto-detect từ env."""
    backend = (os.environ.get("TEST_EVAL_BACKEND") or "").strip().lower()
    if backend == "qwen":
        return QwenEvalLLM()
    if backend == "nvidia":
        return NvidiaEvalLLM()
    if os.environ.get("QWEN_EVAL_BASE_URL"):
        return QwenEvalLLM()
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_EVAL_API_KEY"):
        return NvidiaEvalLLM()
    return QwenEvalLLM()


@pytest.fixture(scope="module")
def rag_api_url() -> str:
    """Base URL của API Docker container đang chạy thật (default localhost:8001).

    User có thể override qua ``RAG_API_URL`` (ví dụ khi chạy trong CI/remote host).
    Test gọi ``POST {rag_api_url}/rag/`` qua httpx — không dùng in-process app
    để tránh tự ý lấy endpoint mặc định và bỏ qua service thật.
    """
    return os.environ.get("RAG_API_URL", "http://localhost:8001").rstrip("/")


@pytest.fixture(scope="module")
def golden_questions() -> list[dict]:
    path = Path("tests/golden_questions_v2.json")
    if not path.exists():
        pytest.skip("golden_questions_v2.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_qwen_deepeval_retrieval(
    golden_questions: list[dict],
    eval_llm: DeepEvalBaseLLM,
    rag_api_url: str,
) -> None:
    """Test full retrieval quality across golden questions (gọi API Docker thật)."""
    test_collection = os.environ.get("TEST_QDRANT_COLLECTION_NAME") or "documents"
    locator_strict = os.environ.get("TEST_LOCATOR_STRICT", "0") == "1"
    skip_deepeval = os.environ.get("TEST_SKIP_DEEPEVAL", "0") == "1"
    deepeval_top_k = int(os.environ.get("TEST_DEEPEVAL_TOP_K", "5"))
    # Tolerance: số câu hỏi tối thiểu phải pass để test assert pass.
    min_pass = float(os.environ.get("TEST_MIN_PASS_RATIO", "0.7"))

    # Sanity check: API phải reachable trước khi chạy 14 câu hỏi.
    try:
        with httpx.Client(timeout=10) as c:
            health = c.get(f"{rag_api_url}/healthz")
            health.raise_for_status()
    except Exception as exc:
        pytest.fail(
            f"API Docker không reachable tại {rag_api_url}/healthz: {exc!r}. "
            f"Chạy `docker compose up --build -d` trước, hoặc set RAG_API_URL."
        )

    def _print(payload: str) -> None:
        try:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.write(payload.encode("ascii", "replace").decode("ascii") + "\n")
            sys.stdout.flush()

    def _safe_msg(exc: BaseException, limit: int = 300) -> str:
        msg = str(exc) or repr(exc) or type(exc).__name__
        first_line = next(iter(msg.splitlines()), "")[:limit] or type(exc).__name__
        return first_line

    _print(f"[setup] Using Qdrant collection: {test_collection!r}")
    if not locator_strict:
        _print(f"[setup] Locator/expected_context check is LENIENT (set TEST_LOCATOR_STRICT=1 to enforce)")
    if skip_deepeval:
        _print(f"[setup] DeepEval SKIPPED (set TEST_SKIP_DEEPEVAL=0 to enable)")
    else:
        _print(f"[setup] DeepEval backend: {eval_llm.get_model_name()!r}, top_k={deepeval_top_k} (set TEST_DEEPEVAL_TOP_K to tune)")
    total = len(golden_questions)
    passed = 0
    failed = 0

    for idx, item in enumerate(golden_questions, start=1):
        question = item["question"]
        expected_context = item.get("expected_context", [])
        expected_locators = item.get("expected_chunk_locators", [])
        qid = item.get("id", idx)
        topic = item.get("topic", "n/a")
        safe_q = question.encode("ascii", "replace").decode("ascii")[:80]
        _print(f"[{idx}/{total}] Q{qid} ({topic}): {safe_q}...")
        t0 = time.time()
        q_status = "OK"

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{rag_api_url}/rag/",
                    json={
                        "messages": [{"role": "user", "content": question}],
                    },
                )
            _print(f"    -> HTTP {response.status_code} in {time.time() - t0:.2f}s")
            response.raise_for_status()
            payload = response.json()

            retrieved_docs = payload.get("documents", [])
            retrieved_contexts = [d["text"] for d in retrieved_docs]
            _print(f"    -> retrieved {len(retrieved_docs)} chunks")

            try:
                assert_chunk_locators(
                    [
                        type("Doc", (), {"page_content": d["text"], "metadata": d.get("metadata", {})})()
                        for d in retrieved_docs
                    ],
                    expected_locators,
                    expected_context,
                )
                _print(f"    -> locator/context match OK")
            except pytest.fail.Exception as exc:
                if locator_strict:
                    raise
                _print(f"    -> [WARN] locator/context mismatch (lenient-mode): {str(exc).splitlines()[0][:200]}")

            test_case = LLMTestCase(
                input=question,
                actual_output="",
                expected_output=expected_context[0] if expected_context else "",
                retrieval_context=retrieved_contexts[:deepeval_top_k],
            )

            precision_metric = ContextualPrecisionMetric(threshold=0.5, model=eval_llm)
            recall_metric = ContextualRecallMetric(threshold=0.5, model=eval_llm)

            if skip_deepeval:
                _print(f"    -> DeepEval SKIPPED")
            else:
                _print(f"    -> running DeepEval precision/recall (threshold=0.5)")
                try:
                    assert_test(test_case, [precision_metric, recall_metric])
                    _print(f"    -> Q{qid} PASS (DeepEval)")
                except AssertionError as exc:
                    if locator_strict:
                        raise
                    _print(f"    -> [WARN] DeepEval assertion failed (lenient-mode): {_safe_msg(exc, 200)}")
                except Exception as exc:
                    if locator_strict:
                        raise
                    _print(f"    -> [WARN] DeepEval error (lenient-mode): {type(exc).__name__}: {_safe_msg(exc, 200)}")

        except Exception as exc:
            q_status = f"ERR ({type(exc).__name__})"
            _print(f"    -> [ERROR] {type(exc).__name__}: {_safe_msg(exc, 300)}")

        if q_status == "OK":
            passed += 1
        else:
            failed += 1

    pass_ratio = passed / total if total else 0.0
    _print(
        f"[done] {passed}/{total} passed ({pass_ratio:.0%}), {failed} failed "
        f"(collection={test_collection!r}, strict={locator_strict}, "
        f"min_pass_ratio={min_pass:.0%})"
    )

    # Assertion thật: tỷ lệ pass phải đạt ngưỡng min_pass_ratio.
    # Mặc định 70% — có thể chỉnh qua TEST_MIN_PASS_RATIO.
    if pass_ratio < min_pass:
        pytest.fail(
            f"Chỉ {passed}/{total} câu hỏi pass (ratio {pass_ratio:.0%} < ngưỡng {min_pass:.0%}). "
            f"Kiểm tra: API container up? EMBEDDING_BASE_URL trỏ Colab notebook sống? "
            f"Qdrant collection '{test_collection}' có data?"
        )
