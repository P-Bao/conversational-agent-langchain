"""DeepEval RAG evaluation — NVIDIA NIM only (single-file judge + generator).

Replaces ``test_rag_deepeval_qwen.py``: one backend only (NVIDIA NIM), one
file instead of two evaluator classes. Full metric coverage matching the
reference notebook ``example/evaluation_deep_eval.ipynb``:

  * GEval (Correctness)             — notebook cell 1
  * FaithfulnessMetric              — notebook cell 2
  * ContextualRelevancyMetric       — notebook cell 3
  * ContextualPrecisionMetric       — extra (from legacy test, kept)
  * ContextualRecallMetric          — extra (from legacy test, kept)

Batch ``evaluate()`` over all test cases (notebook's final ``evaluate()``
pattern), with per-question summary like before.

How it works:
  1. Calls ``POST {RAG_API_URL}/rag/`` (real Docker API) to retrieve docs.
  2. Generates an answer with the same NIM LLM using retrieved chunks as
     context -> this is ``actual_output`` (notebook needs it for
     Correctness/Faithfulness).
  3. Runs the four retrieval/answer metrics against ``expected_output``
     (``expected_context[0]``) and ``retrieval_context``.
  4. Reports per-question PASS/FAIL and one final ``evaluate()`` summary.

Env (all optional except ``NVIDIA_API_KEY``):
  NVIDIA_API_KEY / NVIDIA_EVAL_API_KEY     — NVIDIA NIM auth token (required).
  NVIDIA_EVAL_MODEL                        — default ``meta/llama-3.3-70b-instruct``.
  NVIDIA_EVAL_BASE_URL                     — default ``https://integrate.api.nvidia.com/v1``.
  NVIDIA_EVAL_RPS                          — request rate cap, default ``30``.
  RAG_API_URL                              — default ``http://localhost:8001``.
  TEST_QDRANT_COLLECTION_NAME              — Qdrant collection name (default ``documents``).
  TEST_DEEPEVAL_TOP_K                      — contexts sent to judge (default ``5``).
  TEST_MIN_PASS_RATIO                      — pass threshold, default ``0.7``.
  TEST_LOCATOR_STRICT                      — ``1`` = hard-fail on locator mismatch.
  TEST_SKIP_ANSWER_GEN                     — ``1`` = skip NIM answer generation
                                             (uses top-1 context as actual_output).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from openai import OpenAI

pytestmark = [pytest.mark.qwen]  # reuse existing marker so CI tags still work


# ---------------------------------------------------------------------------
# Rate limiter (thread-safe, reused from legacy test)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Thread-safe rate limiter for NVIDIA NIM API calls."""

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


# ---------------------------------------------------------------------------
# NVIDIA NIM judge LLM (the ONLY eval backend in this file)
# ---------------------------------------------------------------------------


class NvidiaEvalLLM(DeepEvalBaseLLM):
    """NVIDIA NIM DeepEval wrapper with rate limiting."""

    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "NVIDIA_EVAL_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get(
            "NVIDIA_EVAL_API_KEY", ""
        )
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


# ---------------------------------------------------------------------------
# Non-DeepEval assertion (chunk locator / expected_context match)
# ---------------------------------------------------------------------------


def assert_chunk_locators(
    retrieved_docs: list[Any],
    expected_locators: list[dict],
    expected_context: list[str],
) -> None:
    """Verify retrieved chunks sit in the expected location of the source doc."""
    if not expected_locators and not expected_context:
        return

    retrieved_gids = {
        doc.metadata.get("global_id")
        for doc in retrieved_docs
        if doc.metadata.get("global_id")
    }
    expected_gids = {loc["global_id"] for loc in expected_locators if "global_id" in loc}

    if retrieved_gids & expected_gids:
        return

    retrieved_text = " ".join(
        [getattr(doc, "page_content", "") for doc in retrieved_docs]
    ).lower()
    for fragment in expected_context:
        if fragment.strip().lower() in retrieved_text:
            return

    pytest.fail(
        f"None of the retrieved chunks matched expected locators or context. "
        f"Retrieved gids: {retrieved_gids}"
    )


# ---------------------------------------------------------------------------
# Notebook-equivalent helper: batch test-case builder (notebook cell 5)
# ---------------------------------------------------------------------------


def create_deep_eval_test_cases(
    questions: list[str],
    gt_answers: list[str],
    generated_answers: list[str],
    retrieved_documents: list[list[str]],
) -> list[LLMTestCase]:
    """Build ``LLMTestCase`` list from the four parallel lists, exactly like
    the reference notebook's ``create_deep_eval_test_cases``.
    """
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_llm() -> DeepEvalBaseLLM:
    """One judge: NVIDIA NIM."""
    return NvidiaEvalLLM()


@pytest.fixture(scope="module")
def rag_api_url() -> str:
    """Docker API base URL (override via ``RAG_API_URL``)."""
    return os.environ.get("RAG_API_URL", "http://localhost:8001").rstrip("/")


@pytest.fixture(scope="module")
def golden_questions() -> list[dict]:
    path = Path("tests/golden_questions_v2.json")
    if not path.exists():
        pytest.skip("golden_questions_v2.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def test_nim_deepeval_evaluation(
    golden_questions: list[dict],
    eval_llm: DeepEvalBaseLLM,
    rag_api_url: str,
) -> None:
    """Full RAG quality evaluation — all 5 DeepEval metrics, NIM judge only."""
    deepeval_top_k = int(os.environ.get("TEST_DEEPEVAL_TOP_K", "5"))
    min_pass = float(os.environ.get("TEST_MIN_PASS_RATIO", "0.7"))
    locator_strict = os.environ.get("TEST_LOCATOR_STRICT", "0") == "1"
    skip_answer_gen = os.environ.get("TEST_SKIP_ANSWER_GEN", "0") == "1"
    test_collection = os.environ.get("TEST_QDRANT_COLLECTION_NAME") or "documents"

    def _print(payload: str) -> None:
        try:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.write(payload.encode("ascii", "replace").decode("ascii") + "\n")
            sys.stdout.flush()

    def _safe_msg(exc: BaseException, limit: int = 300) -> str:
        msg = str(exc) or repr(exc) or type(exc).__name__
        return next(iter(msg.splitlines()), "")[:limit] or type(exc).__name__

    # Sanity: API must be reachable before running the whole set.
    try:
        with httpx.Client(timeout=10) as c:
            health = c.get(f"{rag_api_url}/healthz")
            health.raise_for_status()
    except Exception as exc:
        pytest.fail(
            f"API Docker unreachable at {rag_api_url}/healthz: {exc!r}. "
            "Run `docker compose up --build -d` first or set RAG_API_URL."
        )

    _print(f"[setup] NVIDIA NIM judge: {eval_llm.get_model_name()!r}")
    _print(f"[setup] Qdrant collection: {test_collection!r}, top_k={deepeval_top_k}")
    _print(f"[setup] locator_strict={locator_strict}, skip_answer_gen={skip_answer_gen}")
    if not locator_strict:
        _print("[setup] Locator/expected_context check is LENIENT")

    # Build all metrics once (notebook's evaluate()/metrics=[...] pattern).
    correctness_metric = GEval(
        name="Correctness",
        model=eval_llm,
        evaluation_params=[
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        evaluation_steps=[
            "Determine whether the actual output is factually correct based on the expected output."
        ],
        threshold=0.5,
    )
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=eval_llm,
        include_reason=False,
    )
    relevancy_metric = ContextualRelevancyMetric(
        threshold=0.5,
        model=eval_llm,
        include_reason=True,
    )
    precision_metric = ContextualPrecisionMetric(threshold=0.5, model=eval_llm)
    recall_metric = ContextualRecallMetric(threshold=0.5, model=eval_llm)

    # ------------------------------------------------------------------
    # Step 1: hit API for every golden question, collect retrieval results
    # ------------------------------------------------------------------
    all_questions: list[str] = []
    all_gt_answers: list[str] = []
    all_generated_answers: list[str] = []
    all_retrieved_documents: list[list[str]] = []
    pin_data: list[dict] = []
    passed_locators = 0
    api_failures = 0

    total = len(golden_questions)
    for idx, item in enumerate(golden_questions, start=1):
        question = item["question"]
        expected_context = item.get("expected_context", [])
        expected_locators = item.get("expected_chunk_locators", [])
        qid = item.get("id", idx)
        topic = item.get("topic", "n/a")
        safe_q = question.encode("ascii", "replace").decode("ascii")[:80]
        _print(f"[{idx}/{total}] Q{qid} ({topic}): {safe_q}...")

        t0 = time.time()
        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{rag_api_url}/rag/",
                    json={"messages": [{"role": "user", "content": question}]},
                )
            elapsed = time.time() - t0
            response.raise_for_status()
            payload = response.json()
            retrieved_docs = payload.get("documents", [])
            retrieved_contexts = [d["text"] for d in retrieved_docs]
            _print(f"    -> HTTP {response.status_code} in {elapsed:.2f}s, {len(retrieved_docs)} chunks")

            # Locator check (same as legacy, no DeepEval involved).
            try:
                assert_chunk_locators(
                    [
                        type(
                            "Doc", (), {"page_content": d["text"], "metadata": d.get("metadata", {})}
                        )()
                        for d in retrieved_docs
                    ],
                    expected_locators,
                    expected_context,
                )
                locator_ok = True
                passed_locators += 1
                _print("    -> locator/context match OK")
            except pytest.fail.Exception as exc:
                locator_ok = False
                if locator_strict:
                    raise
                _print(f"    -> [WARN] locator/context mismatch: {str(exc).splitlines()[0][:200]}")

            # Generate actual_output with NIM (the notebook needs it for
            # Correctness/Faithfulness). Short prompt so the answer stays
            # grounded in retrieved chunks.
            if skip_answer_gen:
                generated_answer = retrieved_contexts[0] if retrieved_contexts else ""
                _print("    -> answer gen SKIPPED (first-chunk fallback)")
            else:
                ctx_block = "\n".join(retrieved_contexts[: deepeval_top_k])
                answer_prompt = (
                    "You are a helpful assistant. Answer the user's question "
                    "using ONLY the context below. Be concise and factual.\n\n"
                    f"Context:\n{ctx_block}\n\n"
                    f"Question: {question}\n"
                    "Answer:"
                )
                try:
                    generated_answer = eval_llm.generate(answer_prompt).strip()
                    _print(f"    -> answer gen OK ({len(generated_answer)} chars)")
                except Exception as exc:
                    generated_answer = (
                        retrieved_contexts[0] if retrieved_contexts else ""
                    )
                    _print(
                        f"    -> [WARN] answer gen failed, using context chunk: "
                        f"{type(exc).__name__}"
                    )

            all_questions.append(question)
            all_gt_answers.append(expected_context[0] if expected_context else "")
            all_generated_answers.append(generated_answer)
            all_retrieved_documents.append(retrieved_contexts[:deepeval_top_k])
            pin_data.append({"qid": qid, "question": question[:120], "locator_ok": locator_ok})

        except Exception as exc:
            api_failures += 1
            _print(f"    -> [ERROR] {type(exc).__name__}: {_safe_msg(exc, 300)}")

    if api_failures == total:
        pytest.fail("All API calls failed — check RAG_API_URL and container health.")

    # ------------------------------------------------------------------
    # Step 2: batch DeepEval (notebook's evaluate(test_cases=[...], metrics=[...]))
    # ------------------------------------------------------------------
    test_cases = create_deep_eval_test_cases(
        questions=all_questions,
        gt_answers=all_gt_answers,
        generated_answers=all_generated_answers,
        retrieved_documents=all_retrieved_documents,
    )

    metrics = [
        correctness_metric,
        faithfulness_metric,
        relevancy_metric,
        precision_metric,
        recall_metric,
    ]

    _print(f"[evaluate] {len(test_cases)} test cases × {len(metrics)} metrics")
    results = evaluate(test_cases=test_cases, metrics=metrics)

    # Per-case summary: count metrics that passed per test case.
    case_pass_counts = []
    for tc_result in results.test_results:
        passed_metrics = sum(
            1 for m in tc_result.metrics_data if m is not None and m.score is not None and m.success
        )
        case_pass_counts.append(passed_metrics)

    ratio = (sum(1 for c in case_pass_counts if c == len(metrics))) / total if total else 0.0
    _print(
        f"[done] {sum(1 for c in case_pass_counts if c == len(metrics))}/{total} questions passed "
        f"all {len(metrics)} metrics ({ratio:.0%}), locators {passed_locators}/{total} OK, "
        f"api_failures={api_failures}, threshold={min_pass:.0%}"
    )

    if ratio < min_pass:
        pytest.fail(
            f"Only {ratio:.0%} questions passed all {len(metrics)} metrics "
            f"({len(metrics)} required). Ratio {ratio:.0%} < threshold {min_pass:.0%}. "
            "Check Qdrant collection, embedding server, and NVIDIA NIM key/quota."
        )
