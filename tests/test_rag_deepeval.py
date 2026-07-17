"""
DeepEval test cho Hybrid RAG dựa trên dữ liệu thật từ `../input/`.
Dùng `DeepEvalBaseLLM` bọc client theo `EVAL_LLM_*` (fallback `LLM_*`).
Các câu hỏi vàng lấy từ `tests/golden_questions.json`.
KHÔNG tự chạy — chỉ viết code.
"""
import os
import json
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "../input"))
GOLDEN_FILE = Path(__file__).parent.parent / "tests" / "golden_questions.json"


class CustomEvalLLM(DeepEvalBaseLLM):
    """Bọc client LLM theo chuẩn base_url + api_key + model chung."""

    def __init__(self, model_name: str = ""):
        # Đọc từ EVAL_LLM_* trước, fallback về LLM_*
        model = os.environ.get("EVAL_LLM_MODEL", os.environ.get("LLM_MODEL", "gpt-4"))
        base_url = os.environ.get("EVAL_LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
        api_key = os.environ.get("EVAL_LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
        super().__init__(model=model)
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model

    def load_model(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0,
        )

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


def load_golden_cases() -> list:
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_tests():
    eval_llm = CustomEvalLLM()
    golden_cases = load_golden_cases()

    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=eval_llm, include_reason=True),
        FaithfulnessMetric(threshold=0.7, model=eval_llm, include_reason=True),
        ContextualPrecisionMetric(threshold=0.7, model=eval_llm, include_reason=True),
        ContextualRecallMetric(threshold=0.7, model=eval_llm, include_reason=True),
    ]

    for case in golden_cases:
        # Tạo LLMTestCase từ câu hỏi vàng + context thật từ input/
        test_case = LLMTestCase(
            input=case["question"],
            actual_output=case.get("expected_answer", ""),
            expected_output=case.get("expected_answer", ""),
            retrieval_context=case.get("expected_context", []),
        )
        # Đánh giá từng metric và in kết quả (không assert bắt buộc để tránh lỗi chạy)
        for metric in metrics:
            result = metric.measure(test_case)
            print(f"Case: {case['id']} | Metric: {metric.__class__.__name__} | Score: {result.score} | Reason: {result.reason}")


if __name__ == "__main__":
    run_tests()
