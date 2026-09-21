"""Unit tests for observability: Prometheus metrics + OTel tracing wrapper."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY

from agent.utils import observability
from agent.utils.observability import TracedRetriever, wrap_retriever

pytestmark = pytest.mark.anyio


class _StubRetriever(BaseRetriever):
    """Real BaseRetriever subclass so TracedRetriever (pydantic model) accepts it."""

    docs: list[Document] = [Document(page_content="doc1", metadata={"source": "a.pdf"})]
    fail: bool = False
    search_kwargs: dict[str, Any] = {}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.fail:
            raise RuntimeError("boom")
        return self.docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.fail:
            raise RuntimeError("boom")
        return self.docs


def _value(name: str, labels: dict[str, str] | None = None) -> float | None:
    return REGISTRY.get_sample_value(name, labels or {})


@pytest.fixture
def in_flight_zero() -> Any:
    """Ensure in-flight gauge is 0 before/after each test (no leak)."""
    yield
    assert _value("rag_retrieval_requests_in_flight") == 0


def _patch_tracer(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Swap the module tracer with one backed by an in-memory exporter."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "tracer", provider.get_tracer("test"))
    return exporter


def _today_hcm() -> str:
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d")


def _this_hour_hcm() -> str:
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%H")


def test_wrap_retriever_sync_metrics(in_flight_zero: None) -> None:
    retriever = wrap_retriever(_StubRetriever())
    before_total = _value("rag_retrieval_requests_total")
    before_day = _value("rag_retrieval_requests_by_day_total", {"day": _today_hcm()})
    before_hod = _value("rag_retrieval_requests_by_hour_of_day_total", {"hod": _this_hour_hcm()})
    before_day_hour = _value(
        "rag_retrieval_requests_by_day_hour_total",
        {"day": _today_hcm(), "hod": _this_hour_hcm()},
    )

    result = retriever.invoke("hello world")

    assert [d.page_content for d in result] == ["doc1"]
    assert _value("rag_retrieval_requests_total") == (before_total or 0) + 1
    assert _value("rag_retrieval_requests_by_day_total", {"day": _today_hcm()}) == (before_day or 0) + 1
    assert _value(
        "rag_retrieval_requests_by_hour_of_day_total", {"hod": _this_hour_hcm()}
    ) == (before_hod or 0) + 1
    assert _value(
        "rag_retrieval_requests_by_day_hour_total",
        {"day": _today_hcm(), "hod": _this_hour_hcm()},
    ) == (before_day_hour or 0) + 1
    duration_count = _value("rag_retrieval_duration_seconds_count")
    assert duration_count is not None and duration_count >= 1
    docs_count = _value("rag_retrieval_docs_returned_count")
    assert docs_count is not None and docs_count >= 1


async def test_wrap_retriever_async_metrics(in_flight_zero: None) -> None:
    retriever = wrap_retriever(_StubRetriever())
    before_total = _value("rag_retrieval_requests_total")

    result = await retriever.ainvoke("hello async")

    assert [d.page_content for d in result] == ["doc1"]
    assert _value("rag_retrieval_requests_total") == (before_total or 0) + 1


async def test_error_increments_errors_and_no_in_flight_leak(in_flight_zero: None) -> None:
    before_errors = _value("rag_retrieval_errors_total")
    retriever = wrap_retriever(_StubRetriever(fail=True))

    with pytest.raises(RuntimeError, match="boom"):
        retriever.invoke("boom query")

    assert _value("rag_retrieval_errors_total") == (before_errors or 0) + 1
    assert _value("rag_retrieval_requests_in_flight") == 0

    with pytest.raises(RuntimeError, match="boom"):
        await retriever.ainvoke("boom query")

    assert _value("rag_retrieval_errors_total") == (before_errors or 0) + 2
    assert _value("rag_retrieval_requests_in_flight") == 0


async def test_tracing_span_contains_query_and_full_docs(
    monkeypatch: pytest.MonkeyPatch, in_flight_zero: None
) -> None:
    exporter = _patch_tracer(monkeypatch)
    retriever = wrap_retriever(_StubRetriever())

    retriever.invoke("what is the capital of France?")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "rag.retrieval"
    assert span.attributes is not None
    assert span.attributes["input.query"] == "what is the capital of France?"
    assert span.attributes["input.retriever_name"] == "_StubRetriever"
    assert span.attributes["output.num_docs"] == 1
    events = [e for e in span.events if e.name == "retrieval.output"]
    assert len(events) == 1
    assert events[0].attributes is not None
    docs_json = events[0].attributes["documents_json"]
    assert isinstance(docs_json, str)
    payload = json.loads(docs_json)
    assert payload[0]["page_content"] == "doc1"
    assert payload[0]["metadata"] == {"source": "a.pdf"}
    assert payload[0]["score"] is None


async def test_tracing_async_span(monkeypatch: pytest.MonkeyPatch, in_flight_zero: None) -> None:
    exporter = _patch_tracer(monkeypatch)
    retriever = wrap_retriever(_StubRetriever())

    await retriever.ainvoke("async query")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes is not None
    assert spans[0].attributes["input.query"] == "async query"


async def test_tracing_error_span(monkeypatch: pytest.MonkeyPatch, in_flight_zero: None) -> None:
    exporter = _patch_tracer(monkeypatch)
    retriever = wrap_retriever(_StubRetriever(fail=True))

    with pytest.raises(RuntimeError, match="boom"):
        retriever.invoke("boom query")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.is_ok is False


def test_wrap_retriever_idempotent() -> None:
    inner = _StubRetriever()
    wrapped = wrap_retriever(inner)
    assert isinstance(wrapped, TracedRetriever)
    assert wrap_retriever(wrapped) is wrapped


def test_config_exposes_search_kwargs() -> None:
    inner = _StubRetriever(search_kwargs={"k": 7})
    wrapped = wrap_retriever(inner)
    assert wrapped.config == {"k": 7}


def test_docs_payload_not_truncated_by_default(
    monkeypatch: pytest.MonkeyPatch, in_flight_zero: None
) -> None:
    """Default limit (2MB) must keep real-sized payloads fully intact."""
    exporter = _patch_tracer(monkeypatch)
    big_docs = [
        Document(page_content="x" * 5000 * (i % 4 + 1), metadata={"i": i})
        for i in range(10)
    ]
    retriever = wrap_retriever(_StubRetriever(docs=big_docs))

    retriever.invoke("big")

    span = exporter.get_finished_spans()[0]
    event = next(e for e in span.events if e.name == "retrieval.output")
    assert event.attributes is not None
    docs_json = event.attributes["documents_json"]
    assert isinstance(docs_json, str)
    assert "...[truncated]" not in docs_json
    payload = json.loads(docs_json)
    assert len(payload) == 10
    for i, item in enumerate(payload):
        assert item["page_content"] == "x" * 5000 * (i % 4 + 1)
        assert item["metadata"] == {"i": i}


def test_docs_payload_truncated_in_event(
    monkeypatch: pytest.MonkeyPatch, in_flight_zero: None
) -> None:
    """Safety net: when the payload exceeds the configured limit (e.g. it would
    blow past the ~4MB OTLP request limit), the event stays under the limit."""
    exporter = _patch_tracer(monkeypatch)
    monkeypatch.setattr(observability, "_MAX_EVENT_ATTR_CHARS", 8000)
    big_docs = [Document(page_content="x" * 5000, metadata={"i": i}) for i in range(10)]
    retriever = wrap_retriever(_StubRetriever(docs=big_docs))

    retriever.invoke("big")

    span = exporter.get_finished_spans()[0]
    event = next(e for e in span.events if e.name == "retrieval.output")
    assert event.attributes is not None
    docs_json = event.attributes["documents_json"]
    assert isinstance(docs_json, str)
    assert len(docs_json) <= 8000
    payload = json.loads(docs_json)
    assert payload[0]["page_content"].endswith("...[truncated]")
    assert len(payload[0]["page_content"]) < 5000
    assert payload[0]["metadata"] == {"i": 0}


def test_docs_payload_truncation_disabled_with_zero_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TRACE_OUTPUT_MAX_LEN=0 disables truncation entirely (no safety net)."""
    payload = [{"page_content": "y" * 5000, "metadata": {}, "score": None}] * 10
    docs_json = observability._truncate_documents_json(payload, max_chars=0)
    assert "...[truncated]" not in docs_json
    assert len(json.loads(docs_json)) == 10


def test_parse_max_event_attr_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACE_OUTPUT_MAX_LEN", raising=False)
    assert observability._parse_max_event_attr_chars() == 2_000_000
    monkeypatch.setenv("TRACE_OUTPUT_MAX_LEN", "12345")
    assert observability._parse_max_event_attr_chars() == 12345
    monkeypatch.setenv("TRACE_OUTPUT_MAX_LEN", "0")
    assert observability._parse_max_event_attr_chars() == 0
    monkeypatch.setenv("TRACE_OUTPUT_MAX_LEN", "not-a-number")
    assert observability._parse_max_event_attr_chars() == 2_000_000


def test_init_telemetry_no_endpoint_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with patch("agent.utils.observability.OTLPSpanExporter") as mock_exporter:
        observability.init_telemetry()
    mock_exporter.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://host.docker.internal:30749", "http://host.docker.internal:30749/v1/traces"),
        ("http://host.docker.internal:30749/", "http://host.docker.internal:30749/v1/traces"),
        (
            "http://host.docker.internal:30749/v1/traces",
            "http://host.docker.internal:30749/v1/traces",
        ),
    ],
)
def test_init_telemetry_passes_full_trace_url(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", raw)
    with patch("agent.utils.observability.OTLPSpanExporter") as mock_exporter:
        observability.init_telemetry()
    mock_exporter.assert_called_once_with(endpoint=expected)
