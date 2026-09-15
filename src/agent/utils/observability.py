"""Observability utilities: Prometheus metrics + OpenTelemetry tracing.

Two separate layers, initialized once:

- **Metrics (Prometheus)**: aggregated counters/histograms with the
  ``rag_retrieval_`` prefix, scraped via ``/metrics`` and rendered in Grafana.
  Day/hour-of-day labels are computed in a fixed timezone
  (``Asia/Ho_Chi_Minh``) — the Grafana dashboard MUST set the same fixed
  timezone instead of ``browser``, otherwise panels drift for viewers in
  other timezones (lesson learned from the earlier chirp3 dashboard).
- **Tracing (OpenTelemetry, OTLP export)**: one span per retrieval with the
  verbatim query and the FULL document list (page_content + metadata +
  score). Large payloads are also written to a structured log line with the
  ``trace_id`` so the full content can be joined in Grafana
  (trace-to-logs) instead of being silently truncated in span attributes.

``TracedRetriever`` wraps any ``BaseRetriever`` and instruments both the
sync and async paths, so every retrieval entry point (``/rag/``,
``/rag/stream``, ``/semantic/search``) is covered automatically.
"""

import json
import os
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode
from prometheus_client import Counter, Gauge, Histogram

DASHBOARD_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "rag-retrieval")

# --- Prometheus metrics (registry: default) --------------------------------

REQUESTS_TOTAL = Counter(
    "rag_retrieval_requests_total",
    "Total number of retrieval requests.",
)
REQUESTS_BY_DAY = Counter(
    "rag_retrieval_requests_by_day",
    "Retrieval requests per calendar day (Asia/Ho_Chi_Minh). Dashboard timezone MUST match.",
    ["day"],
)
REQUESTS_BY_HOUR_OF_DAY = Counter(
    "rag_retrieval_requests_by_hour_of_day",
    "Retrieval requests per hour of day (Asia/Ho_Chi_Minh). Dashboard timezone MUST match.",
    ["hod"],
)
REQUESTS_BY_DAY_HOUR = Counter(
    "rag_retrieval_requests_by_day_hour",
    "Retrieval requests per day+hour combination (Asia/Ho_Chi_Minh). "
    "Allows drilling down into a specific day's hourly pattern "
    "(e.g. one day picker + hour-of-day bars on Grafana).",
    ["day", "hod"],
)
ERRORS_TOTAL = Counter(
    "rag_retrieval_errors_total",
    "Total number of failed retrieval requests.",
)
DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    "Full round-trip retrieval duration in seconds (includes query embedding).",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
DOCS_RETURNED = Histogram(
    "rag_retrieval_docs_returned",
    "Number of documents returned per retrieval request.",
    buckets=(0, 1, 2, 3, 5, 8, 10, 15, 20),
)
REQUESTS_IN_FLIGHT = Gauge(
    "rag_retrieval_requests_in_flight",
    "Retrieval requests currently being processed.",
)

# --- OpenTelemetry init (single shared init for tracing) -------------------


def init_telemetry() -> None:
    """Initialize the OTel TracerProvider once.

    Export via OTLP/HTTP to the endpoint in ``OTEL_EXPORTER_OTLP_ENDPOINT``
    (Tempo or an OTel Collector in the k8s cluster). When the endpoint is
    not set (local dev without a backend), no exporter is registered and
    spans become no-ops.
    """
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        # The OTLP HTTP exporter appends /v1/traces to the endpoint. Strip a
        # trailing /v1/traces if present to avoid a double path (404 Not Found).
        base = endpoint.rstrip("/").removesuffix("/v1/traces")
        exporter = OTLPSpanExporter(endpoint=base)
        # The exporter's requests.Session honors HTTP_PROXY/http_proxy env
        # vars (trust_env=True) and routes the export through a corporate
        # proxy, which returns 404 for host.docker.internal/NodePort targets.
        # Tempo is always reachable directly from inside the cluster network.
        session = getattr(exporter, "_session", None)
        if session is not None:
            session.trust_env = False
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(f"OTel tracing enabled, exporting OTLP/HTTP to {base}")
    else:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing is a no-op")
    trace.set_tracer_provider(provider)


tracer = trace.get_tracer("rag-retrieval")

# Payload safety: span attributes may be truncated by OTel backends. Keep the
# full document payload in the structured log (joined by trace_id) and only
# put a safely truncated snapshot into the span event.
_MAX_EVENT_ATTR_CHARS = 8000


def _now_hcm() -> datetime:
    return datetime.now(DASHBOARD_TIMEZONE)


def _record_request_counters() -> None:
    REQUESTS_TOTAL.inc()
    now = _now_hcm()
    day = now.strftime("%Y-%m-%d")
    hod = now.strftime("%H")
    REQUESTS_BY_DAY.labels(day=day).inc()
    REQUESTS_BY_HOUR_OF_DAY.labels(hod=hod).inc()
    REQUESTS_BY_DAY_HOUR.labels(day=day, hod=hod).inc()


def _docs_payload(documents: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "score": doc.metadata.get("score"),
        }
        for doc in documents
    ]


def _truncate_documents_json(payload: list[dict[str, Any]]) -> str:
    """Serialize the document payload keeping valid JSON under the size limit.

    Shrinks ``page_content`` progressively (per-doc) instead of slicing the
    JSON string mid-token, so the attribute always parses.
    """
    docs_json = json.dumps(payload, ensure_ascii=False)
    if len(docs_json) <= _MAX_EVENT_ATTR_CHARS:
        return docs_json
    for per_doc in (2000, 1000, 500, 200, 100):
        shrunk = [
            {
                **doc,
                "page_content": doc["page_content"][:per_doc] + "...[truncated]",
            }
            if len(doc["page_content"]) > per_doc
            else doc
            for doc in payload
        ]
        docs_json = json.dumps(shrunk, ensure_ascii=False)
        if len(docs_json) <= _MAX_EVENT_ATTR_CHARS:
            return docs_json
    # Final fallback: only the first doc, heavily truncated.
    first = payload[0]
    return json.dumps(
        [{**first, "page_content": first["page_content"][:100] + "...[truncated]"}],
        ensure_ascii=False,
    )


def _record_success(
    span: Span,
    documents: list[Document],
    elapsed: float,
) -> None:
    """Record shared success bookkeeping: metrics, span output, structured log."""
    DURATION_SECONDS.observe(elapsed)
    DOCS_RETURNED.observe(len(documents))
    span.set_attribute("output.num_docs", len(documents))
    payload = _docs_payload(documents)
    span.add_event(
        "retrieval.output",
        attributes={"documents_json": _truncate_documents_json(payload)},
    )
    span.set_status(Status(StatusCode.OK))
    span_id = format(span.get_span_context().span_id, "016x")
    logger.bind(
        trace_id=format(span.get_span_context().trace_id, "032x"),
        span_id=span_id,
        documents=payload,
    ).info(f"rag_retrieval_output num_docs={len(documents)} duration={elapsed:.4f}s")


def _record_error(span: Span, error: BaseException, elapsed: float) -> None:
    ERRORS_TOTAL.inc()
    DURATION_SECONDS.observe(elapsed)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


class TracedRetriever(BaseRetriever):
    """Wrapper retriever adding Prometheus metrics + OTel tracing.

    Delegates to an inner retriever and instruments both paths:

    - ``rag_retrieval_requests_in_flight`` is incremented at the start and
      decremented in ``finally`` (no leak on exceptions).
    - ``rag_retrieval_duration_seconds`` measures the full round-trip,
      including the query embedding step performed by the inner retriever.
    - A ``rag.retrieval`` span captures the verbatim query and the full
      document list (page_content + metadata + score).
    """

    inner: Any  # BaseRetriever in production; Any keeps duck-typed mocks usable in tests

    @property
    def config(self) -> dict[str, Any]:
        """Return the inner retriever's search kwargs (e.g. top_k)."""
        return dict(getattr(self.inner, "search_kwargs", {}) or {})

    def _start_span(self, query: str) -> Span:
        attributes: dict[str, str | int] = {
            "input.query": query,
            "input.retriever_name": self.inner.__class__.__name__,
        }
        top_k = self.config.get("k")
        if top_k is not None:
            attributes["input.top_k"] = top_k
        return tracer.start_span("rag.retrieval", attributes=attributes)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[Document]:
        _record_request_counters()
        REQUESTS_IN_FLIGHT.inc()
        span = self._start_span(query)
        start = time.perf_counter()
        try:
            documents = self.inner.invoke(query)
        except BaseException as exc:
            _record_error(span, exc, time.perf_counter() - start)
            raise
        else:
            _record_success(span, documents, time.perf_counter() - start)
            return documents
        finally:
            REQUESTS_IN_FLIGHT.dec()
            span.end()

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[Document]:
        _record_request_counters()
        REQUESTS_IN_FLIGHT.inc()
        span = self._start_span(query)
        start = time.perf_counter()
        try:
            documents = await self.inner.ainvoke(query)
        except BaseException as exc:
            _record_error(span, exc, time.perf_counter() - start)
            raise
        else:
            _record_success(span, documents, time.perf_counter() - start)
            return documents
        finally:
            REQUESTS_IN_FLIGHT.dec()
            span.end()


def wrap_retriever(retriever: BaseRetriever) -> TracedRetriever:
    """Wrap a retriever with metrics + tracing (idempotent)."""
    if isinstance(retriever, TracedRetriever):
        return retriever
    return TracedRetriever(inner=retriever)
