from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


def _make_state(query_text: str = "query", **extra: Any) -> dict[str, Any]:
    return {"messages": [HumanMessage(content=query_text)], "retry_count": 0, **extra}


def _llm_response(content: str) -> MagicMock:
    r = MagicMock()
    r.content = content
    return r


# ---------------------------------------------------------------------------
# transform_query node
# ---------------------------------------------------------------------------


def test_transform_query_disabled_returns_fallback() -> None:
    from agent.backend.nodes.query_transform import transform_query
    from agent.utils.config import Config

    cfg = Config(query_transform_enabled=False)
    result = transform_query(_make_state("câu hỏi gốc"), {}, cfg=cfg)

    assert result["original_query"] == "câu hỏi gốc"
    assert result["rewritten_query"] == "câu hỏi gốc"
    assert result["step_back_query"] == "câu hỏi gốc"
    assert result["sub_queries"] == ["câu hỏi gốc"]
    assert result["query"] == "câu hỏi gốc"


@patch("agent.backend.nodes.query_transform.ChatOpenAI")
@patch("agent.backend.nodes.query_transform.RunnableParallel")
def test_transform_query_enabled_uses_llm(mock_parallel: MagicMock, mock_chat: MagicMock) -> None:
    from agent.backend.nodes.query_transform import transform_query
    from agent.utils.config import Config

    combined = MagicMock()
    combined.invoke.return_value = {
        "rewrite": _llm_response("rewritten q"),
        "step_back": _llm_response("step-back q"),
        "decompose": _llm_response("Sub-queries:\n1. sub one\n2. sub two"),
    }
    mock_parallel.return_value = combined

    cfg = Config(query_transform_enabled=True)
    result = transform_query(_make_state("câu hỏi gốc"), {}, cfg=cfg)

    assert result["original_query"] == "câu hỏi gốc"
    assert result["rewritten_query"] == "rewritten q"
    assert result["step_back_query"] == "step-back q"
    assert result["sub_queries"] == ["1. sub one", "2. sub two"]
    # query downstream stays anchored to the original for reranking
    assert result["query"] == "câu hỏi gốc"


@patch("agent.backend.nodes.query_transform.ChatOpenAI")
def test_transform_query_llm_failure_falls_back(mock_chat: MagicMock) -> None:
    from agent.backend.nodes.query_transform import transform_query
    from agent.utils.config import Config

    mock_chat.side_effect = RuntimeError("LLM down")

    cfg = Config(query_transform_enabled=True)
    result = transform_query(_make_state("câu hỏi gốc"), {}, cfg=cfg)

    assert result["original_query"] == "câu hỏi gốc"
    assert result["rewritten_query"] == "câu hỏi gốc"
    assert result["sub_queries"] == ["câu hỏi gốc"]


def test_parse_sub_queries_filters_header_and_blanks() -> None:
    from agent.backend.nodes.query_transform import _parse_sub_queries

    raw = "Sub-queries:\n1. aaa\n\n2. bbb\nSub-queries: trailing\n3. ccc"
    assert _parse_sub_queries(raw) == ["1. aaa", "2. bbb", "3. ccc"]


# ---------------------------------------------------------------------------
# retrieval node — multi-query behaviour
# ---------------------------------------------------------------------------


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieval_single_query_when_transform_disabled(
    mock_get_retriever: MagicMock, mock_get_reranker: MagicMock
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(query_transform_enabled=False, rerank_provider="none")
    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="d1")]
    mock_get_retriever.return_value = retriever

    retrieve_documents(_make_state("q"), {}, cfg=cfg)

    assert retriever.invoke.call_count == 1
    retriever.invoke.assert_called_once_with("q")


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieval_multi_query_dedupes_results(
    mock_get_retriever: MagicMock, mock_get_reranker: MagicMock
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(query_transform_enabled=True, rerank_provider="none")
    # Rewritten + step-back + 2 sub-queries = 4 unique non-original queries.
    state = _make_state(
        "original q",
        rewritten_query="rewritten q",
        step_back_query="step-back q",
        sub_queries=["sub 1", "sub 2"],
    )

    shared = Document(page_content="shared", metadata={"global_id": "g1"})
    unique_doc = Document(page_content="unique", metadata={"global_id": "g2"})

    retriever = MagicMock()
    # Each query returns `shared`; only the original query also returns unique_doc.
    retriever.invoke.side_effect = lambda q: [shared, unique_doc] if q == "original q" else [shared]
    mock_get_retriever.return_value = retriever

    result = retrieve_documents(state, {}, cfg=cfg)

    # 5 distinct queries (original + 4 unique variants).
    assert retriever.invoke.call_count == 5
    # Deduped: g1 + g2 only, even though g1 was seen 5x.
    ids = {d.metadata.get("global_id") for d in result["documents"]}
    assert ids == {"g1", "g2"}
    assert len(result["documents"]) == 2


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieval_rerank_uses_original_query(
    mock_get_retriever: MagicMock, mock_get_reranker: MagicMock
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(query_transform_enabled=True, rerank_provider="bge", rerank_top_k=3)
    rerank_fn = MagicMock(side_effect=lambda docs, _q: docs)
    mock_get_reranker.return_value = rerank_fn

    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="d")]
    mock_get_retriever.return_value = retriever

    state = _make_state(
        "original q",
        rewritten_query="rewritten",
        step_back_query="step",
        sub_queries=["sub"],
    )

    result = retrieve_documents(state, {}, cfg=cfg)

    assert result["query"] == "original q"
    rerank_fn.assert_called_once()
    # Ensure reranker received the ORIGINAL query, not a transformed one.
    assert rerank_fn.call_args[0][1] == "original q"


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieval_legacy_when_no_transform_state(
    mock_get_retriever: MagicMock, mock_get_reranker: MagicMock
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(query_transform_enabled=False, rerank_provider="none")
    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="d")]
    mock_get_retriever.return_value = retriever

    result = retrieve_documents({"messages": [HumanMessage(content="old"), AIMessage(content="resp"), HumanMessage(content="plain")], "retry_count": 0}, {}, cfg=cfg)

    retriever.invoke.assert_called_once_with("plain")
    assert result["query"] == "plain"
