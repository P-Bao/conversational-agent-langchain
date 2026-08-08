from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


def _make_state(query_text: str = "query") -> dict[str, Any]:
    return {"messages": [HumanMessage(content=query_text)], "retry_count": 0}


def _make_retriever_value(docs: list[Document]) -> MagicMock:
    r = MagicMock()
    r.invoke.return_value = docs
    return r


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_uses_default_retriever(mock_get_retriever, mock_get_reranker) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="none", retrieval_k=42)
    mock_get_reranker.return_value = lambda docs, _q: docs
    mock_get_retriever.return_value = _make_retriever_value(
        [Document(page_content="doc1", metadata={"source": "test.pdf"})]
    )

    state = _make_state("hello")

    result = retrieve_documents(state, {}, cfg=cfg)

    assert result["query"] == "hello"
    assert len(result["documents"]) == 1
    assert result["documents"][0].page_content == "doc1"
    mock_get_retriever.assert_called_once_with(k=cfg.retrieval_k, cfg=cfg)


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_skips_rerank_when_provider_none(
    mock_get_retriever, mock_get_reranker
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="none")
    retriever = _make_retriever_value(
        [Document(page_content="a"), Document(page_content="b")]
    )
    mock_get_retriever.return_value = retriever

    retrieve_documents(_make_state("q"), {}, cfg=cfg)

    mock_get_reranker.assert_not_called()


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_no_documents_does_not_rerank(
    mock_get_retriever, mock_get_reranker
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="bge")
    mock_get_retriever.return_value = _make_retriever_value([])

    result = retrieve_documents(_make_state("q"), {}, cfg=cfg)

    assert result["documents"] == []
    mock_get_reranker.assert_not_called()


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_uses_configured_collection(
    mock_get_retriever, mock_get_reranker
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="none", qdrant_collection_name="my-coll")
    mock_get_reranker.return_value = lambda d, _q: d
    mock_get_retriever.return_value = _make_retriever_value(
        [Document(page_content="x")]
    )

    state = _make_state("q")

    retrieve_documents(state, {}, cfg=cfg)

    mock_get_retriever.assert_called_once_with(k=cfg.retrieval_k, cfg=cfg)


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_uses_retry_k_after_first_attempt(
    mock_get_retriever, mock_get_reranker
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(
        rerank_provider="none",
        retrieval_k=10,
        retrieval_k_retry=99,
    )
    mock_get_reranker.return_value = lambda d, _q: d
    mock_get_retriever.return_value = _make_retriever_value([Document(page_content="x")])

    state = {**_make_state("q"), "retry_count": 1}
    retrieve_documents(state, {}, cfg=cfg)

    _args, kwargs = mock_get_retriever.call_args
    assert kwargs["k"] == 99


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_top_k_from_state_overrides_cfg(
    mock_get_retriever, mock_get_reranker
) -> None:
    """``state["top_k"]`` overrides ``cfg.rerank_top_k`` when calling get_reranker."""
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="bge", rerank_top_k=5)
    mock_get_reranker.return_value = lambda d, _q: d
    mock_get_retriever.return_value = _make_retriever_value([Document(page_content="x")])

    state = {**_make_state("q"), "top_k": 7}
    retrieve_documents(state, {}, cfg=cfg)

    mock_get_reranker.assert_called_once()
    call_kwargs = mock_get_reranker.call_args[1]
    assert call_kwargs["top_k"] == 7


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_top_k_falls_back_to_cfg(
    mock_get_retriever, mock_get_reranker
) -> None:
    """When ``top_k`` is absent from state, ``cfg.rerank_top_k`` is used."""
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="bge", rerank_top_k=11)
    mock_get_reranker.return_value = lambda d, _q: d
    mock_get_retriever.return_value = _make_retriever_value([Document(page_content="x")])

    # top_k explicitly None — should fall back via `or cfg.rerank_top_k`
    state = {**_make_state("q"), "top_k": None}
    retrieve_documents(state, {}, cfg=cfg)

    mock_get_reranker.assert_called_once()
    call_kwargs = mock_get_reranker.call_args[1]
    assert call_kwargs["top_k"] == 11


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_top_k_missing_from_state_falls_back_to_cfg(
    mock_get_retriever, mock_get_reranker
) -> None:
    """Legacy states without the ``top_k`` key still work; cfg fallback applies."""
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="bge", rerank_top_k=13)
    mock_get_reranker.return_value = lambda d, _q: d
    mock_get_retriever.return_value = _make_retriever_value([Document(page_content="x")])

    # No "top_k" key at all
    state = _make_state("q")
    retrieve_documents(state, {}, cfg=cfg)

    mock_get_reranker.assert_called_once()
    call_kwargs = mock_get_reranker.call_args[1]
    assert call_kwargs["top_k"] == 13
