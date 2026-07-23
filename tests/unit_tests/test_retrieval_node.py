from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


def _make_state(query_text: str = "query") -> dict[str, Any]:
    return {"messages": [HumanMessage(content=query_text)], "retry_count": 0}


def _make_config(collection: str | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {"metadata": {}}
    if collection is not None:
        cfg["metadata"]["collection_name"] = collection
    return cfg


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
    config = _make_config("coll-x")

    result = retrieve_documents(state, config, cfg=cfg)

    assert result["query"] == "hello"
    assert len(result["documents"]) == 1
    assert result["documents"][0].page_content == "doc1"
    mock_get_retriever.assert_called_once_with(
        k=cfg.retrieval_k, collection_name="coll-x", cfg=cfg
    )


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

    retrieve_documents(_make_state("q"), _make_config("c"), cfg=cfg)

    mock_get_reranker.assert_not_called()


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_applies_rerank_when_provider_bge(
    mock_get_retriever, mock_get_reranker
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="bge", rerank_top_k=2, rerank_model="stub")
    mock_get_retriever.return_value = _make_retriever_value(
        [Document(page_content="a"), Document(page_content="b"), Document(page_content="c")]
    )
    rerank_calls: list[tuple[list[Document], str]] = []

    def _fake_rerank(documents: list[Document], query: str) -> list[Document]:
        rerank_calls.append((documents, query))
        return documents[: cfg.rerank_top_k]

    mock_get_reranker.return_value = _fake_rerank

    result = retrieve_documents(_make_state("q"), _make_config("c"), cfg=cfg)

    assert len(rerank_calls) == 1
    assert rerank_calls[0][1] == "q"
    assert len(result["documents"]) == 2
    mock_get_reranker.assert_called_once_with(
        provider="bge",
        top_k=2,
        model_name="stub",
    )


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_no_documents_does_not_rerank(
    mock_get_retriever, mock_get_reranker
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="bge")
    mock_get_retriever.return_value = _make_retriever_value([])

    result = retrieve_documents(_make_state("q"), _make_config("c"), cfg=cfg)

    assert result["documents"] == []
    mock_get_reranker.assert_not_called()


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_falls_back_to_default_collection(
    mock_get_retriever, mock_get_reranker
) -> None:
    from agent.backend.nodes.retrieval import retrieve_documents
    from agent.utils.config import Config

    cfg = Config(rerank_provider="none", qdrant_collection_name="default-coll")
    mock_get_reranker.return_value = lambda d, _q: d
    mock_get_retriever.return_value = _make_retriever_value(
        [Document(page_content="x")]
    )

    state = _make_state("q")
    empty_config: dict[str, Any] = {"metadata": {}}

    retrieve_documents(state, empty_config, cfg=cfg)

    mock_get_retriever.assert_called_once_with(
        k=cfg.retrieval_k, collection_name="default-coll", cfg=cfg
    )


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
    retrieve_documents(state, _make_config("c"), cfg=cfg)

    _args, kwargs = mock_get_retriever.call_args
    assert kwargs["k"] == 99
