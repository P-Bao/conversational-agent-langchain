from __future__ import annotations

from unittest.mock import patch

import pytest
from agent.utils import reranker as reranker_module
from agent.utils.reranker import get_reranker, rerank_with_bge
from langchain_core.documents import Document


def test_get_reranker_none_passthrough_truncates_to_top_k() -> None:
    fn = get_reranker(provider="none", top_k=3)
    docs = [Document(page_content=f"doc{i}") for i in range(5)]
    result = fn(docs, "query")
    assert len(result) == 3
    assert [d.page_content for d in result] == ["doc0", "doc1", "doc2"]


def test_get_reranker_none_fewer_docs_than_top_k() -> None:
    fn = get_reranker(provider="none", top_k=5)
    docs = [Document(page_content="a"), Document(page_content="b")]
    result = fn(docs, "query")
    assert len(result) == 2


def test_get_reranker_bge_returns_callable() -> None:
    fn = get_reranker(provider="bge", top_k=2)
    assert callable(fn)


def test_get_reranker_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        get_reranker(provider="invalid", top_k=2)


def test_get_reranker_cohere_removed() -> None:
    with pytest.raises(ValueError):
        get_reranker(provider="cohere", top_k=2)


def test_get_reranker_flashrank_removed() -> None:
    with pytest.raises(ValueError):
        get_reranker(provider="flashrank", top_k=2)


def test_get_reranker_normalizes_provider_case() -> None:
    fn = get_reranker(provider="NONE", top_k=1)
    docs = [Document(page_content="a"), Document(page_content="b")]
    assert fn(docs, "q") == [Document(page_content="a")]


@patch("agent.utils.reranker._get_bge_reranker")
def test_rerank_with_bge_sorts_descending(mock_get_bge_reranker) -> None:
    mock_reranker = mock_get_bge_reranker.return_value
    mock_reranker.compute_score.return_value = [0.2, 0.9, 0.5]

    docs = [
        Document(page_content="doc0"),
        Document(page_content="doc1"),
        Document(page_content="doc2"),
    ]
    result = rerank_with_bge(docs, "query", top_k=2)

    assert [d.page_content for d in result] == ["doc1", "doc2"]


@patch("agent.utils.reranker._get_bge_reranker")
def test_rerank_with_bge_empty_input_returns_empty(mock_get_bge_reranker) -> None:
    assert rerank_with_bge([], "query", top_k=3) == []
    mock_get_bge_reranker.assert_not_called()


@patch("agent.utils.reranker._get_bge_reranker")
def test_get_reranker_bge_calls_compute_score(mock_get_bge_reranker) -> None:
    mock_reranker = mock_get_bge_reranker.return_value
    mock_reranker.compute_score.return_value = [0.7, 0.3]

    fn = get_reranker(provider="bge", top_k=2, model_name="custom/model")
    docs = [Document(page_content="x"), Document(page_content="y")]

    result = fn(docs, "q")

    assert len(result) == 2
    mock_get_bge_reranker.assert_called_once_with("custom/model")
    args, _ = mock_reranker.compute_score.call_args
    assert args[0] == [["q", "x"], ["q", "y"]]
