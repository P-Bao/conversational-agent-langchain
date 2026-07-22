"""Tests for the reranker utility module."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from agent.utils.reranker import get_reranker, rerank_with_cohere, rerank_with_flashrank


def test_get_reranker_none():
    """Test get_reranker with 'none' provider returns passthrough."""
    reranker = get_reranker(provider="none", top_k=3)
    docs = [Document(page_content=f"doc{i}") for i in range(5)]
    result = reranker(docs, "query")
    assert len(result) == 3
    assert result[0].page_content == "doc0"


def test_get_reranker_none_fewer_docs():
    """Test passthrough doesn't truncate when fewer docs than top_k."""
    reranker = get_reranker(provider="none", top_k=5)
    docs = [Document(page_content=f"doc{i}") for i in range(2)]
    result = reranker(docs, "query")
    assert len(result) == 2


def test_get_reranker_cohere_missing_key():
    """Test get_reranker raises error when Cohere key is missing."""
    with pytest.raises(ValueError, match="Cohere API key is required"):
        get_reranker(provider="cohere", top_k=3, cohere_api_key=None)


def test_get_reranker_invalid_provider():
    """Test get_reranker raises error for unknown provider."""
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        get_reranker(provider="invalid", top_k=3)


@patch("langchain_cohere.CohereRerank")
def test_rerank_with_cohere(mock_cohere_rerank):
    """Test rerank_with_cohere calls CohereRerank correctly."""
    mock_reranker_instance = MagicMock()
    mock_cohere_rerank.return_value = mock_reranker_instance

    docs = [Document(page_content="doc1"), Document(page_content="doc2")]
    mock_reranker_instance.compress_documents.return_value = [docs[1], docs[0]]

    result = rerank_with_cohere(docs, "query", top_k=2, api_key="test_key")

    mock_cohere_rerank.assert_called_once_with(model="rerank-v3.5", cohere_api_key="test_key", top_n=2)
    mock_reranker_instance.compress_documents.assert_called_once_with(documents=docs, query="query")
    assert len(result) == 2


def test_rerank_with_cohere_empty_docs():
    """Test rerank_with_cohere handles empty documents."""
    result = rerank_with_cohere([], "query", top_k=3, api_key="test_key")
    assert result == []


@patch("flashrank.Ranker")
@patch("flashrank.RerankRequest")
def test_rerank_with_flashrank(mock_rerank_request, mock_ranker):
    """Test rerank_with_flashrank calls FlashRank correctly."""
    mock_ranker_instance = MagicMock()
    mock_ranker.return_value = mock_ranker_instance

    docs = [Document(page_content="doc1"), Document(page_content="doc2")]
    # Simulate flashrank returning results with scores
    mock_ranker_instance.rerank.return_value = [
        {"id": 1, "score": 0.9, "text": "doc2"},
        {"id": 0, "score": 0.5, "text": "doc1"},
    ]

    result = rerank_with_flashrank(docs, "query", top_k=2)

    mock_ranker.assert_called_once()
    assert len(result) == 2
    # Higher score doc should be first
    assert result[0].page_content == "doc2"
    assert result[1].page_content == "doc1"


def test_rerank_with_flashrank_empty_docs():
    """Test rerank_with_flashrank handles empty documents."""
    result = rerank_with_flashrank([], "query", top_k=3)
    assert result == []


@patch("agent.utils.reranker._get_bge_reranker")
def test_rerank_with_bge(mock_get_bge_reranker):
    """Test rerank_with_bge calls FlagReranker and sorts by score."""
    from agent.utils.reranker import rerank_with_bge

    mock_reranker = MagicMock()
    mock_get_bge_reranker.return_value = mock_reranker
    mock_reranker.compute_score.return_value = [0.2, 0.9, 0.5]

    docs = [
        Document(page_content="doc0"),
        Document(page_content="doc1"),
        Document(page_content="doc2"),
    ]

    result = rerank_with_bge(docs, "query", top_k=2)

    assert len(result) == 2
    # Highest score (doc1, score 0.9) should be first, then doc2 (0.5)
    assert result[0].page_content == "doc1"
    assert result[1].page_content == "doc2"


@patch("agent.utils.reranker._get_bge_reranker")
def test_rerank_preserves_order(mock_get_bge_reranker):
    """Test rerank preserves exact sorted order without shuffling."""
    from agent.utils.reranker import rerank_with_bge

    mock_reranker = MagicMock()
    mock_get_bge_reranker.return_value = mock_reranker
    mock_reranker.compute_score.return_value = [0.1, 0.8, 0.6, 0.9]

    docs = [
        Document(page_content="d0"),
        Document(page_content="d1"),
        Document(page_content="d2"),
        Document(page_content="d3"),
    ]

    result = rerank_with_bge(docs, "q", top_k=4)
    contents = [d.page_content for d in result]
    assert contents == ["d3", "d1", "d2", "d0"]


def test_get_reranker_bge():
    """Test get_reranker with default 'bge' provider."""
    reranker_fn = get_reranker(provider="bge", top_k=2)
    assert callable(reranker_fn)
