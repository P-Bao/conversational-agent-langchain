from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from agent.utils import reranker as reranker_module
from agent.utils.reranker import get_reranker, rerank_with_remote


def _cfg(provider: str = "none", base_url: str = "", top_k: int = 3) -> MagicMock:
    return MagicMock(
        rerank_provider=provider,
        rerank_base_url=base_url,
        rerank_top_k=top_k,
    )


def test_get_reranker_none_passthrough_truncates_to_top_k() -> None:
    fn = get_reranker(_cfg(top_k=3))
    docs = [Document(page_content=f"doc{i}") for i in range(5)]
    result = fn(docs, "query")
    assert len(result) == 3
    assert [d.page_content for d in result] == ["doc0", "doc1", "doc2"]


def test_get_reranker_none_fewer_docs_than_top_k() -> None:
    fn = get_reranker(_cfg(top_k=5))
    docs = [Document(page_content="a"), Document(page_content="b")]
    result = fn(docs, "query")
    assert len(result) == 2


def test_get_reranker_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        get_reranker(_cfg(provider="invalid"))


def test_get_reranker_normalizes_provider_case() -> None:
    fn = get_reranker(_cfg(provider="NONE", top_k=1))
    docs = [Document(page_content="a"), Document(page_content="b")]
    out = fn(docs, "q")
    assert len(out) == 1
    assert out[0].page_content == "a"


def test_get_reranker_remote_missing_base_url_raises() -> None:
    with pytest.raises(ValueError, match="RERANK_BASE_URL is required"):
        get_reranker(_cfg(provider="remote", base_url=""))


@patch("agent.utils.reranker.httpx.Client")
def test_rerank_with_remote_sorts_descending(mock_client_cls) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"index": 1, "document": "doc1", "score": 0.9},
            {"index": 2, "document": "doc2", "score": 0.5},
        ]
    }
    mock_resp.raise_for_status.return_value = None
    mock_client = MagicMock()
    mock_client.__enter__.return_value.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    docs = [
        Document(page_content="doc0"),
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="docignored"),
    ]
    result = rerank_with_remote(docs, "q", top_k=2, base_url="https://x.ngrok.app")
    assert [d.page_content for d in result] == ["doc1", "doc2"]


def test_rerank_with_remote_empty_input_returns_empty() -> None:
    assert rerank_with_remote([], "q", top_k=3, base_url="https://x.ngrok.app") == []
