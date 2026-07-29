from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agent.utils.reranker import get_reranker


def _cfg(provider: str = "none", top_k: int = 3) -> MagicMock:
    return MagicMock(
        rerank_provider=provider,
        rerank_top_k=top_k,
        rerank_model="BAAI/bge-reranker-v2-m3",
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
