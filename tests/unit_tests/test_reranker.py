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
        rerank_base_url="http://rerank-server",
        rerank_min_score=0.0,
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


# ---------------------------------------------------------------------------
# Score propagation into Document.metadata
# ---------------------------------------------------------------------------


def test_rerank_with_bge_writes_scores_to_metadata(monkeypatch) -> None:
    """``rerank_with_bge`` must populate ``metadata["score"]`` for each returned doc."""
    from agent.utils import reranker as reranker_module
    from agent.utils.reranker import rerank_with_bge

    fake_reranker = MagicMock()
    # Scores in reranker-returned order (original docs order: a, b, c).
    fake_reranker.compute_score.return_value = [0.1, 0.9, 0.5]
    monkeypatch.setattr(reranker_module, "_get_local_reranker", lambda _m: fake_reranker)

    docs = [
        Document(page_content="a", metadata={"source": "s.pdf"}),
        Document(page_content="b", metadata={"source": "s.pdf"}),
        Document(page_content="c", metadata={"source": "s.pdf"}),
    ]

    result = rerank_with_bge(docs, "query", top_k=2)

    # Sorted desc: b (0.9), c (0.5); truncated to top_k=2.
    assert [d.page_content for d in result] == ["b", "c"]
    assert result[0].metadata["score"] == pytest.approx(0.9)
    assert result[1].metadata["score"] == pytest.approx(0.5)
    # Original metadata preserved alongside the new score.
    assert result[0].metadata["source"] == "s.pdf"


def test_rerank_with_bge_single_score_list_normalization(monkeypatch) -> None:
    """Single-doc path: ``compute_score`` returns a float, we normalize to list."""
    from agent.utils import reranker as reranker_module
    from agent.utils.reranker import rerank_with_bge

    fake_reranker = MagicMock()
    fake_reranker.compute_score.return_value = 0.42  # single float
    monkeypatch.setattr(reranker_module, "_get_local_reranker", lambda _m: fake_reranker)

    docs = [Document(page_content="only")]

    result = rerank_with_bge(docs, "q", top_k=1)

    assert len(result) == 1
    assert result[0].metadata["score"] == pytest.approx(0.42)


def test_rerank_with_remote_writes_scores_to_metadata(monkeypatch) -> None:
    """``rerank_with_remote`` map ranked_indices + scores (contract mới) vào metadata."""
    from agent.utils.reranker import rerank_with_remote

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(
        return_value={
            "scores": [0.31, 0.95],
            "ranked_indices": [1, 0],
        }
    )

    import httpx

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return fake_response

    monkeypatch.setattr(httpx, "Client", FakeClient)

    docs = [
        Document(page_content="a", metadata={"source": "s.pdf"}),
        Document(page_content="b"),
    ]

    result = rerank_with_remote(docs, "q", top_k=2, base_url="http://x", min_score=0.0, timeout=5)

    assert [d.page_content for d in result] == ["b", "a"]
    assert result[0].metadata["score"] == pytest.approx(0.95)
    assert result[1].metadata["score"] == pytest.approx(0.31)
    # document "a" preserved its existing metadata
    assert result[1].metadata["source"] == "s.pdf"


def test_rerank_with_remote_skips_indexes_out_of_range(monkeypatch) -> None:
    """Out-of-range / missing indices in ``ranked_indices`` dropped safely."""
    from agent.utils.reranker import rerank_with_remote

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(
        return_value={
            "scores": [0.99, 0.4],
            "ranked_indices": [5, 1],  # 5 out of range, 1 valid
        }
    )

    import httpx

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return fake_response

    monkeypatch.setattr(httpx, "Client", FakeClient)

    docs = [Document(page_content="a"), Document(page_content="b")]

    result = rerank_with_remote(docs, "q", top_k=2, base_url="http://x", min_score=0.0, timeout=5)

    # Only index 1 valid; out-of-range 5 dropped.
    assert [d.page_content for d in result] == ["b"]
    assert result[0].metadata["score"] == pytest.approx(0.4)


def test_rerank_with_remote_forwards_min_score_and_top_k(monkeypatch) -> None:
    """Payload gửi server chứa ``min_score`` + ``top_k`` theo cfg."""
    from agent.utils.reranker import rerank_with_remote

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(
        return_value={"scores": [0.9, 0.7, 0.5], "ranked_indices": [0, 1, 2]}
    )

    captured: dict = {}

    import httpx

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, _url, json=None, **_kwargs):
            captured["payload"] = json
            return fake_response

    monkeypatch.setattr(httpx, "Client", FakeClient)

    docs = [Document(page_content="a"), Document(page_content="b"), Document(page_content="c")]
    rerank_with_remote(
        docs, "q", top_k=3, base_url="http://x", min_score=0.1, timeout=5
    )

    assert captured["payload"]["min_score"] == 0.1
    assert captured["payload"]["top_k"] == 3
    assert "normalize" not in captured["payload"]
    assert captured["payload"]["query"] == "q"
    assert captured["payload"]["documents"] == ["a", "b", "c"]


def test_rerank_with_remote_backward_compat_results_contract(monkeypatch) -> None:
    """Fallback tương thích contract cũ ``{"results": [{index, score}]}``."""
    from agent.utils.reranker import rerank_with_remote

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(
        return_value={
            "results": [
                {"index": 1, "score": 0.95},
                {"index": 0, "score": 0.31},
            ]
        }
    )

    import httpx

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return fake_response

    monkeypatch.setattr(httpx, "Client", FakeClient)

    docs = [Document(page_content="a"), Document(page_content="b")]

    result = rerank_with_remote(docs, "q", top_k=2, base_url="http://x", min_score=0.0, timeout=5)

    assert [d.page_content for d in result] == ["b", "a"]
    assert result[0].metadata["score"] == pytest.approx(0.95)
    assert result[1].metadata["score"] == pytest.approx(0.31)


def test_get_reranker_remote_requires_base_url() -> None:
    """``provider=remote`` mà thiếu ``rerank_base_url`` phải raise."""
    from agent.utils.reranker import get_reranker

    cfg = MagicMock(
        rerank_provider="remote",
        rerank_top_k=3,
        rerank_model="BAAI/bge-reranker-v2-m3",
        rerank_base_url="",
        rerank_min_score=0.0,
    )
    with pytest.raises(ValueError, match="RERANK_BASE_URL is required"):
        get_reranker(cfg)


def test_get_reranker_default_provider_is_remote(monkeypatch) -> None:
    """Config default ``rerank_provider = 'remote'`` (không còn ``bge``)."""
    from agent.utils.config import Config
    from agent.utils.reranker import rerank_with_remote

    cfg = Config(
        rerank_provider="remote",
        rerank_base_url="http://rerank-server",
        rerank_top_k=2,
    )
    fn = get_reranker(cfg, top_k=2)
    docs = [Document(page_content="a"), Document(page_content="b")]

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(
        return_value={"scores": [0.1, 0.9], "ranked_indices": [1, 0]}
    )

    import httpx

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return fake_response

    monkeypatch.setattr(httpx, "Client", FakeClient)

    direct = rerank_with_remote(docs, "q", top_k=2, base_url="http://rerank-server", min_score=0.0)
    assert [d.page_content for d in direct] == ["b", "a"], f"direct got {[(d.page_content, d.metadata.get('score')) for d in direct]}"

    result = fn(docs, "q")
    assert [d.page_content for d in result] == ["b", "a"], f"closure got {[(d.page_content, d.metadata.get('score')) for d in result]}"
    assert result[0].metadata["score"] == pytest.approx(0.9)
