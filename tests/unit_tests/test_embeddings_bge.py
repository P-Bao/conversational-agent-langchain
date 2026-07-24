from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.utils.embeddings import (
    BGEM3RemoteEmbeddings,
    get_embedding_model,
)


def _fake_embed_response() -> dict:
    return {"dense_vecs": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}


def test_remote_embed_query() -> None:
    with patch("agent.utils.embeddings.httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _fake_embed_response()
        mock_resp.raise_for_status.return_value = None
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        emb = BGEM3RemoteEmbeddings("https://example.ngrok-free.app")
        vec = emb.embed_query("hello")
        assert vec == [1.0, 2.0, 3.0]


def test_remote_embed_documents_empty() -> None:
    emb = BGEM3RemoteEmbeddings("https://example.ngrok-free.app")
    assert emb.embed_documents([]) == []


def test_remote_embed_documents() -> None:
    with patch("agent.utils.embeddings.httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _fake_embed_response()
        mock_resp.raise_for_status.return_value = None
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        emb = BGEM3RemoteEmbeddings("https://example.ngrok-free.app")
        vecs = emb.embed_documents(["a", "b"])
        assert vecs == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_get_embedding_model_remote() -> None:
    cfg = MagicMock(embedding_provider="remote", embedding_base_url="https://example.ngrok-free.app")
    res = get_embedding_model(cfg)
    assert isinstance(res, BGEM3RemoteEmbeddings)


def test_get_embedding_model_missing_base_url_raises() -> None:
    cfg = MagicMock(embedding_provider="remote", embedding_base_url="")
    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL is required"):
        get_embedding_model(cfg)
