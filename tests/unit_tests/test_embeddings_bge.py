from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.utils.embeddings import (
    BGE3Embeddings,
    BGE3SparseEmbeddings,
    get_embedding_model,
    get_sparse_embedding,
)


@pytest.fixture
def fake_bge3_model() -> MagicMock:
    m = MagicMock()
    m.encode.return_value = {
        "dense_vecs": [[1.0, 2.0, 3.0]],
        "lexical_weights": [{"42": 0.5, "7": 0.25}],
    }
    return m


@patch("agent.utils.embeddings._get_bge3_model")
def test_bge_dense_embed_query(mock_get_model, fake_bge3_model) -> None:
    mock_get_model.return_value = fake_bge3_model
    emb = BGE3Embeddings(fake_bge3_model)
    vec = emb.embed_query("hello")
    assert vec == [1.0, 2.0, 3.0]
    fake_bge3_model.encode.assert_called_once()


@patch("agent.utils.embeddings._get_bge3_model")
def test_bge_dense_embed_documents(mock_get_model, fake_bge3_model) -> None:
    fake_bge3_model.encode.return_value = {"dense_vecs": [[1.0], [2.0]]}
    mock_get_model.return_value = fake_bge3_model
    emb = BGE3Embeddings(fake_bge3_model)
    vecs = emb.embed_documents(["a", "b"])
    assert vecs == [[1.0], [2.0]]


@patch("agent.utils.embeddings._get_bge3_sparse_model")
def test_bge_sparse_embed_query(mock_get_sparse_model, fake_bge3_model) -> None:
    fake_bge3_model.encode.return_value = {"lexical_weights": [{"1": 0.5}]}
    mock_get_sparse_model.return_value = fake_bge3_model
    sparse = BGE3SparseEmbeddings(fake_bge3_model)
    vec = sparse.embed_query("hello")
    assert vec.indices == [1]
    assert vec.values == [0.5]


@patch("agent.utils.embeddings._get_bge3_model")
def test_get_embedding_model_returns_bge_provider(mock_get_model, fake_bge3_model) -> None:
    mock_get_model.return_value = fake_bge3_model
    emb = get_embedding_model.__wrapped__ if hasattr(get_embedding_model, "__wrapped__") else get_embedding_model
    cfg = MagicMock(embedding_provider="bge", embedding_model="BAAI/bge-m3")
    res = get_embedding_model(cfg)
    assert isinstance(res, BGE3Embeddings)


def test_get_embedding_model_unknown_provider_raises() -> None:
    cfg = MagicMock(embedding_provider="unknown_provider")
    with pytest.raises(KeyError, match="No suitable embedding Model configured"):
        get_embedding_model(cfg)


def test_get_sparse_embedding_returns_sparse_wrapper() -> None:
    with patch("agent.utils.embeddings._get_bge3_sparse_model") as m:
        m.return_value = MagicMock()
        cfg = MagicMock(sparse_model="BAAI/bge-m3", embedding_model="BAAI/bge-m3")
        res = get_sparse_embedding(cfg)
        assert isinstance(res, BGE3SparseEmbeddings)
