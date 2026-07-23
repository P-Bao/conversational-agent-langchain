from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models

from agent.utils import retriever as retriever_module
from agent.utils.retriever import get_retriever
from tests.fakes.rag import FakeAsyncRetriever, FakeDoc


@patch("agent.routes.search.get_retriever")
def test_search_returns_documents(mock_get_retriever, client) -> None:
    mock_get_retriever.return_value = FakeAsyncRetriever(
        [FakeDoc(page_content="content", metadata={"page": 1, "source": "test.pdf"})],
    )

    response = client.post(
        "/semantic/search",
        json={"query": "test", "collection_name": "test_coll", "k": 2},
    )

    assert response.status_code == 200
    assert response.json() == [{"text": "content", "page": 1, "source": "test.pdf"}]


@patch("agent.routes.search.get_retriever")
def test_search_no_documents_returns_message(mock_get_retriever, client) -> None:
    mock_get_retriever.return_value = FakeAsyncRetriever([])

    response = client.post(
        "/semantic/search",
        json={"query": "test", "collection_name": "test_coll"},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "No documents found."}


@patch("agent.routes.search.get_retriever")
def test_search_passes_collection_and_k(mock_get_retriever, client) -> None:
    mock_get_retriever.return_value = FakeAsyncRetriever([])

    client.post(
        "/semantic/search",
        json={"query": "q", "collection_name": "my_coll", "k": 7},
    )

    mock_get_retriever.assert_called_once_with(collection_name="my_coll", k=7)


@patch("agent.utils.retriever.qdrant_client")
@patch("agent.utils.retriever.get_sparse_embedding")
@patch("agent.utils.retriever.QdrantVectorStore")
@patch("agent.utils.retriever.get_embedding_model")
def test_get_retriever_uses_rrf(
    mock_get_embedding_model,
    mock_vector_store,
    _mock_sparse,
    _mock_client,
) -> None:
    mock_vstore_instance = MagicMock()
    mock_vector_store.return_value = mock_vstore_instance
    retriever_module._embeddings_cache.clear()
    retriever_module._sparse_embeddings_cache.clear()
    retriever_module._vector_store_cache.clear()
    mock_get_embedding_model.return_value = MagicMock()

    get_retriever(k=5, collection_name="my_coll")

    mock_vector_store.assert_called_once()
    _, kwargs = mock_vstore_instance.as_retriever.call_args
    search_kwargs = kwargs["search_kwargs"]
    assert search_kwargs["k"] == 5
    assert isinstance(search_kwargs["hybrid_fusion"], models.FusionQuery)
    assert search_kwargs["hybrid_fusion"].fusion == models.Fusion.RRF


@patch("agent.utils.retriever.qdrant_client")
@patch("agent.utils.retriever.get_sparse_embedding")
@patch("agent.utils.retriever.QdrantVectorStore")
@patch("agent.utils.retriever.get_embedding_model")
def test_get_retriever_uses_dbsf(
    mock_get_embedding_model,
    mock_vector_store,
    _mock_sparse,
    _mock_client,
) -> None:
    mock_vstore_instance = MagicMock()
    mock_vector_store.return_value = mock_vstore_instance
    retriever_module._embeddings_cache.clear()
    retriever_module._sparse_embeddings_cache.clear()
    retriever_module._vector_store_cache.clear()
    mock_get_embedding_model.return_value = MagicMock()

    dbsf_cfg = retriever_module.Config(fusion_algorithm="dbsf")
    get_retriever(k=3, collection_name="my_coll", cfg=dbsf_cfg)

    _, kwargs = mock_vstore_instance.as_retriever.call_args
    search_kwargs = kwargs["search_kwargs"]
    assert search_kwargs["k"] == 3
    assert isinstance(search_kwargs["hybrid_fusion"], models.FusionQuery)
    assert search_kwargs["hybrid_fusion"].fusion == models.Fusion.DBSF


@patch("agent.utils.retriever.qdrant_client")
@patch("agent.utils.retriever.get_sparse_embedding")
@patch("agent.utils.retriever.QdrantVectorStore")
@patch("agent.utils.retriever.get_embedding_model")
def test_get_retriever_invalid_fusion_raises(
    mock_get_embedding_model,
    mock_vector_store,
    _mock_sparse,
    _mock_client,
) -> None:
    retriever_module._embeddings_cache.clear()
    retriever_module._sparse_embeddings_cache.clear()
    retriever_module._vector_store_cache.clear()
    mock_get_embedding_model.return_value = MagicMock()
    mock_vector_store.return_value = MagicMock()

    bad_cfg = retriever_module.Config(fusion_algorithm="banana")
    with pytest.raises(ValueError, match="Unknown fusion_algorithm"):
        get_retriever(k=3, collection_name="my_coll", cfg=bad_cfg)


@patch("agent.utils.retriever.qdrant_client")
@patch("agent.utils.retriever.get_sparse_embedding")
@patch("agent.utils.retriever.QdrantVectorStore")
@patch("agent.utils.retriever.get_embedding_model")
def test_get_retriever_caches_vector_store(
    mock_get_embedding_model,
    mock_vector_store,
    _mock_sparse,
    _mock_client,
) -> None:
    mock_vstore_instance = MagicMock()
    mock_vstore_instance.as_retriever.return_value = "fake"
    mock_vector_store.return_value = mock_vstore_instance
    retriever_module._embeddings_cache.clear()
    retriever_module._sparse_embeddings_cache.clear()
    retriever_module._vector_store_cache.clear()
    mock_get_embedding_model.return_value = MagicMock()

    get_retriever(collection_name="cached_coll")
    get_retriever(collection_name="cached_coll")
    get_retriever(collection_name="cached_coll")

    assert mock_vector_store.call_count == 1
