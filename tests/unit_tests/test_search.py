from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        json={"query": "test", "k": 2},
    )

    assert response.status_code == 200
    assert response.json() == [{"text": "content"}]


@patch("agent.routes.search.get_retriever")
def test_search_no_documents_returns_message(mock_get_retriever, client) -> None:
    mock_get_retriever.return_value = FakeAsyncRetriever([])

    response = client.post(
        "/semantic/search",
        json={"query": "test"},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "No documents found."}


@patch("agent.routes.search.get_retriever")
def test_search_passes_k(mock_get_retriever, client) -> None:
    mock_get_retriever.return_value = FakeAsyncRetriever([])

    client.post(
        "/semantic/search",
        json={"query": "q", "k": 7},
    )

    mock_get_retriever.assert_called_once_with(k=7)


@patch("agent.utils.retriever.qdrant_client")
@patch("agent.utils.retriever.QdrantVectorStore")
@patch("agent.utils.retriever.get_embedding_model")
def test_get_retriever_dense_only(
    mock_get_embedding_model,
    mock_vector_store,
    _mock_client,
) -> None:
    from agent.utils.config import Config

    mock_vstore_instance = MagicMock()
    mock_vector_store.return_value = mock_vstore_instance
    retriever_module._embeddings_cache.clear()
    retriever_module._vector_store_cache.clear()
    mock_get_embedding_model.return_value = MagicMock()

    cfg = Config(embedding_provider="remote", embedding_base_url="https://x.ngrok.app")
    get_retriever(k=5, cfg=cfg)

    mock_vector_store.assert_called_once()
    _, kwargs = mock_vector_store.call_args
    # Current code uses HYBRID mode (dense + sparse BGE-M3).
    assert kwargs["retrieval_mode"].value == "hybrid"

    _, rk_kwargs = mock_vstore_instance.as_retriever.call_args
    assert rk_kwargs["search_kwargs"]["k"] == 5


@patch("agent.utils.retriever.qdrant_client")
@patch("agent.utils.retriever.QdrantVectorStore")
@patch("agent.utils.retriever.get_embedding_model")
def test_get_retriever_caches_vector_store(
    mock_get_embedding_model,
    mock_vector_store,
    _mock_client,
) -> None:
    mock_vstore_instance = MagicMock()
    mock_vstore_instance.as_retriever.return_value = "fake"
    mock_vector_store.return_value = mock_vstore_instance
    retriever_module._embeddings_cache.clear()
    retriever_module._vector_store_cache.clear()
    mock_get_embedding_model.return_value = MagicMock()

    get_retriever()
    get_retriever()
    get_retriever()

    assert mock_vector_store.call_count == 1
