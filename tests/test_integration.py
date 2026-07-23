from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inline_snapshot import snapshot

pytestmark = pytest.mark.integration


def test_read_root(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the RAG Backend" in response.text


def test_docs_endpoint(client) -> None:
    response = client.get("/docs")
    assert response.status_code == 200


def test_collection_routes_removed(client) -> None:
    assert client.post("/collection/create/x", params={"embeddings_size": 1536}).status_code == 404


def test_embeddings_routes_removed(client) -> None:
    assert client.post("/embeddings/documents", params={"collection_name": "x"}).status_code == 404


def test_delete_route_removed(client) -> None:
    assert client.delete("/embeddings/delete/x", params={"collection_name": "x"}).status_code == 404


@patch("agent.routes.search.get_retriever")
def test_search_endpoint_execution(mock_get_retriever, client) -> None:
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.ainvoke = AsyncMock(
        return_value=[
            MagicMock(
                page_content="Test document content",
                metadata={"source": "test.pdf", "page": 1},
            )
        ]
    )
    mock_get_retriever.return_value = mock_retriever_instance

    response = client.post(
        "/semantic/search",
        json={"query": "test", "collection_name": "default", "k": 4},
    )
    assert response.status_code == 200
    assert response.json() == snapshot(
        [{"text": "Test document content", "page": 1, "source": "test.pdf"}]
    )
