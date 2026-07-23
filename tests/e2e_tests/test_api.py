"""API tests."""

from __future__ import annotations

from http import HTTPStatus

import pytest

pytestmark = pytest.mark.integration


def test_read_root(client) -> None:
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert "Welcome to the RAG Backend" in response.text


def test_docs_endpoint(client) -> None:
    response = client.get("/docs")
    assert response.status_code == HTTPStatus.OK


def test_collection_routes_removed(client) -> None:
    response = client.post("/collection/create/x", params={"embeddings_size": 1536})
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_embeddings_documents_route_removed(client) -> None:
    response = client.post("/embeddings/documents", params={"collection_name": "x"})
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_delete_route_removed(client) -> None:
    response = client.delete("/embeddings/delete/x", params={"collection_name": "x"})
    assert response.status_code == HTTPStatus.NOT_FOUND
