from __future__ import annotations

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from unittest.mock import AsyncMock, patch


def test_healthz_returns_ok(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_does_not_call_qdrant(client) -> None:
    with patch("agent.routes.health.get_async_qdrant_client") as mocked:
        response = client.get("/healthz")
        assert response.status_code == 200
        mocked.assert_not_called()


@pytest.mark.anyio
async def test_readyz_returns_ready_when_collection_exists(client) -> None:
    from agent import routes as routes_module

    with patch.object(routes_module.health, "get_async_qdrant_client") as mocked_factory:
        client_obj = AsyncMock()
        client_obj.collection_exists = AsyncMock(return_value=True)
        mocked_factory.return_value = client_obj

        response = client.get("/readyz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["collection"] == "documents"


def test_readyz_returns_503_when_collection_missing(client) -> None:
    from agent import routes as routes_module

    with patch.object(routes_module.health, "get_async_qdrant_client") as mocked_factory:
        client_obj = AsyncMock()
        client_obj.collection_exists = AsyncMock(return_value=False)
        mocked_factory.return_value = client_obj

        response = client.get("/readyz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "fail"
    assert payload["reason"] == "collection_missing"


def test_readyz_returns_503_on_unexpected_response(client) -> None:
    from agent import routes as routes_module

    with patch.object(routes_module.health, "get_async_qdrant_client") as mocked_factory:
        client_obj = AsyncMock()
        client_obj.collection_exists = AsyncMock(
            side_effect=UnexpectedResponse(
                status_code=500,
                reason_phrase="Internal Server Error",
                content=b"boom",
                headers={},
            )
        )
        mocked_factory.return_value = client_obj

        response = client.get("/readyz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "fail"
    assert payload["reason"] == "qdrant_error"


def test_readyz_returns_503_on_transport_error(client) -> None:
    from agent import routes as routes_module

    with patch.object(routes_module.health, "get_async_qdrant_client") as mocked_factory:
        client_obj = AsyncMock()
        client_obj.collection_exists = AsyncMock(side_effect=ConnectionError("refused"))
        mocked_factory.return_value = client_obj

        response = client.get("/readyz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "fail"
    assert payload["reason"] == "qdrant_unreachable"
