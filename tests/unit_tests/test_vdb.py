from __future__ import annotations

from agent.utils.vdb import (
    async_qdrant_client,
    get_async_qdrant_client,
    load_vec_db_conn,
    qdrant_client,
)


def test_sync_client_singleton() -> None:
    assert load_vec_db_conn() is qdrant_client


def test_async_client_singleton() -> None:
    assert get_async_qdrant_client() is async_qdrant_client


def test_qdrant_client_has_collection_exists_method() -> None:
    assert hasattr(qdrant_client, "collection_exists")


def test_async_client_has_collection_exists_method() -> None:
    assert hasattr(async_qdrant_client, "collection_exists")
