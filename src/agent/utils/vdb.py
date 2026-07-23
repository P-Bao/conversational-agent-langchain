"""Vector Database client only.

Collection management is delegated to an external system. This module only
exposes the sync and async Qdrant clients used by retrieval.
"""

import warnings

from qdrant_client import AsyncQdrantClient, QdrantClient

from agent.utils.config import Config, config


def _build_sync(cfg: Config) -> QdrantClient:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Api key is used with an insecure connection",
        )
        return QdrantClient(
            location=cfg.qdrant_url,
            port=cfg.qdrant_port,
            api_key=cfg.qdrant_api_key,
            prefer_grpc=cfg.qdrant_prefer_grpc,
        )


def _build_async(cfg: Config) -> AsyncQdrantClient:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Api key is used with an insecure connection",
        )
        return AsyncQdrantClient(
            location=cfg.qdrant_url,
            port=cfg.qdrant_port,
            api_key=cfg.qdrant_api_key,
            prefer_grpc=cfg.qdrant_prefer_grpc,
        )


qdrant_client: QdrantClient = _build_sync(config)
async_qdrant_client: AsyncQdrantClient = _build_async(config)


def load_vec_db_conn() -> QdrantClient:
    """Return the module-level synchronous QdrantClient singleton."""
    return qdrant_client


def get_async_qdrant_client() -> AsyncQdrantClient:
    """Return the module-level asynchronous QdrantClient singleton."""
    return async_qdrant_client
