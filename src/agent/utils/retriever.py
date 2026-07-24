"""Retriever utils with cached embeddings and vector stores (dense-only remote)."""

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from agent.utils.config import Config, config
from agent.utils.embeddings import get_embedding_model
from agent.utils.vdb import qdrant_client

_embeddings_cache: dict[tuple[str, str], Embeddings] = {}
_vector_store_cache: dict[str, QdrantVectorStore] = {}


def _get_cached_embedding(cfg: Config) -> Embeddings:
    """Get or create cached dense embeddings (remote BGE-m3)."""
    key = (cfg.embedding_provider, cfg.embedding_base_url)
    if key not in _embeddings_cache:
        _embeddings_cache[key] = get_embedding_model(cfg)
    return _embeddings_cache[key]


def _get_cached_vector_store(cfg: Config) -> QdrantVectorStore:
    """Get or create cached QdrantVectorStore for the configured collection.

    Dense-only retrieval (remote BGE-m3 không có sparse vector).
    """
    collection_name = cfg.qdrant_collection_name
    if collection_name not in _vector_store_cache:
        _vector_store_cache[collection_name] = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=_get_cached_embedding(cfg),
            retrieval_mode=RetrievalMode.DENSE,
        )
    return _vector_store_cache[collection_name]


def get_retriever(k: int = 4, *, cfg: Config | None = None) -> BaseRetriever:
    """Create a Vector Database retriever (dense-only, remote BGE-m3)."""
    active_cfg = cfg or config
    vector_db = _get_cached_vector_store(active_cfg)
    return vector_db.as_retriever(search_kwargs={"k": k})
