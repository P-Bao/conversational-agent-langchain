"""Retriever utils with cached embeddings and vector stores."""

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import models

from agent.utils.config import Config, config
from agent.utils.embeddings import get_embedding_model, get_sparse_embedding
from agent.utils.vdb import qdrant_client

_embeddings_cache: dict[tuple[str, str], Embeddings] = {}
_sparse_embeddings_cache: dict[tuple[str, str], SparseEmbeddings] = {}


def _get_cached_embedding(cfg: Config) -> Embeddings:
    """Get or create cached embeddings for the configured provider."""
    key = (cfg.embedding_provider, cfg.embedding_model)
    if key not in _embeddings_cache:
        _embeddings_cache[key] = get_embedding_model(cfg)
    return _embeddings_cache[key]


def _get_cached_sparse(cfg: Config) -> SparseEmbeddings:
    """Get or create cached sparse embeddings."""
    key = (cfg.embedding_provider, cfg.sparse_model)
    if key not in _sparse_embeddings_cache:
        _sparse_embeddings_cache[key] = get_sparse_embedding(cfg)
    return _sparse_embeddings_cache[key]


def _build_fusion_query(cfg: Config) -> models.FusionQuery:
    """Build a Qdrant FusionQuery from config."""
    algo = (cfg.fusion_algorithm or "rrf").lower().strip()
    match algo:
        case "rrf":
            return models.FusionQuery(fusion=models.Fusion.RRF)
        case "dbsf":
            return models.FusionQuery(fusion=models.Fusion.DBSF)
        case _:
            msg = f"Unknown fusion_algorithm: {cfg.fusion_algorithm!r}. Must be 'rrf' or 'dbsf'."
            raise ValueError(msg)


_vector_store_cache: dict[str, QdrantVectorStore] = {}


def _get_cached_vector_store(collection_name: str, cfg: Config) -> QdrantVectorStore:
    """Get or create cached QdrantVectorStore for a collection."""
    if collection_name not in _vector_store_cache:
        _vector_store_cache[collection_name] = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=_get_cached_embedding(cfg),
            sparse_embedding=_get_cached_sparse(cfg),
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name=cfg.sparse_vector_name,
        )
    return _vector_store_cache[collection_name]


def get_retriever(k: int = 4, collection_name: str = "default", *, cfg: Config | None = None) -> BaseRetriever:
    """Create a Vector Database retriever with hybrid search."""
    active_cfg = cfg or config
    vector_db = _get_cached_vector_store(collection_name, active_cfg)
    fusion_query = _build_fusion_query(active_cfg)
    return vector_db.as_retriever(
        search_kwargs={"k": k, "hybrid_fusion": fusion_query},
    )
