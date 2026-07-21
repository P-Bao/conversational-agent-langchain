"""Retriever utils with cached embeddings and vector stores."""

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import models

from agent.utils.config import Config, config
from agent.utils.embeddings import get_embedding_model
from agent.utils.vdb import qdrant_client, sparse_embeddings

# Cache embeddings - created once per model name
_embeddings_cache: dict[tuple[str, str], Embeddings] = {}


def _get_cached_embedding() -> Embeddings:
    """Get or create cached embeddings for the configured provider."""
    key = (config.embedding_provider, config.embedding_model_name)
    if key not in _embeddings_cache:
        _embeddings_cache[key] = get_embedding_model(config)
    return _embeddings_cache[key]


def _build_fusion_query(cfg: Config) -> models.FusionQuery:
    """Build a Qdrant FusionQuery from config.

    Args:
        cfg: Config instance carrying ``fusion_algorithm`` (``"rrf"`` or ``"dbsf"``).

    Returns:
        models.FusionQuery describing RRF or DBSF. The qdrant-client SDK only
        exposes the algorithm choice; smoothing constants (rrf_k, dbsf_window)
        use Qdrant server defaults (60 and 1000 respectively).

    Raises:
        ValueError: if fusion_algorithm is not "rrf" or "dbsf".

    """
    algo = (cfg.fusion_algorithm or "rrf").lower().strip()
    match algo:
        case "rrf":
            return models.FusionQuery(fusion=models.Fusion.RRF)
        case "dbsf":
            return models.FusionQuery(fusion=models.Fusion.DBSF)
        case _:
            msg = f"Unknown fusion_algorithm: {cfg.fusion_algorithm!r}. Must be 'rrf' or 'dbsf'."
            raise ValueError(msg)


# Cache vector stores per collection
_vector_store_cache: dict[str, QdrantVectorStore] = {}


def _get_cached_vector_store(collection_name: str) -> QdrantVectorStore:
    """Get or create cached QdrantVectorStore for a collection."""
    if collection_name not in _vector_store_cache:
        _vector_store_cache[collection_name] = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=_get_cached_embedding(),
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="fast-sparse-bm25",
        )
    return _vector_store_cache[collection_name]


def get_retriever(k: int = 4, collection_name: str = "default", *, cfg: Config | None = None) -> BaseRetriever:
    """Create a Vector Database retriever with hybrid search.

    Uses cached embeddings and vector stores for performance.

    Args:
        k: Number of documents to retrieve.
        collection_name: Name of the collection to search.
        cfg: Optional Config override; defaults to the module-level ``config``.

    Returns:
        BaseRetriever: Qdrant + Cohere Embeddings Retriever with hybrid search
        (RRF by default, configurable via ``cfg.fusion_algorithm``).

    """
    active_cfg = cfg or config
    vector_db = _get_cached_vector_store(collection_name)
    fusion_query = _build_fusion_query(active_cfg)
    return vector_db.as_retriever(
        search_kwargs={"k": k, "hybrid_fusion": fusion_query},
    )
