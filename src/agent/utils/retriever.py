"""Retriever utils with cached embeddings and vector stores (hybrid dense+sparse remote)."""

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from agent.utils.config import Config, config
from agent.utils.embeddings import BGEM3RemoteEmbeddings, get_embedding_model
from agent.utils.vdb import qdrant_client

_embeddings_cache: dict[tuple[str, str], BGEM3RemoteEmbeddings] = {}
_vector_store_cache: dict[str, QdrantVectorStore] = {}


class BGE_M3SparseEmbeddings(Embeddings):
    """Sparse embeddings wrapper using BGE-M3 remote endpoint via cached dense embeddings."""

    def __init__(self, dense_embeddings: BGEM3RemoteEmbeddings) -> None:
        self._dense_embeddings = dense_embeddings

    def embed_documents(self, texts: list[str]) -> list[dict]:
        # Trigger dense embedding to populate sparse cache
        _ = self._dense_embeddings.embed_documents(texts)
        sparse_vecs = self._dense_embeddings.get_last_sparse_vecs()
        if sparse_vecs is None:
            return [{"indices": [], "values": []} for _ in texts]
        return sparse_vecs

    def embed_query(self, text: str) -> dict:
        return self.embed_documents([text])[0]


def _get_cached_embedding(cfg: Config) -> BGEM3RemoteEmbeddings:
    """Get or create cached dense embeddings (remote BGE-m3)."""
    key = (cfg.embedding_provider, cfg.embedding_base_url)
    if key not in _embeddings_cache:
        _embeddings_cache[key] = get_embedding_model(cfg)
    return _embeddings_cache[key]


def _get_cached_sparse_embedding(cfg: Config) -> BGE_M3SparseEmbeddings:
    """Get or create cached sparse embeddings wrapper."""
    dense = _get_cached_embedding(cfg)
    return BGE_M3SparseEmbeddings(dense)


def _get_cached_vector_store(cfg: Config) -> QdrantVectorStore:
    """Get or create cached QdrantVectorStore for the configured collection.

    Hybrid retrieval using BGE-M3 dense + sparse vectors.
    """
    collection_name = cfg.qdrant_collection_name
    if collection_name not in _vector_store_cache:
        dense_embedding = _get_cached_embedding(cfg)
        sparse_embedding = _get_cached_sparse_embedding(cfg)
        _vector_store_cache[collection_name] = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
    return _vector_store_cache[collection_name]


def get_retriever(k: int = 4, *, cfg: Config | None = None) -> BaseRetriever:
    """Create a Vector Database retriever (hybrid, remote BGE-m3 dense+sparse)."""
    active_cfg = cfg or config
    vector_db = _get_cached_vector_store(active_cfg)
    return vector_db.as_retriever(search_kwargs={"k": k})
