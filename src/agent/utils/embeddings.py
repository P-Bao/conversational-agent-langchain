"""Embedding model utilities for BGE-m3 (dense + sparse from same model).

Sử dụng FlagEmbedding.BGEM3FlagModel — hỗ trợ native dense+sparse+colbert trong 1 forward pass.
Mặc định dense và sparse dùng cùng model `BAAI/bge-m3`, nhưng env tách (`EMBEDDING_MODEL`
cho dense, `SPARSE_MODEL` cho sparse) để swap linh hoạt. Vì cùng model, một instance
BGEM3FlagModel chia sẻ cho cả 2 wrapper (tránh load model 2 lần tốn RAM/GPU).

Chỉ hỗ trợ provider ``bge`` ở v7 — OpenAI-compatible provider đã được chuyển
sang hệ thống ngoài (ingestion repo).
"""

from __future__ import annotations

import torch
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector
from loguru import logger

from agent.utils.config import Config


class BGE3Embeddings(Embeddings):
    """LangChain `Embeddings` wrapper cho phần dense của BGE-m3."""

    def __init__(self, model: "BGEM3FlagModel") -> None:  # type: ignore[name-defined]
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense = result["dense_vecs"]
        return [list(map(float, vec)) for vec in dense]

    def embed_query(self, text: str) -> list[float]:
        result = self._model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return list(map(float, result["dense_vecs"][0]))


class BGE3SparseEmbeddings(SparseEmbeddings):
    """Sparse wrapper tương thích `langchain_qdrant.SparseEmbeddings` interface.

    Trả về SparseVector(indices, values) cho mỗi input text cho Qdrant named vector.
    """

    def __init__(self, model: "BGEM3FlagModel") -> None:  # type: ignore[name-defined]
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        result = self._model.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        sparse = result["lexical_weights"]
        out = []
        for doc in sparse:
            indices = [int(k) for k in doc.keys()]
            values = [float(v) for v in doc.values()]
            out.append(SparseVector(indices=indices, values=values))
        return out

    def embed_query(self, text: str) -> SparseVector:
        result = self._model.encode([text], return_dense=False, return_sparse=True, return_colbert_vecs=False)
        sparse = result["lexical_weights"][0]
        indices = [int(k) for k in sparse.keys()]
        values = [float(v) for v in sparse.values()]
        return SparseVector(indices=indices, values=values)


# Cache singleton: chia sẻ 1 instance BGEM3FlagModel giữa dense + sparse wrapper
_bge3_model: "BGEM3FlagModel | None" = None  # type: ignore[name-defined]
_bge3_model_name: str | None = None


def _get_bge3_model(cfg: Config):
    """Load (cache) BGEM3FlagModel. Nếu dense và sparse cùng model name → dùng chung 1 instance."""
    global _bge3_model, _bge3_model_name

    model_name = cfg.embedding_model
    if _bge3_model is None or _bge3_model_name != model_name:
        from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415

        logger.info(f"Loading BGE-m3 model: {model_name}")
        use_fp16 = torch.cuda.is_available()
        _bge3_model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        _bge3_model_name = model_name
    return _bge3_model


def _get_bge3_sparse_model(cfg: Config):
    """Sparse model instance. Nếu sparse_model == dense_model → tái dùng dense singleton."""
    if cfg.sparse_model == cfg.embedding_model:
        return _get_bge3_model(cfg)

    global _bge3_model, _bge3_model_name  # noqa: PLW0603
    from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415

    logger.info(f"Loading BGE-m3 sparse model (separate): {cfg.sparse_model}")
    use_fp16 = torch.cuda.is_available()
    return BGEM3FlagModel(cfg.sparse_model, use_fp16=use_fp16)


def get_embedding_model(cfg: Config) -> Embeddings:
    """Return dense embeddings client (chỉ hỗ trợ provider BGE ở v7)."""
    provider = cfg.embedding_provider
    if provider != "bge":
        msg = f"No suitable embedding Model configured for provider: {provider!r}. Only 'bge' supported in v7."
        raise KeyError(msg)
    return BGE3Embeddings(_get_bge3_model(cfg))


def get_sparse_embedding(cfg: Config) -> BGE3SparseEmbeddings:
    """Return sparse embeddings client (BGE-m3 lexical weights).

    Named vector `sparse_vector_name` trong Config (`bge-m3-sparse`) dùng làm
    named vector trong Qdrant.
    """
    return BGE3SparseEmbeddings(_get_bge3_sparse_model(cfg))
