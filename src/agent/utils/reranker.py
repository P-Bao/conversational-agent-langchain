"""Optional reranker utilities.

Default provider is ``none`` (passthrough / no rerank). Set ``RERANK_PROVIDER=bge``
in the environment to enable the local BGE reranker v2-m3 model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from langchain_core.documents import Document
from loguru import logger

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker

RerankerProvider = Literal["bge", "none"]

RerankerFn = Callable[[list[Document], str], list[Document]]

def _prepare_for_model_fallback(model_input, options=None):
    """Fallback for prepare_for_model method missing in newer transformers."""
    if isinstance(model_input, dict) and 'input_ids' in model_input:
        return model_input
    elif isinstance(model_input, (list, tuple)):
        return {'input_ids': list(model_input)}
    elif hasattr(model_input, 'input_ids'):
        return {'input_ids': model_input.input_ids}
    return model_input

# Cache for BGE reranker (expensive to load)
_bge_reranker: "FlagReranker | None" = None
_bge_reranker_model: str | None = None

def _get_bge_reranker(model_name: str = "BAAI/bge-reranker-v2-m3") -> "FlagReranker":
    """Get or create cached BGE reranker with compatibility fix."""
    global _bge_reranker, _bge_reranker_model
    if _bge_reranker is None or _bge_reranker_model != model_name:
        use_fp16 = torch.cuda.is_available()
        
        # Try multiple import paths for FlagReranker
        try:
            # First try: New format (FlagEmbedding.inference.reranker...)
            from FlagEmbedding.inference.reranker.encoder_only.base import FlagReranker as BaseFlagReranker
            _bge_reranker = BaseFlagReranker(model_name, use_fp16=use_fp16)
        except ImportError:
            try:
                # Second try: Old format (direct import)
                from FlagEmbedding import FlagReranker as BaseFlagReranker
                _bge_reranker = BaseFlagReranker(model_name, use_fp16=use_fp16)
            except ImportError:
                msg = "FlagEmbedding package not properly installed. Try: uv pip install --upgrade FlagEmbedding"
                logger.error(msg)
                raise ImportError(msg)
        
        # Add prepare_for_model if missing (for compatibility with newer transformers)
        tokenizer = _bge_reranker.tokenizer
        if not hasattr(tokenizer, 'prepare_for_model'):
            logger.info("Adding prepare_for_model fallback (newer transformers compatibility)")
            tokenizer.prepare_for_model = lambda model_input, options=None: _prepare_for_model_fallback(model_input, options)
        
        _bge_reranker_model = model_name
    return _bge_reranker

def rerank_with_bge(
    documents: list[Document],
    query: str,
    top_k: int,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> list[Document]:
    """Rerank documents using BGE Reranker v2-m3 (local model)."""
    if not documents:
        return documents

    try:
        reranker = _get_bge_reranker(model_name)
        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker.compute_score(pairs, normalize=True)

    except Exception as e:
        logger.warning(f"BGE reranking failed (prepare_for_model compatibility): {type(e).__name__}: {e}")
        logger.warning("Falling back to manual similarity calculation")

        scores = []
        for q, doc in pairs:
            query_words = set(q.lower().split())
            doc_words = set(doc.lower().split())
            intersection = query_words & doc_words
            union = query_words | doc_words
            score = len(intersection) / len(union) if union else 0.0
            scores.append(score)

    if isinstance(scores, (float, int)):
        scores = [float(scores)]

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
    logger.info(f"BGE reranked {len(documents)} documents to top {len(ranked)}")
    return [doc for doc, _ in ranked]


def get_reranker(
    provider: str,
    top_k: int,
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> RerankerFn:
    """Return a reranking callable for the given provider.

    Supported providers:
    - ``"none"``: passthrough (truncate to ``top_k`` only).
    - ``"bge"``: local BGE reranker v2-m3 via ``FlagEmbedding``.

    The Cohere and FlashRank providers have been removed from this branch to
    avoid extra dependencies. Set ``RERANK_PROVIDER=none`` (default) to skip
    reranking entirely.
    """
    normalized = (provider or "none").strip().lower()

    if normalized == "none":
        def passthrough(docs: list[Document], _query: str) -> list[Document]:
            return docs[:top_k]
        return passthrough

    if normalized == "bge":
        def bge_rerank(docs: list[Document], query: str) -> list[Document]:
            return rerank_with_bge(docs, query, top_k=top_k, model_name=model_name)
        return bge_rerank

    msg = (
        f"Unknown reranker provider: {provider!r}. "
        "Supported providers in this branch: 'bge', 'none'."
    )
    raise ValueError(msg)