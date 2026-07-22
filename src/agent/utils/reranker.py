"""Reranker utilities for document reranking."""

from typing import TYPE_CHECKING, Literal

from langchain_core.documents import Document
from loguru import logger

if TYPE_CHECKING:
    from flashrank import Ranker
    from FlagEmbedding import FlagReranker

RerankerProvider = Literal["bge", "cohere", "flashrank", "none"]

# Cache for FlashRank model (expensive to load)
_flashrank_ranker: "Ranker | None" = None
_bge_reranker: "FlagReranker | None" = None
_bge_reranker_model: str | None = None


def _get_flashrank_ranker() -> "Ranker":
    """Get or create cached FlashRank ranker."""
    global _flashrank_ranker
    if _flashrank_ranker is None:
        from flashrank import Ranker  # noqa: PLC0415

        _flashrank_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        logger.info("FlashRank model loaded and cached")
    return _flashrank_ranker


def _get_bge_reranker(model_name: str = "BAAI/bge-reranker-v2-m3") -> "FlagReranker":
    """Get or create cached BGE reranker."""
    global _bge_reranker, _bge_reranker_model
    if _bge_reranker is None or _bge_reranker_model != model_name:
        import torch
        from FlagEmbedding import FlagReranker  # noqa: PLC0415

        use_fp16 = torch.cuda.is_available()
        logger.info(f"Loading BGE reranker model: {model_name}")
        _bge_reranker = FlagReranker(model_name, use_fp16=use_fp16)
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

    reranker = _get_bge_reranker(model_name)
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, (float, int)):
        scores = [float(scores)]

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
    logger.info(f"BGE reranked {len(documents)} documents to top {len(ranked)}")
    return [doc for doc, _ in ranked]


def rerank_with_cohere(
    documents: list[Document],
    query: str,
    top_k: int,
    api_key: str,
    model: str = "rerank-v3.5",
) -> list[Document]:
    """Rerank documents using Cohere Rerank API."""
    from langchain_cohere import CohereRerank  # noqa: PLC0415

    if not documents:
        return documents

    reranker = CohereRerank(model=model, cohere_api_key=api_key, top_n=top_k)
    reranked = reranker.compress_documents(documents=documents, query=query)
    logger.info(f"Cohere reranked {len(documents)} documents to top {len(reranked)}")
    return list(reranked)


def rerank_with_flashrank(documents: list[Document], query: str, top_k: int) -> list[Document]:
    """Rerank documents using FlashRank (local model)."""
    from flashrank import RerankRequest  # noqa: PLC0415

    if not documents:
        return documents

    ranker = _get_flashrank_ranker()

    passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(documents)]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    reranked_docs = []
    for result in sorted_results:
        original_idx = result["id"]
        reranked_docs.append(documents[original_idx])

    logger.info(f"FlashRank reranked {len(documents)} documents to top {len(reranked_docs)}")
    return reranked_docs


def get_reranker(
    provider: RerankerProvider = "bge",
    top_k: int = 5,
    cohere_api_key: str | None = None,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> callable:
    """Get a reranker function based on the provider."""
    match provider:
        case "none":
            logger.info("Reranking disabled, using passthrough")
            return lambda docs, _: docs[:top_k] if len(docs) > top_k else docs

        case "bge":
            return lambda docs, query: rerank_with_bge(docs, query, top_k, model_name)

        case "cohere":
            if not cohere_api_key:
                msg = "Cohere API key is required for Cohere reranker"
                raise ValueError(msg)
            return lambda docs, query: rerank_with_cohere(docs, query, top_k, cohere_api_key)

        case "flashrank":
            _get_flashrank_ranker()
            return lambda docs, query: rerank_with_flashrank(docs, query, top_k)

        case _:
            msg = f"Unknown reranker provider: {provider}"
            raise ValueError(msg)
