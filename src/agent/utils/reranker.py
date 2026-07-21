"""Reranker utilities for document reranking."""

from typing import TYPE_CHECKING, Literal

from langchain_core.documents import Document
from loguru import logger

if TYPE_CHECKING:
    from flashrank import Ranker

RerankerProvider = Literal["cohere", "flashrank", "none"]

# Cache for FlashRank model (expensive to load)
_flashrank_ranker: "Ranker | None" = None


def _get_flashrank_ranker() -> "Ranker":
    """Get or create cached FlashRank ranker."""
    global _flashrank_ranker
    if _flashrank_ranker is None:
        from flashrank import Ranker  # noqa: PLC0415

        _flashrank_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        logger.info("FlashRank model loaded and cached")
    return _flashrank_ranker


def rerank_with_cohere(
    documents: list[Document],
    query: str,
    top_k: int,
    api_key: str,
    model: str = "rerank-v3.5",
) -> list[Document]:
    """Rerank documents using Cohere Rerank API.

    Args:
        documents: List of documents to rerank.
        query: The query to rerank against.
        top_k: Number of top documents to return.
        api_key: Cohere API key.
        model: Cohere rerank model name.

    Returns:
        Reranked list of documents.

    """
    from langchain_cohere import CohereRerank  # noqa: PLC0415

    if not documents:
        return documents

    reranker = CohereRerank(model=model, cohere_api_key=api_key, top_n=top_k)
    reranked = reranker.compress_documents(documents=documents, query=query)
    logger.info(f"Cohere reranked {len(documents)} documents to top {len(reranked)}")
    return list(reranked)


def rerank_with_flashrank(documents: list[Document], query: str, top_k: int) -> list[Document]:
    """Rerank documents using FlashRank (local model).

    Args:
        documents: List of documents to rerank.
        query: The query to rerank against.
        top_k: Number of top documents to return.

    Returns:
        Reranked list of documents.

    """
    from flashrank import RerankRequest  # noqa: PLC0415

    if not documents:
        return documents

    ranker = _get_flashrank_ranker()

    # Convert documents to flashrank format
    passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(documents)]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    # Sort by score and take top_k
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    # Reconstruct documents preserving original metadata
    reranked_docs = []
    for result in sorted_results:
        original_idx = result["id"]
        reranked_docs.append(documents[original_idx])

    logger.info(f"FlashRank reranked {len(documents)} documents to top {len(reranked_docs)}")
    return reranked_docs


def rerank_with_api(
    documents: list[Document],
    query: str,
    top_k: int,
    base_url: str,
    api_key: str,
    model: str,
) -> list[Document]:
    """Rerank documents using a generic OpenAI-compatible / Nvidia-compatible Rerank API."""
    import requests

    if not documents:
        return documents

    # Use NVIDIARerank if using Nvidia API
    if "nvidia.com" in base_url:
        try:
            from langchain_nvidia_ai_endpoints import NVIDIARerank
            
            client = NVIDIARerank(
                model=model,
                nvidia_api_key=api_key,
                base_url=base_url,
                top_n=top_k,
            )
            return client.compress_documents(
                query=query,
                documents=documents
            )
        except ImportError:
            logger.warning("langchain-nvidia-ai-endpoints not found, falling back to requests")

    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else "",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Format passes either as "passages" (Nvidia/Cohere style) or "documents"
    # We will use "passages" as it matches Nvidia NIM which is in the .env
    passages = [{"text": doc.page_content} for doc in documents]
    payload = {
        "model": model,
        "query": {"text": query},
        "passages": passages,
    }

    try:
        session = requests.Session()
        response = session.post(base_url, headers=headers, json=payload, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        
        # Parse rankings array from response
        # Cohere/Nvidia NIM trả về dạng: {"rankings": [{"index": 0, "relevance_score"/ "logit": 0.9}, ...]}
        rankings = data.get("rankings", [])
        
        # Sort and filter (giữ cẩn thận metadata ban đầu)
        reranked_docs = []
        # Fallback if API returns fewer results than requested, though top_k is applied here
        for rank in rankings[:top_k]:
            idx = rank.get("index")
            if idx is not None and 0 <= idx < len(documents):
                reranked_docs.append(documents[idx])
                
        logger.info(f"API reranked {len(documents)} documents to top {len(reranked_docs)}")
        return reranked_docs
    except Exception as e:
        logger.error(f"Error calling rerank API at {base_url}: {e}")
        # Fallback
        return documents[:top_k] if len(documents) > top_k else documents


def get_reranker(
    provider: RerankerProvider,
    top_k: int = 3,
    cohere_api_key: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> callable:
    """Get a reranker function based on the provider.

    Args:
        provider: The reranker provider to use.
        top_k: Number of top documents to return after reranking.
        cohere_api_key: Cohere API key (required if provider is "cohere").
        base_url: Rerank base URL for generic API.
        api_key: Rerank API key for generic API.
        model: Rerank model name for generic API.

    Returns:
        A callable that takes (documents, query) and returns reranked documents.

    """
    match provider:
        case "none":
            logger.info("Reranking disabled, using passthrough")
            return lambda docs, _: docs[:top_k] if len(docs) > top_k else docs

        case "cohere":
            if not cohere_api_key:
                msg = "Cohere API key is required for Cohere reranker"
                raise ValueError(msg)
            return lambda docs, query: rerank_with_cohere(docs, query, top_k, cohere_api_key)

        case "flashrank":
            # Pre-warm the model on startup
            _get_flashrank_ranker()
            return lambda docs, query: rerank_with_flashrank(docs, query, top_k)
            
        case "openai-compatible" | "api" | "custom":
            if not base_url:
                msg = "base_url is required for generic API reranker"
                raise ValueError(msg)
            return lambda docs, query: rerank_with_api(docs, query, top_k, base_url, api_key, model)

        case _:
            msg = f"Unknown reranker provider: {provider}"
            raise ValueError(msg)
