"""Optional remote reranker utilities (BGE-reranker-v2-m3 qua HTTP).

Không chạy local reranker (tránh tải model về cache Docker). Rerank được delegate
tới HTTP endpoint ngoài (cùng Colab server với embedding) qua biến ``RERANK_BASE_URL``.

Endpoint contract (xem notebook ``rag_test_bge_m3_reranker_ngrok.ipynb``):
    POST {base_url}/rerank
        body: {"query": str, "documents": [...], "top_k": int|null, "normalize": true}
        resp: {"results": [{"index": int, "document": str, "score": float}, ...]}
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal

import httpx
from langchain_core.documents import Document
from loguru import logger

from agent.utils.config import Config

_REMOTE_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "60"))

RerankerProvider = Literal["none", "remote"]
RerankerFn = Callable[[list[Document], str], list[Document]]


def rerank_with_remote(
    documents: list[Document],
    query: str,
    *,
    top_k: int,
    base_url: str,
    timeout: float = _REMOTE_TIMEOUT,
) -> list[Document]:
    """Rerank documents bằng remote /rerank endpoint (BGE-reranker-v2-m3)."""
    if not documents:
        return documents
    if top_k <= 0 or len(documents) <= top_k:
        top_k = len(documents)

    url = f"{base_url.rstrip('/')}/rerank"
    payload: dict[str, Any] = {
        "query": query,
        "documents": [d.page_content for d in documents],
        "top_k": top_k,
        "normalize": True,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    results = data.get("results", [])
    ranked = [documents[item["index"]] for item in results if 0 <= item["index"] < len(documents)]
    logger.info(f"Remote reranked {len(documents)} documents to top {len(ranked)}")
    return ranked


def get_reranker(
    cfg: Config,
    *,
    top_k: int | None = None,
) -> RerankerFn:
    """Return a reranking callable cho provider trong cfg.

    Providers:
    - ``none``: passthrough (truncate top_k).
    - ``remote``: HTTP tới ``RERANK_BASE_URL`` (Colab ngrok server).
    """
    normalized = (cfg.rerank_provider or "none").strip().lower()
    k = top_k if top_k is not None else cfg.rerank_top_k

    if normalized == "none":
        def passthrough(docs: list[Document], _query: str) -> list[Document]:
            return docs[:k]
        return passthrough

    if normalized == "remote":
        if not cfg.rerank_base_url:
            msg = "RERANK_BASE_URL is required when RERANK_PROVIDER=remote."
            raise ValueError(msg)
        base_url = cfg.rerank_base_url

        def remote_rerank(docs: list[Document], query: str) -> list[Document]:
            return rerank_with_remote(docs, query, top_k=k, base_url=base_url)
        return remote_rerank

    msg = f"Unknown reranker provider: {cfg.rerank_provider!r}. Supported: 'none', 'remote'."
    raise ValueError(msg)
