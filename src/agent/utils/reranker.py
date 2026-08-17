"""Optional reranker utilities (BGE-reranker-v2-m3).

Supports three modes:
- ``remote`` (default): delegate tới HTTP rerank server (``RERANK_BASE_URL``,
  vd ``http://127.0.0.1:8010``). Hỗ trợ các định dạng response:
    1. {"scores": [float, ...], "ranked_indices": [int, ...]}
       ``scores`` theo thứ tự input documents; ``ranked_indices`` đã sort giảm dần + áp ``top_k``.
    2. {"scores": [float, ...]}
       Server chỉ trả raw scores theo thứ tự input docs -> client tự sort desc và lấy ``top_k``.
    3. {"results": [{"index": int, "score": float}, ...]} (legacy contract).
- ``bge`` (fallback): run locally via ``FlagEmbedding.FlagReranker`` (cần GPU).
- ``none``: passthrough (truncate to top_k).

Local BGE reranker loads model on first use and caches it in memory.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal

import torch
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document
from loguru import logger

from agent.utils.config import Config

_REMOTE_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "60"))

RerankerProvider = Literal["none", "remote", "bge"]
RerankerFn = Callable[[list[Document], str], list[Document]]

_reranker_cache: FlagReranker | None = None


def _get_local_reranker(model_name: str) -> FlagReranker:
    """Get or create cached FlagReranker instance."""
    global _reranker_cache
    if _reranker_cache is None:
        use_fp16 = torch.cuda.is_available()
        logger.info(f"Loading local BGE reranker: {model_name} (fp16={use_fp16})")
        _reranker_cache = FlagReranker(model_name, use_fp16=use_fp16)
    return _reranker_cache


def rerank_with_bge(
    documents: list[Document],
    query: str,
    *,
    top_k: int,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> list[Document]:
    """Rerank documents using local BGE-reranker-v2-m3 via FlagEmbedding.

    Each returned document gets ``score`` written into ``metadata`` so the API
    can expose it (previously scores were only used for sorting and then
    dropped, which made the API's ``score`` field always null).
    """
    if not documents:
        return documents
    if top_k <= 0 or len(documents) <= top_k:
        top_k = len(documents)

    reranker = _get_local_reranker(model_name)
    pairs = [[query, d.page_content] for d in documents]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_k]
    ranked: list[Document] = []
    for i in order:
        doc = documents[i]
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["score"] = float(scores[i])
        ranked.append(doc)
    logger.info(f"Local reranked {len(documents)} documents to top {len(ranked)}")
    return ranked


def rerank_with_remote(
    documents: list[Document],
    query: str,
    *,
    top_k: int,
    base_url: str,
    timeout: float = _REMOTE_TIMEOUT,
) -> list[Document]:
    """Rerank documents by remote /rerank endpoint (rerank server, vd :8010).

    Response format hỗ trợ:
    1. {"scores": [float, ...], "ranked_indices": [int, ...]}
       ``scores`` theo thứ tự input documents; ``ranked_indices`` đã sort giảm dần + áp ``top_k``.
    2. {"scores": [float, ...]}
       Chỉ trả mảng điểm -> client tự sắp xếp giảm dần và lấy ``top_k``.
    3. {"results": [{"index": int, "score": float}, ...]} (legacy fallback).

    Fail-fast: lỗi HTTP/timeout được raise (không auto-fallback sang local).
    """
    if not documents:
        return documents
    if top_k <= 0 or len(documents) <= top_k:
        top_k = len(documents)

    import httpx

    url = f"{base_url.rstrip('/')}/rerank"
    payload: dict[str, Any] = {
        "query": query,
        "documents": [d.page_content for d in documents],
        "top_k": top_k,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    # Build index → score map and determine ranking order
    scores: list[float] = data.get("scores") or []
    indices = data.get("ranked_indices")

    if indices is not None and len(indices) > 0:
        # Contract đầy đủ {"scores", "ranked_indices"}
        scores_map: dict[int, float | None] = {
            i: scores[i] for i in range(min(len(scores), len(documents)))
        }
    elif "results" in data:
        # Fallback contract cũ {"results": [{"index", "score"}]}
        results = data.get("results", [])
        indices = [it.get("index") for it in results]
        scores_map = {
            it.get("index"): it.get("score") for it in results
        }
    elif scores:
        # Response dạng {"scores": [...]} -> client tự sort desc theo score và lấy top_k
        valid_len = min(len(scores), len(documents))
        indices = sorted(range(valid_len), key=lambda i: scores[i], reverse=True)[:top_k]
        scores_map = {i: scores[i] for i in range(valid_len)}
    else:
        # Khi server trả về rỗng không có scores hay ranked_indices
        logger.warning(
            f"Remote rerank returned empty ranked_indices for {len(documents)} docs (top_k={top_k}); "
            "client-side fallback ordering applied"
        )
        indices = list(range(min(top_k, len(documents))))
        scores_map = {}

    ranked: list[Document] = []
    for idx in indices:
        if idx is None or not (0 <= idx < len(documents)):
            continue
        doc = documents[idx]
        if doc.metadata is None:
            doc.metadata = {}
        s = scores_map.get(idx)
        if s is not None:
            doc.metadata["score"] = float(s)
        ranked.append(doc)
    logger.info(f"Remote reranked {len(documents)} documents to top {len(ranked)}")
    return ranked


def get_reranker(
    cfg: Config,
    *,
    top_k: int | None = None,
) -> RerankerFn:
    """Return a reranking callable for the provider in cfg.

    Providers:
    - ``remote`` (default): HTTP tới ``RERANK_BASE_URL`` (rerank server, vd :8010).
      Fail-fast — không tự động fallback sang local khi server lỗi.
    - ``bge`` (fallback): local FlagEmbedding.FlagReranker (BAAI/bge-reranker-v2-m3).
    - ``none``: passthrough (truncate to top_k).
    """
    normalized = (cfg.rerank_provider or "none").strip().lower()
    k = top_k if top_k is not None else cfg.rerank_top_k
    model_name = cfg.rerank_model

    if normalized == "remote":
        base_url = cfg.rerank_base_url
        if not base_url:
            msg = "RERANK_BASE_URL is required when RERANK_PROVIDER=remote."
            raise ValueError(msg)

        def remote_rerank(docs: list[Document], query: str) -> list[Document]:
            return rerank_with_remote(docs, query, top_k=k, base_url=base_url)
        return remote_rerank

    if normalized == "bge":
        def local_rerank(docs: list[Document], query: str) -> list[Document]:
            return rerank_with_bge(docs, query, top_k=k, model_name=model_name)
        return local_rerank

    if normalized == "none":
        def passthrough(docs: list[Document], _query: str) -> list[Document]:
            return docs[:k]
        return passthrough

    msg = f"Unknown reranker provider: {cfg.rerank_provider!r}. Supported: 'none', 'remote', 'bge'."
    raise ValueError(msg)
