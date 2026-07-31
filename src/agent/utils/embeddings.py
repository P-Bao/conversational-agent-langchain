"""Embedding model utilities — remote BGE-m3 (dense + sparse).

Không chạy local BGE-m3 (tránh tải model ~2.7GB về cache Docker, gây build nặng /
lỗi). Toàn bộ embedding được delegate tới HTTP endpoint ngoài (embedding-server)
qua biến ``EMBEDDING_BASE_URL``.

Endpoint contract (xem notebook ``rag_test_bge_m3_reranker_ngrok.ipynb``):
    POST {base_url}/embed
        body: {"texts": [...], "return_dense": true, "return_sparse": true}
        resp: {"dense_vecs": [[float, ...], ...], "sparse_vecs": [{"indices": [int, ...], "values": [float, ...]}, ...]}

Sparse embedding: BGE-M3 trả cả dense và sparse, dùng cho hybrid retrieval.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from loguru import logger

from agent.utils.config import Config

_REMOTE_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "60"))


class BGEM3RemoteEmbeddings(Embeddings):
    """LangChain ``Embeddings`` wrapper gọi remote BGE-m3 endpoint (dense + sparse)."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = _REMOTE_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._last_sparse_vecs: list[dict] | None = None

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/embed"
        with httpx.Client(timeout=self._timeout, headers=self._headers()) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._post({"texts": texts, "return_dense": True, "return_sparse": True})
        self._last_sparse_vecs = data.get("sparse_vecs")
        return [list(map(float, vec)) for vec in data["dense_vecs"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def get_last_sparse_vecs(self) -> list[dict] | None:
        """Return sparse vectors from the last embed_documents/embed_query call."""
        return self._last_sparse_vecs


def get_embedding_model(cfg: Config) -> Embeddings:
    """Return dense embeddings client (remote BGE-m3 endpoint)."""
    if not cfg.embedding_base_url:
        msg = "EMBEDDING_BASE_URL is required (remote provider — không chạy local BGE)."
        raise ValueError(msg)
    logger.info(f"Using remote BGE-m3 embedding endpoint: {cfg.embedding_base_url} (sparse enabled)")
    return BGEM3RemoteEmbeddings(cfg.embedding_base_url, api_key=cfg.embedding_api_key)
