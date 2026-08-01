"""Retrieval nodes for the graph."""

import hashlib

from langchain_core.documents import Document
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig
from loguru import logger

from agent.backend.state import AgentState
from agent.utils.config import Config
from agent.utils.reranker import get_reranker
from agent.utils.retriever import get_retriever


def _doc_key(doc: Document) -> str:
    """Stable dedupe key for a retrieved chunk."""
    if doc.metadata.get("global_id"):
        return str(doc.metadata["global_id"])
    if doc.metadata.get("_id"):
        return str(doc.metadata["_id"])
    if doc.metadata.get("id"):
        return str(doc.metadata["id"])
    # Fallback: hash page content.
    return hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()


def _build_queries(state: AgentState, base_query: str, *, cfg: Config) -> list[str]:
    """Build the set of queries used for retrieval (deduped, order-preserved).

    When query transformation is disabled this is just ``[base_query]`` —
    identical behaviour to the pre-transformation pipeline.
    """
    if not cfg.query_transform_enabled:
        return [base_query]

    queries: list[str] = [
        base_query,
        state.get("rewritten_query") or base_query,
        state.get("step_back_query") or base_query,
        *state.get("sub_queries", []),
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q = (q or "").strip()
        if q and q not in seen:
            unique.append(q)
            seen.add(q)
    return unique or [base_query]


def retrieve_documents(state: AgentState, config: RunnableConfig, *, cfg: Config) -> AgentState:
    """Retrieve documents (multi-query when enabled) and rerank them."""
    retry_count = state.get("retry_count", 0)
    k = cfg.retrieval_k if retry_count == 0 else cfg.retrieval_k_retry

    retriever = get_retriever(k=k, cfg=cfg)

    messages = convert_to_messages(messages=state["messages"])
    query = state.get("query") or messages[-1].content

    queries = _build_queries(state, query, cfg=cfg)
    if len(queries) > 1:
        logger.debug(f"Retrieving with {len(queries)} expanded queries")

    seen_keys: set[str] = set()
    relevant_documents: list[Document] = []
    for q in queries:
        for doc in retriever.invoke(q):
            key = _doc_key(doc)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            relevant_documents.append(doc)

    if not relevant_documents:
        logger.info(f"No relevant documents found for the query: {query}")

    # Rerank always scores against the ORIGINAL user query, never a
    # transformed variant, so the reranker measures true relevance.
    if relevant_documents and cfg.rerank_provider != "none":
        reranker_fn = get_reranker(cfg, top_k=cfg.rerank_top_k)
        relevant_documents = reranker_fn(relevant_documents, query)

    return {"query": query, "documents": relevant_documents, "retry_count": retry_count}
