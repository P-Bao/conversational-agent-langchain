"""Retrieval nodes for the graph."""

from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig
from loguru import logger

from agent.backend.state import AgentState
from agent.utils.config import Config
from agent.utils.reranker import get_reranker
from agent.utils.retriever import get_retriever


def retrieve_documents(state: AgentState, config: RunnableConfig, *, cfg: Config) -> AgentState:
    """Retrieve documents from the retriever and rerank them."""
    retry_count = state.get("retry_count", 0)
    k = cfg.retrieval_k if retry_count == 0 else cfg.retrieval_k_retry

    collection_name = config.get("metadata", {}).get("collection_name") or cfg.qdrant_collection_name
    retriever = get_retriever(k=k, collection_name=collection_name, cfg=cfg)

    messages = convert_to_messages(messages=state["messages"])
    query = state.get("query") or messages[-1].content

    relevant_documents = retriever.invoke(query)
    if not relevant_documents:
        logger.info(f"No relevant documents found for the query: {query}")

    if relevant_documents and cfg.rerank_provider != "none":
        reranker_fn = get_reranker(
            provider=cfg.rerank_provider,
            top_k=cfg.rerank_top_k,
            model_name=cfg.rerank_model,
        )
        relevant_documents = reranker_fn(relevant_documents, query)

    return {"query": query, "documents": relevant_documents, "retry_count": retry_count}
