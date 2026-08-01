"""Query transformation node (Qwen self-host, OpenAI-compatible).

Applies three query-transformation techniques (adapted from
``example/query_transformations.py``, prompts kept verbatim):

1. **Rewrite** — reformulate the query to be more specific/detailed.
2. **Step-back** — generate a broader query for background context.
3. **Decompose** — split a complex query into 2-4 simpler sub-queries.

The three prompts run in parallel via ``RunnableParallel`` (one round,
three requests) to save latency. Any LLM failure (or the feature being
disabled via ``QUERY_TRANSFORM_ENABLED=0``) falls back to the original
query so the retrieval pipeline never breaks because of this layer.
"""

from langchain_core.messages import convert_to_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableParallel
from langchain_openai import ChatOpenAI
from loguru import logger

from agent.backend.state import AgentState
from agent.utils.config import Config

QUERY_REWRITE_TEMPLATE = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""

STEP_BACK_TEMPLATE = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

Original query: {original_query}

Step-back query:"""

SUBQUERY_DECOMPOSITION_TEMPLATE = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

Original query: {original_query}

example: What are the impacts of climate change on the environment?

Sub-queries:
1. What are the impacts of climate change on biodiversity?
2. How does climate change affect the oceans?
3. What are the effects of climate change on agriculture?
4. What are the impacts of climate change on human health?"""


def _parse_sub_queries(raw: str) -> list[str]:
    """Split the decompose raw response into clean sub-questions."""
    return [
        q.strip()
        for q in raw.split("\n")
        if q.strip() and not q.strip().startswith("Sub-queries:")
    ]


def transform_query(state: AgentState, config: RunnableConfig, *, cfg: Config) -> dict:
    """Transform the user query into rewritten / step-back / sub-queries."""
    messages = convert_to_messages(messages=state["messages"])
    original_query = state.get("query") or messages[-1].content

    fallback = {
        "original_query": original_query,
        "query": original_query,
        "rewritten_query": original_query,
        "step_back_query": original_query,
        "sub_queries": [original_query],
    }

    if not cfg.query_transform_enabled:
        return fallback

    try:
        llm = ChatOpenAI(
            model=cfg.qwen_model,
            base_url=cfg.qwen_base_url,
            api_key=cfg.qwen_api_key,
            temperature=0,
        )
        rewrite_chain = PromptTemplate(input_variables=["original_query"], template=QUERY_REWRITE_TEMPLATE) | llm
        step_back_chain = PromptTemplate(input_variables=["original_query"], template=STEP_BACK_TEMPLATE) | llm
        decompose_chain = (
            PromptTemplate(input_variables=["original_query"], template=SUBQUERY_DECOMPOSITION_TEMPLATE) | llm
        )

        # One parallel round instead of three sequential LLM calls.
        combined = RunnableParallel(
            rewrite=rewrite_chain,
            step_back=step_back_chain,
            decompose=decompose_chain,
        )
        results = combined.invoke({"original_query": original_query})
        rewritten_resp, step_back_resp, decompose_resp = (
            results["rewrite"],
            results["step_back"],
            results["decompose"],
        )

        rewritten = rewritten_resp.content.strip() or original_query
        step_back = step_back_resp.content.strip() or original_query
        sub_queries = _parse_sub_queries(decompose_resp.content) or [original_query]

        logger.debug(
            "Query transformation done: rewritten={!r}, step_back={!r}, {} sub-queries",
            rewritten,
            step_back,
            len(sub_queries),
        )

        return {
            "original_query": original_query,
            # `query` = the query used downstream as the primary/rerank anchor.
            "query": original_query,
            "rewritten_query": rewritten,
            "step_back_query": step_back,
            "sub_queries": sub_queries,
        }

    except Exception as exc:  # noqa: BLE001 — never break retrieval because of this layer
        logger.warning("Query transformation failed, falling back to original query: {!r}", exc)
        return fallback
