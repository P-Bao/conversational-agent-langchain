"""Defining the retrieval graph (with optional query transformation)."""

import functools

from langgraph.graph import END, StateGraph

from agent.backend.nodes.query_transform import transform_query
from agent.backend.nodes.retrieval import retrieve_documents
from agent.backend.state import AgentState
from agent.utils.config import Config

settings = Config()


class Graph:
    """The retrieval LangGraph Graph."""

    def __init__(self, cfg: Config | None = None) -> None:
        """Initialize the Graph."""
        self.cfg = cfg or settings

    def build_graph(self) -> StateGraph:
        """Build the retrieval graph.

        Pipeline:
            QUERY_TRANSFORM_ENABLED=1: entry -> query_transform -> retriever -> END
            QUERY_TRANSFORM_ENABLED=0: entry -> retriever -> END  (legacy)
        """
        workflow = StateGraph(state_schema=AgentState)

        if self.cfg.query_transform_enabled:
            workflow.add_node("query_transform", functools.partial(transform_query, cfg=self.cfg))
            workflow.add_node("retriever", functools.partial(retrieve_documents, cfg=self.cfg))
            workflow.set_entry_point("query_transform")
            workflow.add_edge("query_transform", "retriever")
            workflow.add_edge("retriever", END)
        else:
            workflow.add_node("retriever", functools.partial(retrieve_documents, cfg=self.cfg))
            workflow.set_entry_point("retriever")
            workflow.add_edge("retriever", END)

        return workflow.compile()
