"""Defining the retrieval-only graph."""

import functools

from langgraph.graph import END, StateGraph

from agent.backend.nodes.retrieval import retrieve_documents
from agent.backend.state import AgentState
from agent.utils.config import Config

settings = Config()


class Graph:
    """The retrieval-only LangGraph Graph."""

    def __init__(self, cfg: Config | None = None) -> None:
        """Initialize the Graph."""
        self.cfg = cfg or settings

    def build_graph(self) -> StateGraph:
        """Build the retrieval-only graph.

        Pipeline:
            entry -> retriever -> END
        """
        workflow = StateGraph(state_schema=AgentState)

        workflow.add_node("retriever", functools.partial(retrieve_documents, cfg=self.cfg))
        workflow.set_entry_point("retriever")
        workflow.add_edge("retriever", END)

        return workflow.compile()
