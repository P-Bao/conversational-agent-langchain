"""Defining the graph."""

import functools
from typing import Literal

from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph

from agent.backend.nodes.generation import generate_response_default
from agent.backend.nodes.grading import grade_documents
from agent.backend.nodes.retrieval import retrieve_documents, retrieve_documents_with_chat_history
from agent.backend.nodes.rewrite import rewrite_query
from agent.backend.state import AgentState
from agent.utils.config import Config

settings = Config()


class Graph:
    """The LangGraph Graph."""

    def __init__(self) -> None:
        """Initialize the Graph."""
        self.cfg = settings

        self.llm = ChatLiteLLM(
            model_name=self.cfg.generation_model,
            api_base=self.cfg.llm_base_url,
            api_key=self.cfg.llm_api_key,
            streaming=True,
        )

    def route_to_retriever(
        self,
        state: AgentState,
    ) -> Literal["retriever", "retriever_with_chat_history"]:
        """Route to the appropriate retriever based on the state."""
        if len(state["messages"]) == 1:
            return "retriever"
        else:
            return "retriever_with_chat_history"

    def build_graph(self) -> StateGraph:
        """Build the graph for the agent.

        Pipeline:
            retriever -> grade_documents -> {response_synthesizer | rewrite_query}
            rewrite_query -> retriever (loop)
        """
        workflow = StateGraph(state_schema=AgentState)

        # define nodes
        workflow.add_node("retriever", functools.partial(retrieve_documents, cfg=self.cfg))
        workflow.add_node("retriever_with_chat_history", functools.partial(retrieve_documents_with_chat_history, cfg=self.cfg, llm=self.llm))
        workflow.add_node("rewrite_query", functools.partial(rewrite_query, llm=self.llm))
        workflow.add_node("response_synthesizer", functools.partial(generate_response_default, llm=self.llm))

        # set entry point to retrievers
        workflow.set_conditional_entry_point(path=self.route_to_retriever)

        # connect retrievers to grader
        workflow.add_conditional_edges(
            source="retriever",
            path=functools.partial(grade_documents, llm=self.llm),
            path_map={"response_synthesizer": "response_synthesizer", "rewrite_query": "rewrite_query"},
        )
        workflow.add_conditional_edges(
            source="retriever_with_chat_history",
            path=functools.partial(grade_documents, llm=self.llm),
            path_map={"response_synthesizer": "response_synthesizer", "rewrite_query": "rewrite_query"},
        )

        # connect rewriter back to retriever (loop)
        workflow.add_edge("rewrite_query", "retriever")

        # connect synthesizer to terminal node
        workflow.add_edge(start_key="response_synthesizer", end_key=END)

        return workflow.compile()
