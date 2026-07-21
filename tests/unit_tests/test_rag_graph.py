import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

with patch("agent.backend.graph.ChatLiteLLM") as _mock_llm_cls:
    _mock_llm_cls.return_value = MagicMock()
    from agent.backend.graph import Graph
    from agent.backend.state import AgentState, Grade

# --- Tests for Graph Class ---

@pytest.fixture
def graph_instance():
    with patch("agent.backend.graph.ChatLiteLLM") as mock_llm_cls:
        mock_llm_cls.return_value = MagicMock()
        yield Graph()

from agent.backend.nodes.retrieval import retrieve_documents, retrieve_documents_with_chat_history, get_chat_history
from agent.backend.nodes.grading import grade_documents
from agent.backend.nodes.rewrite import rewrite_query
from agent.backend.nodes.generation import generate_response_default

def test_route_to_retriever_single_message(graph_instance):
    state = {"messages": [HumanMessage(content="hi")]}
    result = graph_instance.route_to_retriever(state)
    assert result == "retriever"

def test_route_to_retriever_multi_message(graph_instance):
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello"), HumanMessage(content="bye")]}
    result = graph_instance.route_to_retriever(state)
    assert result == "retriever_with_chat_history"

def test_get_chat_history(graph_instance):
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
        HumanMessage(content="bye")
    ]
    history = get_chat_history(messages)
    assert len(history) == 3
    assert history[0]["role"] == "human"
    assert history[1]["role"] == "ai"

def test_route_to_response_synthesizer_default(graph_instance):
    # route_to_response_synthesizer was removed in commit 938772f; routing is now done by grade_documents.
    pytest.skip("route_to_response_synthesizer removed in refactor #157")

def test_route_to_response_synthesizer_cohere(graph_instance):
    pytest.skip("route_to_response_synthesizer removed in refactor #157")

@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents(mock_get_retriever, graph_instance):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="doc1")]
    mock_get_retriever.return_value = mock_retriever

    state = {"messages": [HumanMessage(content="query")]}
    config = {"metadata": {"collection_name": "test_coll"}}

    result = retrieve_documents(state, config, cfg=graph_instance.cfg)

    assert result["query"] == "query"
    assert len(result["documents"]) == 1
    assert result["documents"][0].page_content == "doc1"
    mock_get_retriever.assert_called_with(collection_name="test_coll", k=graph_instance.cfg.retrieval_k)

@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents_with_chat_history(mock_get_retriever, graph_instance):
    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="doc1")]
    mock_get_retriever.return_value = mock_retriever

    # Mock the LLM and the chain
    graph_instance.llm = MagicMock()

    # We need to mock the chain that is constructed inside the method:
    # condense_question_chain = (condense_queston_prompt | model | StrOutputParser()).with_config(...)
    # retriever_with_condensed_question = condense_question_chain | retriever

    # Since we can't easily mock the internal chain construction without patching the classes,
    # we will patch the classes involved in the chain construction to return mocks that we can control.

    with patch("agent.backend.nodes.retrieval.PromptTemplate.from_template") as mock_prompt, \
         patch("agent.backend.nodes.rewrite.StrOutputParser") as mock_parser:

        # Setup the mock chain components
        mock_prompt_instance = MagicMock()
        mock_prompt.return_value = mock_prompt_instance

        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance

        # The chain is: prompt | model | parser
        # We mock the pipe operations
        mock_chain_step1 = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_chain_step1

        mock_chain_step2 = MagicMock()
        mock_chain_step1.__or__.return_value = mock_chain_step2

        mock_condense_question_chain = MagicMock()
        mock_chain_step2.with_config.return_value = mock_condense_question_chain

        # The final chain is: condense_question_chain | retriever
        mock_final_chain = MagicMock()
        mock_condense_question_chain.__or__.return_value = mock_final_chain

        mock_final_chain.invoke.return_value = [Document(page_content="doc1")]

        state = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
                HumanMessage(content="followup")
            ]
        }
        config = {"metadata": {"collection_name": "test_coll"}}

        result = retrieve_documents_with_chat_history(state, config, cfg=graph_instance.cfg, llm=graph_instance.llm)

        # Verify the result
        assert result["query"] == "followup"
        assert len(result["documents"]) == 1
        assert result["documents"][0].page_content == "doc1"

        # Verify that the final chain was invoked with the expected input
        # The input to the final chain is {"question": query, "chat_history": ...}
        args, _ = mock_final_chain.invoke.call_args
        assert args[0]["question"] == "followup"
        assert len(args[0]["chat_history"]) == 2


def test_grade_documents(graph_instance):
    # grade_documents builds: chain = prompt | model | parser
    # chain.invoke(...) returns a Grade. We mock prompt | model to be a MagicMock
    # whose __or__ returns the same chain so .invoke is controllable.
    graph_instance.llm = MagicMock()

    mock_chain = MagicMock()

    with patch("agent.backend.nodes.grading.PromptTemplate") as mock_prompt_cls, \
         patch("langchain_core.output_parsers.PydanticOutputParser") as mock_parser_cls:
        mock_prompt_instance = MagicMock()
        mock_prompt_cls.return_value = mock_prompt_instance
        mock_parser_cls.return_value = MagicMock()

        # prompt | model -> mock_chain
        mock_prompt_instance.__or__.return_value = mock_chain
        # mock_chain | parser -> mock_chain (same object, so .invoke stays controllable)
        mock_chain.__or__.return_value = mock_chain

        state = {
            "documents": [Document(page_content="doc1")],
            "query": "test query",
            "retry_count": 0,
        }
        config = {"configurable": {"model_name": "gemini"}}

        # Case 1: Relevant documents
        mock_chain.invoke.return_value = Grade(is_relevant=True)
        assert grade_documents(state, config, llm=graph_instance.llm) == "response_synthesizer"

        # Case 2: Irrelevant documents, retry_count = 0 -> rewrite
        mock_chain.invoke.return_value = Grade(is_relevant=False)
        state["retry_count"] = 0
        assert grade_documents(state, config, llm=graph_instance.llm) == "rewrite_query"

        # Case 3: Irrelevant documents but max retries reached -> response_synthesizer
        mock_chain.invoke.return_value = Grade(is_relevant=False)
        state["retry_count"] = 2
        assert grade_documents(state, config, llm=graph_instance.llm) == "response_synthesizer"


def test_rewrite_query(graph_instance):
    # Mock the LLM
    graph_instance.llm = MagicMock()
    mock_model_with_config = MagicMock()
    graph_instance.llm.with_config.return_value = mock_model_with_config

    # Mock the chain
    mock_chain = MagicMock()

    with patch("agent.backend.nodes.rewrite.PromptTemplate") as mock_prompt_cls, \
         patch("agent.backend.nodes.rewrite.StrOutputParser") as mock_parser:

        mock_prompt_instance = MagicMock()
        mock_prompt_cls.return_value = mock_prompt_instance

        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance

        # Chain: prompt | model | parser
        # 1. prompt | model -> step1
        mock_chain_step1 = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_chain_step1

        # 2. step1 | parser -> chain
        # Note: StrOutputParser() is the argument to __or__
        mock_chain_step1.__or__.return_value = mock_chain

        mock_chain.invoke.return_value = "rewritten query"

        state = {
            "query": "original query",
            "retry_count": 0
        }

        result = rewrite_query(state, llm=graph_instance.llm)

        assert result["query"] == "rewritten query"
        assert result["retry_count"] == 1


def test_generate_response(graph_instance):
    # Mock the LLM
    graph_instance.llm = MagicMock()

    state = {
        "query": "test query",
        "documents": [Document(page_content="doc1")],
        "messages": [HumanMessage(content="test query")]
    }

    # Test generate_response_default only. generate_response_cohere was removed in refactor #157.
    with patch("agent.backend.nodes.generation.ChatPromptTemplate.from_messages") as mock_prompt_cls:
        mock_prompt_instance = MagicMock()
        mock_prompt_cls.return_value = mock_prompt_instance

        mock_chain = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_chain
        mock_chain.invoke.return_value = AIMessage(content="default response")

        result = generate_response_default(state, llm=graph_instance.llm)
        assert result["messages"][0].content == "default response"

# --- Tests for RAG Routes ---

@patch("agent.routes.rag.graph")
def test_rag_question_answer(mock_graph, client):
    # Mock the graph.ainvoke method
    mock_graph.with_config.return_value.ainvoke = AsyncMock(return_value={
        "documents": [Document(page_content="doc1", metadata={"source": "test"})],
        "messages": [AIMessage(content="The answer")]
    })

    payload = {
        "messages": [{"role": "user", "content": "question"}],
        "collection_name": "test"
    }

    response = client.post("/rag/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The answer"
    assert len(data["meta_data"]) == 1
    assert data["meta_data"][0]["document"][0] == "doc1"

@patch("agent.routes.rag.graph")
def test_rag_stream(mock_graph, client):
    # Mock the graph.astream_events method
    async def mock_stream(*args, **kwargs):
        yield {
            "event": "on_chain_start",
            "name": "retriever",
            "data": {}
        }
        yield {
            "event": "on_chain_end",
            "name": "retriever",
            "data": {"output": {"documents": [Document(page_content="doc1")]}}
        }
        yield {
            "event": "on_chat_model_start",
            "name": "model",
            "data": {}
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "response_synthesizer"},
            "data": {"chunk": AIMessage(content="chunk1")}
        }
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"documents": [Document(page_content="doc1", metadata={"s": "1"})]}}
        }

    mock_graph.with_config.return_value.astream_events = mock_stream

    payload = {
        "messages": [{"role": "user", "content": "question"}],
        "collection_name": "test"
    }

    with client.stream("POST", "/rag/stream", json=payload) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())
        assert len(lines) > 0
        # Verify we got some expected events
        assert "Starting request..." in lines[0]
        # We can check for specific content in the lines
