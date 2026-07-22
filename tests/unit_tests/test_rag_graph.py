import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from agent.backend.graph import Graph
from agent.backend.nodes.retrieval import retrieve_documents


@pytest.fixture
def graph_instance():
    return Graph()


@patch("agent.backend.nodes.retrieval.get_reranker")
@patch("agent.backend.nodes.retrieval.get_retriever")
def test_retrieve_documents(mock_get_retriever, mock_get_reranker, graph_instance):
    mock_get_reranker.return_value = lambda docs, query: docs
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="doc1", metadata={"source": "test.pdf"})]
    mock_get_retriever.return_value = mock_retriever

    state = {"messages": [HumanMessage(content="query")]}
    config = {"metadata": {"collection_name": "test_coll"}}

    result = retrieve_documents(state, config, cfg=graph_instance.cfg)

    assert result["query"] == "query"
    assert len(result["documents"]) == 1
    assert result["documents"][0].page_content == "doc1"
    mock_get_retriever.assert_called_once_with(k=graph_instance.cfg.retrieval_k, collection_name="test_coll", cfg=graph_instance.cfg)


@patch("agent.routes.rag.graph")
def test_rag_question_answer(mock_graph, client):
    mock_graph.with_config.return_value.ainvoke = AsyncMock(return_value={
        "query": "question",
        "documents": [Document(page_content="doc1", metadata={"source": "test"})],
    })

    payload = {
        "messages": [{"role": "user", "content": "question"}],
        "collection_name": "test"
    }

    response = client.post("/rag/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "question"
    assert len(data["documents"]) == 1
    assert data["documents"][0]["text"] == "doc1"


@patch("agent.routes.rag.graph")
def test_rag_stream(mock_graph, client):
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
        assert "Starting request..." in lines[0]
