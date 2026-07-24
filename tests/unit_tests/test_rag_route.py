from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document


pytestmark = pytest.mark.anyio


@patch("agent.routes.rag.graph")
def test_rag_question_answer_returns_retrieval_response(mock_graph, client) -> None:
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "query": "question",
            "documents": [
                Document(page_content="doc1", metadata={"source": "test.pdf", "page": 1}),
                Document(page_content="doc2", metadata={"source": "test.pdf", "page": 2, "score": 0.9}),
            ],
        }
    )

    response = client.post(
        "/rag/",
        json={"messages": [{"role": "user", "content": "question"}]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "question"
    assert len(data["documents"]) == 2
    assert data["documents"][0]["text"] == "doc1"
    assert data["documents"][0]["source"] == "test.pdf"
    assert data["documents"][1]["score"] == 0.9


@patch("agent.routes.rag.graph")
def test_rag_question_answer_empty_documents(mock_graph, client) -> None:
    mock_graph.ainvoke = AsyncMock(
        return_value={"query": "question", "documents": []}
    )

    response = client.post(
        "/rag/",
        json={"messages": [{"role": "user", "content": "question"}]},
    )

    assert response.status_code == 200
    assert response.json() == {"query": "question", "documents": []}


@patch("agent.routes.rag.graph")
def test_rag_question_answer_invokes_graph_without_per_request_collection(mock_graph, client) -> None:
    """Collection is no longer per-request; verify graph.ainvoke is called directly."""
    mock_graph.ainvoke = AsyncMock(
        return_value={"query": "q", "documents": []}
    )

    client.post(
        "/rag/",
        json={"messages": [{"role": "user", "content": "q"}]},
    )

    mock_graph.ainvoke.assert_awaited_once_with({"messages": [{"role": "user", "content": "q"}]})


@patch("agent.routes.rag.graph")
def test_rag_stream_emits_ndjson_events(mock_graph, client) -> None:
    async def mock_stream(*_args: Any, **_kwargs: Any):
        yield {
            "event": "on_chain_start",
            "name": "retriever",
            "data": {},
        }
        yield {
            "event": "on_chain_end",
            "name": "retriever",
            "data": {"output": {"documents": [Document(page_content="doc1")]}},
        }
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {
                "output": {
                    "documents": [Document(page_content="doc1", metadata={"source": "a.pdf"})]
                }
            },
        }

    mock_graph.astream_events = mock_stream

    payload = {
        "messages": [{"role": "user", "content": "question"}],
    }

    with client.stream("POST", "/rag/stream", json=payload) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    parsed = [json.loads(line) for line in lines]
    statuses = [p for p in parsed if p.get("type") == "status"]
    documents = [p for p in parsed if p.get("type") == "documents"]

    assert any(s["data"] == "Starting request..." for s in statuses)
    assert any(s["data"] == "Done." for s in statuses)
    assert any("Found 1 documents" in s["data"] for s in statuses)
    assert documents and documents[0]["data"][0]["text"] == "doc1"
