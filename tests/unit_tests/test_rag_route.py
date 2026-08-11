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
    # Top-level page/source are removed from API output; only metadata retains them.
    assert "page" not in data["documents"][0]
    assert "source" not in data["documents"][0]
    assert data["documents"][0]["metadata"]["source"] == "test.pdf"
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

    mock_graph.ainvoke.assert_awaited_once_with({"messages": [{"role": "user", "content": "q"}], "top_k": None})


@patch("agent.routes.rag.graph")
def test_rag_question_answer_forwards_top_k_to_graph(mock_graph, client) -> None:
    """When the client passes ``top_k`` it's forwarded into the graph state."""
    mock_graph.ainvoke = AsyncMock(
        return_value={"query": "q", "documents": []}
    )

    response = client.post(
        "/rag/",
        json={"messages": [{"role": "user", "content": "q"}], "top_k": 7},
    )

    assert response.status_code == 200
    args, _kwargs = mock_graph.ainvoke.call_args
    assert args[0]["top_k"] == 7


@patch("agent.routes.rag.graph")
def test_rag_question_answer_default_top_k_is_none(mock_graph, client) -> None:
    """When the client omits ``top_k``, the graph receives ``None``."""
    mock_graph.ainvoke = AsyncMock(
        return_value={"query": "q", "documents": []}
    )

    client.post(
        "/rag/",
        json={"messages": [{"role": "user", "content": "q"}]},
    )

    args, _kwargs = mock_graph.ainvoke.call_args
    assert args[0]["top_k"] is None


@patch("agent.routes.rag.graph")
def test_rag_question_answer_top_k_validation_above_max(mock_graph, client) -> None:
    """``top_k`` > 40 bị Pydantic validate reject (le=40)."""
    response = client.post(
        "/rag/",
        json={"messages": [{"role": "user", "content": "q"}], "top_k": 41},
    )
    assert response.status_code == 422


@patch("agent.routes.rag.graph")
def test_rag_question_answer_top_k_validation_below_min(mock_graph, client) -> None:
    """``top_k`` < 1 bị Pydantic validate reject (ge=1)."""
    response = client.post(
        "/rag/",
        json={"messages": [{"role": "user", "content": "q"}], "top_k": 0},
    )
    assert response.status_code == 422


@patch("agent.routes.rag.graph")
def test_rag_question_answer_top_k_at_boundary(mock_graph, client) -> None:
    """``top_k = 1`` và ``top_k = 40`` đều hợp lệ."""
    mock_graph.ainvoke = AsyncMock(
        return_value={"query": "q", "documents": []}
    )
    for k in (1, 40):
        response = client.post(
            "/rag/",
            json={"messages": [{"role": "user", "content": "q"}], "top_k": k},
        )
        assert response.status_code == 200, k


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
