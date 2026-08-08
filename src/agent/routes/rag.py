"""The RAG Routes."""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agent.backend.graph import Graph
from agent.data_model.request_data_model import RAGRequest
from agent.data_model.response_data_model import RetrievalResponse, RetrievedDoc

graph = Graph().build_graph()


router = APIRouter()


@router.post("/", tags=["rag"])
async def question_answer(rag: RAGRequest) -> RetrievalResponse:
    """Retrieving relevant documents for the query."""
    messages = [dict(m) for m in rag.messages]
    chain_result = await graph.ainvoke({"messages": messages, "top_k": rag.top_k})

    docs = chain_result.get("documents", [])
    retrieved_docs = [
        RetrievedDoc(
            text=doc.page_content,
            score=doc.metadata.get("score"),
            metadata=doc.metadata,
        )
        for doc in docs
    ]
    query = chain_result.get("query") or (messages[-1]["content"] if messages else "")
    return RetrievalResponse(query=query, documents=retrieved_docs)


_STREAM_METADATA_OMIT_KEYS = {"page", "source"}


def _strip_stream_metadata(metadata: dict) -> dict:
    """Drop internal-only keys from streamed metadata."""
    return {k: v for k, v in (metadata or {}).items() if k not in _STREAM_METADATA_OMIT_KEYS}


@router.post("/stream", tags=["rag"])
async def question_answer_stream(rag: RAGRequest) -> StreamingResponse:
    """Stream the document retrieval."""
    messages = [dict(m) for m in rag.messages]

    async def stream() -> AsyncGenerator[str, None]:
        yield json.dumps({"type": "status", "data": "Starting request..."}) + "\n"

        async for chunk in graph.astream_events(
            {"messages": messages, "top_k": rag.top_k},
            version="v2",
        ):
            if chunk["event"] == "on_chain_start" and chunk["name"] == "retriever":
                yield json.dumps({"type": "status", "data": "Searching documents..."}) + "\n"

            elif chunk["event"] == "on_chain_end" and chunk["name"] == "retriever":
                output_docs = chunk["data"]["output"].get("documents", [])
                yield json.dumps({"type": "status", "data": f"Found {len(output_docs)} documents."}) + "\n"

            elif chunk["name"] == "LangGraph" and chunk["event"] == "on_chain_end" and "documents" in chunk.get("data", {}).get("output", {}):
                documents = [
                    {
                        "text": doc.page_content,
                        "metadata": _strip_stream_metadata(doc.metadata),
                    }
                    for doc in chunk["data"]["output"]["documents"]
                ]
                yield json.dumps({"type": "documents", "data": documents}) + "\n"

        yield json.dumps({"type": "status", "data": "Done."}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
