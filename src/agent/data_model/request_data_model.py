"""Script that contains the Pydantic Models for the Rest Request."""

from pydantic import BaseModel, Field


class SearchParams(BaseModel):
    """The request parameters for searching the database."""

    query: str = Field(..., title="Query", description="The search query.")
    k: int = Field(3, title="Amount", description="The number of search results to return.")


class ChatMessages(BaseModel):
    """The Chat Messages Model."""

    role: str = Field(
        ...,
        title="Role",
        description="The role of the sender can be either user or assistant.",
    )
    content: str = Field(default=..., title="Content", description="The content of the message.")


class RAGRequest(BaseModel):
    """Request for the QA endpoint.

    Collection name is read from the app config (``QDRANT_COLLECTION_NAME``)
    and is no longer provided per-request.
    """

    messages: list[ChatMessages] | None = Field(
        default=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
        title="History",
        description="A list of previous questions and answers to include in the context.",
    )
    top_k: int | None = Field(
        default=None,
        title="Top K",
        ge=1,
        description="Number of documents to return after reranking. Falls back to the server-side ``RERANK_TOP_K`` config when omitted; clamped against the number of retrieved documents (``RETRIEVAL_K``).",
    )


class EmbeddTextRequest(BaseModel):
    """The request parameters for embedding text."""

    text: str = Field(..., title="Text", description="The text to embed.")
    file_name: str = Field(
        ...,
        title="File Name",
        description="The name of the file to save the embedded text to.",
    )
    separator: str = Field(
        "###",
        title="Separator",
        description="The separator to use between embedded texts.",
    )
