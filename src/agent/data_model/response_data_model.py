"""Script that contains the Pydantic Models for the Rest API Response."""

from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    """Status."""

    SUCCESS = "success"
    FAILURE = "failure"


class SearchResponse(BaseModel):
    """The request parameters for explaining the output."""

    text: str = Field(..., title="Text", description="The text of the document.")
    page: int | None = Field(default=None, title="Page", description="Page number if available.")
    source: str | None = Field(default=None, title="Source", description="Source document or link.")


class EmbeddingResponse(BaseModel):
    """The Response for the Embedding endpoint."""

    status: Status = Field(Status.SUCCESS, title="Status", description="The status of the request.")
    files: list[str] = Field([], title="Files", description="The list of files that were embedded.")


class RetrievedDoc(BaseModel):
    """Retrieved document chunk with metadata and score."""

    text: str = Field(..., title="Text", description="Content of the document chunk.")
    page: int | None = Field(default=None, title="Page", description="Page number if available.")
    source: str | None = Field(default=None, title="Source", description="Source document or link.")
    score: float | None = Field(default=None, title="Score", description="Relevance or rerank score.")
    metadata: dict = Field(default_factory=dict, title="Metadata", description="Chunk metadata payload.")


class RetrievalResponse(BaseModel):
    """The Response for the Retrieval endpoint (v6 retrieval-only)."""

    query: str = Field(..., title="Query", description="The search query.")
    documents: list[RetrievedDoc] = Field([], title="Documents", description="Retrieved relevant document chunks.")
