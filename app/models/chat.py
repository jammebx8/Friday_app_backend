"""
Pydantic v2 models for Chat / RAG endpoints.
"""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.models.chunk import RetrievedChunk


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    book_id: UUID = Field(..., description="Book to query against")
    question: str = Field(..., min_length=1, max_length=4096, description="Student question")


class ChatResponse(BaseModel):
    """Response from POST /chat."""

    answer: str
    citations: list[int] = Field(default_factory=list, description="Page numbers cited")
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Top-K chunks used to build context (for debugging / UI display)",
    )
