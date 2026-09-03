"""
Pydantic v2 models for document chunks.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """Mirrors the `book_chunks` table row (embedding omitted for API responses)."""

    id: UUID
    book_id: UUID
    chapter: str
    page: int
    content: str
    created_at: datetime


class ChunkInsertPayload(BaseModel):
    """What we build before upserting into Supabase."""

    book_id: str  # UUID as string for JSON serialisation
    chapter: str
    page: int
    content: str
    embedding: list[float] = Field(..., min_length=384, max_length=384)


class RetrievedChunk(BaseModel):
    """A chunk returned from semantic similarity search."""

    id: str
    book_id: str
    chapter: str
    page: int
    content: str
    score: float = Field(ge=0.0, le=1.0)
