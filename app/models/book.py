"""
Pydantic v2 models for Books.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ── DB row representation ─────────────────────────────────────────────────────

class BookRecord(BaseModel):
    """Mirrors the `books` table row."""

    id: UUID
    title: str
    subject: str
    filename: str
    total_pages: int
    status: Literal["pending", "processing", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    pages_done: int
    error_msg: str | None = None
    created_at: datetime
    updated_at: datetime


# ── API request / response schemas ────────────────────────────────────────────

class BookUploadResponse(BaseModel):
    """Returned immediately after a PDF is accepted for ingestion."""

    book_id: UUID
    status: Literal["pending", "processing"] = "pending"
    message: str = "Ingestion started in the background."


class BookStatusResponse(BaseModel):
    """Book metadata + current processing progress."""

    id: UUID
    title: str
    subject: str
    filename: str
    total_pages: int
    pages_done: int
    progress: int = Field(ge=0, le=100)
    status: Literal["pending", "processing", "completed", "failed"]
    error_msg: str | None = None
    created_at: datetime
    updated_at: datetime


class BookCreatePayload(BaseModel):
    """Internal payload used when inserting a new book row."""

    title: str
    subject: str
    filename: str
