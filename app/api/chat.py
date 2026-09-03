"""
Chat router — context-aware RAG question answering.

POST /chat  — embed question → retrieve chunks → generate answer via GPT-OSS-120B
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.db.supabase import db_select
from app.models.chat import ChatRequest, ChatResponse
from app.services.rag import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question about an uploaded textbook",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Full RAG pipeline for a student question.

    1. Validates the book exists and ingestion is complete.
    2. Embeds the question with BGE.
    3. Retrieves the top-K semantically similar chunks from pgvector.
    4. Builds a prompt and calls GPT-OSS-120B via Groq.
    5. Returns the answer, page citations, and retrieved chunks.

    The ``retrieved_chunks`` field is included in the response so the frontend
    can optionally display source excerpts for transparency / debugging.
    """
    bid = str(request.book_id)

    # ── Guard: book must exist and be fully ingested ──────────────────────────
    rows = await db_select("books", filters={"id": bid}, limit=1)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book {request.book_id} not found.",
        )

    book = rows[0]
    if book["status"] == "processing" or book["status"] == "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Book ingestion is still in progress ({book['progress']}%). "
                "Please wait until status is 'completed' before asking questions."
            ),
        )
    if book["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Book ingestion failed: {book.get('error_msg', 'unknown error')}.",
        )

    # ── Run RAG ───────────────────────────────────────────────────────────────
    try:
        result = await answer_question(
            question=request.question,
            book_id=bid,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("RAG pipeline error for book_id=%s", bid)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {exc}",
        ) from exc

    return ChatResponse(
        answer=result.answer,
        citations=result.citations,
        retrieved_chunks=result.retrieved_chunks,
    )
