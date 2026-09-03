"""
Books router — PDF upload + status endpoints.

POST /books/upload  — accept a PDF, create a books row, kick off background ingestion
GET  /books/{id}    — return book metadata + progress
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from app.api.ingest import run_ingestion
from app.db.supabase import db_insert, db_select
from app.models.book import BookStatusResponse, BookUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/books", tags=["books"])

_MAX_PDF_BYTES = 100 * 1024 * 1024  # 100 MB hard limit


# ── POST /books/upload ────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=BookUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF textbook and start ingestion",
)
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to ingest"),
    title: str = Form(..., description="Human-readable book title"),
    subject: str = Form(default="", description="Subject / topic label"),
) -> BookUploadResponse:
    """
    Accept a PDF upload, create the book record, and enqueue background ingestion.

    Returns immediately with ``book_id`` and ``status: pending`` so the frontend
    can start polling progress while ingestion runs asynchronously.
    """
    # ── Validate file type ────────────────────────────────────────────────────
    content_type = file.content_type or ""
    filename = file.filename or "upload.pdf"

    if "pdf" not in content_type.lower() and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    # ── Read + size-check ─────────────────────────────────────────────────────
    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(pdf_bytes) > _MAX_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds maximum allowed size of {_MAX_PDF_BYTES // 1024 // 1024} MB.",
        )

    # ── Create books row ──────────────────────────────────────────────────────
    try:
        row = await db_insert(
            "books",
            {
                "title": title.strip() or filename,
                "subject": subject.strip(),
                "filename": filename,
                "status": "pending",
                "progress": 0,
                "pages_done": 0,
                "total_pages": 0,
            },
        )
    except Exception as exc:
        logger.exception("Failed to create book record")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc}",
        ) from exc

    book_id: UUID = row["id"]
    logger.info("Book record created: id=%s title=%s", book_id, title)

    # ── Enqueue background ingestion ──────────────────────────────────────────
    background_tasks.add_task(
        run_ingestion,
        book_id=book_id,
        pdf_bytes=pdf_bytes,
        title=title.strip(),
        subject=subject.strip(),
    )

    return BookUploadResponse(
        book_id=book_id,
        status="pending",
        message="PDF accepted. Ingestion started in the background.",
    )


# ── GET /books/{id} ───────────────────────────────────────────────────────────

@router.get(
    "/{book_id}",
    response_model=BookStatusResponse,
    summary="Get book metadata and ingestion progress",
)
async def get_book(book_id: UUID) -> BookStatusResponse:
    """
    Return the book's metadata and current ingestion progress (0–100 %).

    The frontend polls this endpoint every few seconds to update the circular
    progress indicator shown above the prompt bar.
    """
    rows = await db_select("books", filters={"id": str(book_id)}, limit=1)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book {book_id} not found.",
        )
    row = rows[0]
    return BookStatusResponse(**row)
