"""
Books router.

POST /books/process  — tell the backend to ingest a PDF already in Supabase Storage
GET  /books/{id}     — return book metadata + ingestion progress

Architecture note
-----------------
The frontend uploads PDFs directly to Supabase Storage (bypassing Vercel's
4.5 MB body limit entirely).  Once the upload completes the frontend calls
POST /books/process with a small JSON body containing only the storage path.
The backend downloads the PDF from Storage server-side and runs the full
OCR → chunking → embedding → pgvector pipeline in a BackgroundTask.
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.api.ingest import run_ingestion
from app.db.supabase import db_insert, db_select
from app.models.book import BookStatusResponse, BookUploadResponse, ProcessBookRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/books", tags=["books"])

# Supabase Storage bucket that the frontend uploads PDFs into.
# Create this bucket in your Supabase dashboard (Storage → New bucket).
STORAGE_BUCKET = "books"


# ── POST /books/process ───────────────────────────────────────────────────────

@router.post(
    "/process",
    response_model=BookUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a PDF that has already been uploaded to Supabase Storage",
)
async def process_book(
    body: ProcessBookRequest,
    background_tasks: BackgroundTasks,
) -> BookUploadResponse:
    """
    Accepts a small JSON payload (storage path + title), creates the book
    record, and kicks off background ingestion.

    The frontend:
      1. Uploads the PDF directly to Supabase Storage (client-side).
      2. Calls this endpoint with the resulting storage path.

    This approach avoids sending the binary file through Vercel's edge network
    and stays well under the 4.5 MB request-body limit.
    """
    storage_path = body.storage_path.strip()
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="storage_path must not be empty.",
        )

    filename = body.filename.strip() or storage_path.split("/")[-1]
    title = body.title.strip() or filename

    # ── Create books row ──────────────────────────────────────────────────────
    try:
        row = await db_insert(
            "books",
            {
                "title": title,
                "subject": body.subject.strip(),
                "filename": filename,
                "status": "pending",
                "progress": 0,
                "pages_done": 0,
                "total_pages": 0,
            },
        )
    except Exception as exc:
        logger.exception("Failed to create book record for path=%s", storage_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc}",
        ) from exc

    book_id: UUID = row["id"]
    logger.info(
        "Book record created: id=%s title=%s storage_path=%s",
        book_id, title, storage_path,
    )

    # ── Enqueue background ingestion ──────────────────────────────────────────
    background_tasks.add_task(
        run_ingestion,
        book_id=book_id,
        storage_path=storage_path,
        storage_bucket=STORAGE_BUCKET,
        title=title,
        subject=body.subject.strip(),
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
    return BookStatusResponse(**rows[0])
