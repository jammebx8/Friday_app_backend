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

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from app.api.ingest import run_ingestion
from app.db.supabase import db_insert, db_select
from app.models.book import BookStatusResponse, BookUploadResponse, ProcessBookRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/books", tags=["books"])

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
    request: Request,
) -> BookUploadResponse:
    """
    Accepts a small JSON payload (storage path + title), creates the book
    record, and kicks off background ingestion.
    """
    logger.info(
        "POST /books/process  origin=%s  storage_path=%s  title=%s",
        request.headers.get("origin", "-"),
        body.storage_path,
        body.title,
    )

    storage_path = body.storage_path.strip()
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="storage_path must not be empty.",
        )

    filename = body.filename.strip() or storage_path.split("/")[-1]
    title = body.title.strip() or filename

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
        logger.exception("DB insert failed for storage_path=%s", storage_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc}",
        ) from exc

    book_id: UUID = row["id"]
    logger.info("Book record created  id=%s  title=%s  path=%s", book_id, title, storage_path)

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
async def get_book(book_id: UUID, request: Request) -> BookStatusResponse:
    """Return the book's metadata and current ingestion progress (0–100 %)."""
    logger.info(
        "GET /books/%s  origin=%s",
        book_id,
        request.headers.get("origin", "-"),
    )
    rows = await db_select("books", filters={"id": str(book_id)}, limit=1)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book {book_id} not found.",
        )
    return BookStatusResponse(**rows[0])
