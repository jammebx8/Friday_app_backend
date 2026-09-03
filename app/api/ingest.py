"""
Background ingestion pipeline.

Runs entirely in a FastAPI BackgroundTask after the upload endpoint returns.
Stages (with progress tracking):
  1. PDF → images          (0–10 %)
  2. OCR (vision model)    (10–60 %)
  3. Semantic chunking     (60–75 %)
  4. BGE embedding         (75–90 %)
  5. pgvector insert       (90–100 %)

Progress + status are written back to the `books` table after every stage so
the frontend can poll GET /books/{id} and show a live progress ring.
"""
from __future__ import annotations

import logging
from uuid import UUID

from app.db.supabase import db_update
from app.models.chunk import ChunkInsertPayload
from app.services.chunking import chunk_pages
from app.services.embeddings import embed_documents
from app.services.ocr import ocr_pages
from app.services.pdf import cleanup_images, pdf_to_images
from app.services.vector_store import insert_chunks

logger = logging.getLogger(__name__)


# ── Progress helper ────────────────────────────────────────────────────────────

async def _set_progress(book_id: str, progress: int, pages_done: int = 0, status: str = "processing") -> None:
    await db_update(
        "books",
        {"progress": progress, "pages_done": pages_done, "status": status},
        {"id": book_id},
    )


async def _set_failed(book_id: str, error_msg: str) -> None:
    await db_update(
        "books",
        {"status": "failed", "error_msg": error_msg[:1000]},
        {"id": book_id},
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_ingestion(book_id: UUID, pdf_bytes: bytes, title: str, subject: str) -> None:
    """
    Full OCR + RAG ingestion pipeline for a single book.

    Designed to run as a FastAPI BackgroundTask — never raises; all exceptions
    are caught and written to the `books.error_msg` column.

    Args:
        book_id:   UUID of the already-created books row.
        pdf_bytes: Raw PDF file content.
        title:     Book title (used as chapter fallback name).
        subject:   Subject / topic label.
    """
    bid = str(book_id)
    image_paths: list[str] = []

    try:
        # ── Stage 1: PDF → images ──────────────────────────────────────────
        logger.info("[ingest %s] Stage 1: rendering PDF to images", bid)
        await _set_progress(bid, 2)

        try:
            image_paths = pdf_to_images(pdf_bytes)
        except (ValueError, RuntimeError) as exc:
            await _set_failed(bid, f"PDF rendering failed: {exc}")
            return

        total_pages = len(image_paths)
        logger.info("[ingest %s] %d pages rendered", bid, total_pages)

        await db_update(
            "books",
            {"total_pages": total_pages, "progress": 10, "status": "processing"},
            {"id": bid},
        )

        # ── Stage 2: OCR ───────────────────────────────────────────────────
        logger.info("[ingest %s] Stage 2: OCR (%d pages)", bid, total_pages)

        ocr_results = await ocr_pages(image_paths)

        # Filter out completely empty pages
        non_empty = [(r.page, r.markdown) for r in ocr_results if r.markdown.strip()]
        logger.info(
            "[ingest %s] OCR complete — %d/%d pages with content",
            bid, len(non_empty), total_pages,
        )

        await _set_progress(bid, 60, pages_done=len(non_empty))

        if not non_empty:
            await _set_failed(bid, "OCR produced no text from any page. PDF may be image-only without readable content.")
            cleanup_images(image_paths)
            return

        # ── Stage 3: Semantic chunking ─────────────────────────────────────
        logger.info("[ingest %s] Stage 3: semantic chunking", bid)
        await _set_progress(bid, 62, pages_done=len(non_empty))

        text_chunks = chunk_pages(
            page_markdowns=non_empty,
            chapter_name=title,
        )

        logger.info("[ingest %s] %d semantic chunks produced", bid, len(text_chunks))
        await _set_progress(bid, 75, pages_done=len(non_empty))

        if not text_chunks:
            await _set_failed(bid, "Chunking produced zero chunks.")
            cleanup_images(image_paths)
            return

        # ── Stage 4: BGE embeddings ────────────────────────────────────────
        logger.info("[ingest %s] Stage 4: embedding %d chunks", bid, len(text_chunks))

        texts = [c.content for c in text_chunks]
        vectors = await embed_documents(texts)

        logger.info("[ingest %s] Embeddings computed", bid)
        await _set_progress(bid, 90, pages_done=len(non_empty))

        # ── Stage 5: Insert into pgvector ──────────────────────────────────
        logger.info("[ingest %s] Stage 5: inserting into pgvector", bid)

        payloads = [
            ChunkInsertPayload(
                book_id=bid,
                chapter=chunk.chapter,
                page=chunk.page,
                content=chunk.content,
                embedding=vectors[i],
            )
            for i, chunk in enumerate(text_chunks)
        ]

        inserted = await insert_chunks(payloads)
        logger.info("[ingest %s] %d chunks inserted into pgvector", bid, inserted)

        # ── Done ───────────────────────────────────────────────────────────
        await db_update(
            "books",
            {
                "status": "completed",
                "progress": 100,
                "pages_done": total_pages,
                "total_pages": total_pages,
            },
            {"id": bid},
        )
        logger.info("[ingest %s] Ingestion complete ✓", bid)

    except Exception as exc:
        logger.exception("[ingest %s] Unexpected error during ingestion", bid)
        await _set_failed(bid, str(exc))
    finally:
        # Always clean up temp images
        if image_paths:
            cleanup_images(image_paths)
