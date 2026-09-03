"""
Background ingestion pipeline — fully in-memory, no disk writes.

Vercel serverless /tmp is only ~512 MB.  A 50 MB PDF rendered at 200 DPI
would produce several GB of PNG files — instantly hitting the limit.
This pipeline keeps all image data as Python bytes objects in RAM and
passes them directly to the OCR service without touching the filesystem.

Stages:
  1. Download PDF from Supabase Storage     (0–5 %)
  2. Render PDF pages to PNG bytes          (5–10 %)
  3. OCR via Groq Vision (5 pages/batch)    (10–60 %)
  4. Semantic chunking                      (60–75 %)
  5. BGE embeddings via embed worker        (75–90 %)
  6. Insert into pgvector                   (90–100 %)
"""
from __future__ import annotations

import logging
from uuid import UUID

from app.db.supabase import db_update, storage_download
from app.models.chunk import ChunkInsertPayload
from app.services.chunking import chunk_pages
from app.services.embedding_client import embed_documents
from app.services.ocr import ocr_pages
from app.services.pdf import pdf_to_image_bytes
from app.services.vector_store import insert_chunks

logger = logging.getLogger(__name__)


# ── Progress helpers ───────────────────────────────────────────────────────────

async def _set_progress(book_id: str, progress: int, pages_done: int = 0, book_status: str = "processing") -> None:
    await db_update(
        "books",
        {"progress": progress, "pages_done": pages_done, "status": book_status},
        {"id": book_id},
    )


async def _set_failed(book_id: str, error_msg: str) -> None:
    await db_update(
        "books",
        {"status": "failed", "error_msg": error_msg[:1000]},
        {"id": book_id},
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_ingestion(
    book_id: UUID,
    storage_path: str,
    storage_bucket: str,
    title: str,
    subject: str,
) -> None:
    """
    Full in-memory OCR + RAG ingestion pipeline.

    Downloads the PDF from Supabase Storage, renders pages to PNG bytes in
    RAM, runs OCR, chunks, embeds, and inserts into pgvector — no disk I/O.

    Args:
        book_id:        UUID of the already-created books row.
        storage_path:   Path inside the Supabase Storage bucket.
        storage_bucket: Bucket name (e.g. "books").
        title:          Book title used as chapter fallback.
        subject:        Subject / topic label.
    """
    bid = str(book_id)

    try:
        # ── Stage 1: Download PDF ──────────────────────────────────────────
        logger.info("[ingest %s] Stage 1: downloading %s", bid, storage_path)
        await _set_progress(bid, 2)

        try:
            pdf_bytes = await storage_download(storage_bucket, storage_path)
        except RuntimeError as exc:
            await _set_failed(bid, f"Storage download failed: {exc}")
            return

        logger.info("[ingest %s] Downloaded %d bytes", bid, len(pdf_bytes))
        await _set_progress(bid, 5)

        # ── Stage 2: Render PDF → PNG bytes (in memory) ────────────────────
        logger.info("[ingest %s] Stage 2: rendering PDF pages to memory", bid)

        try:
            page_images: list[bytes] = pdf_to_image_bytes(pdf_bytes)
        except (ValueError, RuntimeError) as exc:
            await _set_failed(bid, f"PDF rendering failed: {exc}")
            return
        finally:
            # Free the raw PDF bytes as soon as rendering is done
            del pdf_bytes

        total_pages = len(page_images)
        logger.info("[ingest %s] %d pages rendered in memory", bid, total_pages)

        await db_update(
            "books",
            {"total_pages": total_pages, "progress": 10, "status": "processing"},
            {"id": bid},
        )

        # ── Stage 3: OCR ───────────────────────────────────────────────────
        logger.info("[ingest %s] Stage 3: OCR (%d pages, ~%.0fs estimated)",
                    bid, total_pages, total_pages * 3.5)

        # Progress callback: update DB after every OCR'd page so the frontend
        # shows live progress during the slow OCR stage (10% → 60%).
        async def ocr_progress(pages_done: int, total: int) -> None:
            pct = 10 + int((pages_done / total) * 50)   # maps 0..total → 10..60
            await _set_progress(bid, pct, pages_done=pages_done)

        ocr_results = await ocr_pages(page_images, progress_callback=ocr_progress)

        # Free all page images from memory now that OCR is done
        del page_images

        non_empty = [(r.page, r.markdown) for r in ocr_results if r.markdown.strip()]
        logger.info("[ingest %s] OCR done — %d/%d pages with content", bid, len(non_empty), total_pages)
        await _set_progress(bid, 60, pages_done=len(non_empty))

        if not non_empty:
            await _set_failed(bid, "OCR produced no text. PDF may be image-only or corrupted.")
            return

        # ── Stage 4: Semantic chunking ─────────────────────────────────────
        logger.info("[ingest %s] Stage 4: chunking", bid)
        await _set_progress(bid, 62, pages_done=len(non_empty))

        text_chunks = chunk_pages(page_markdowns=non_empty, chapter_name=title)
        logger.info("[ingest %s] %d chunks produced", bid, len(text_chunks))
        await _set_progress(bid, 75, pages_done=len(non_empty))

        if not text_chunks:
            await _set_failed(bid, "Chunking produced zero chunks.")
            return

        # ── Stage 5: Embeddings ────────────────────────────────────────────
        logger.info("[ingest %s] Stage 5: embedding %d chunks", bid, len(text_chunks))

        vectors = await embed_documents([c.content for c in text_chunks])
        logger.info("[ingest %s] Embeddings done", bid)
        await _set_progress(bid, 90, pages_done=len(non_empty))

        # ── Stage 6: Insert into pgvector ──────────────────────────────────
        logger.info("[ingest %s] Stage 6: inserting into pgvector", bid)

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
        logger.info("[ingest %s] %d chunks inserted", bid, inserted)

        # ── Done ───────────────────────────────────────────────────────────
        await db_update(
            "books",
            {"status": "completed", "progress": 100,
             "pages_done": total_pages, "total_pages": total_pages},
            {"id": bid},
        )
        logger.info("[ingest %s] Ingestion complete ✓", bid)

    except Exception as exc:
        logger.exception("[ingest %s] Unexpected error", bid)
        await _set_failed(bid, str(exc))
