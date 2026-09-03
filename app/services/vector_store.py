"""
Vector store service — async Supabase/pgvector operations.

Provides:
  - insert_chunks()      : Bulk-upsert embedded chunks into book_chunks
  - similarity_search()  : ANN cosine search via pgvector RPC function

The pgvector similarity search is exposed via a Supabase RPC function
``match_book_chunks`` which must exist in the database.  The SQL for that
function is included at the bottom of this module as a docstring so you can
run it once in Supabase.
"""
from __future__ import annotations

import logging
from typing import Any

from app.db.supabase import db_insert_many, db_rpc
from app.models.chunk import ChunkInsertPayload, RetrievedChunk

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

async def insert_chunks(chunks: list[ChunkInsertPayload]) -> int:
    """
    Bulk-insert embedded chunks into the ``book_chunks`` table.

    Rows are inserted in batches of 100 to stay within Supabase request size
    limits and give predictable memory usage.

    Args:
        chunks: List of ChunkInsertPayload objects (each has a 384-d embedding).

    Returns:
        Total number of rows inserted.
    """
    if not chunks:
        return 0

    batch_size = 100
    total_inserted = 0

    rows = [
        {
            "book_id": c.book_id,
            "chapter": c.chapter,
            "page": c.page,
            "content": c.content,
            "embedding": c.embedding,  # pgvector accepts Python list[float]
        }
        for c in chunks
    ]

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        await db_insert_many("book_chunks", batch)
        total_inserted += len(batch)
        logger.debug("Inserted chunk batch %d/%d (%d rows)", i // batch_size + 1, -(-len(rows) // batch_size), len(batch))

    logger.info("Total chunks inserted: %d", total_inserted)
    return total_inserted


async def similarity_search(
    query_embedding: list[float],
    book_id: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Retrieve the *top_k* most semantically similar chunks for a query.

    Uses the ``match_book_chunks`` Supabase RPC function (see SQL below).
    Results are ordered by cosine similarity descending (closest first).

    Args:
        query_embedding: 384-dimensional query vector.
        book_id:         UUID string of the book to search within.
        top_k:           Number of chunks to return.

    Returns:
        List of RetrievedChunk ordered by descending similarity score.
    """
    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "target_book_id": book_id,
        "match_count": top_k,
    }

    rows: list[dict[str, Any]] = await db_rpc("match_book_chunks", params)

    results: list[RetrievedChunk] = []
    for row in rows or []:
        results.append(
            RetrievedChunk(
                id=str(row.get("id", "")),
                book_id=str(row.get("book_id", "")),
                chapter=row.get("chapter", ""),
                page=int(row.get("page", 0)),
                content=row.get("content", ""),
                score=float(row.get("similarity", 0.0)),
            )
        )

    logger.debug(
        "similarity_search book_id=%s top_k=%d → %d results",
        book_id,
        top_k,
        len(results),
    )
    return results


# ── SQL for the RPC function (run once in Supabase SQL editor) ────────────────
"""
-- Add this function to your Supabase project (SQL editor → run once):

CREATE OR REPLACE FUNCTION match_book_chunks(
    query_embedding  VECTOR(384),
    target_book_id   UUID,
    match_count      INT DEFAULT 5
)
RETURNS TABLE (
    id         UUID,
    book_id    UUID,
    chapter    TEXT,
    page       INTEGER,
    content    TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        bc.id,
        bc.book_id,
        bc.chapter,
        bc.page,
        bc.content,
        1 - (bc.embedding <=> query_embedding) AS similarity
    FROM book_chunks bc
    WHERE bc.book_id = target_book_id
      AND bc.embedding IS NOT NULL
    ORDER BY bc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""
