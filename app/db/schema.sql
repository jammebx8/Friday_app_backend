-- ─────────────────────────────────────────────────────────────────────────────
-- Friday RAG — Supabase / PostgreSQL schema migration
-- Run this once against your Supabase project via the SQL editor or psql.
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ─── books ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS books (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    title       TEXT        NOT NULL,
    subject     TEXT        NOT NULL DEFAULT '',
    filename    TEXT        NOT NULL DEFAULT '',
    total_pages INTEGER     NOT NULL DEFAULT 0,
    status      TEXT        NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    progress    INTEGER     NOT NULL DEFAULT 0      -- 0–100 percentage
                            CHECK (progress >= 0 AND progress <= 100),
    pages_done  INTEGER     NOT NULL DEFAULT 0,
    error_msg   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Automatically update updated_at on every row change
CREATE OR REPLACE FUNCTION books_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_books_updated_at ON books;
CREATE TRIGGER trg_books_updated_at
    BEFORE UPDATE ON books
    FOR EACH ROW EXECUTE FUNCTION books_set_updated_at();

-- ─── book_chunks ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS book_chunks (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    book_id     UUID        NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter     TEXT        NOT NULL DEFAULT '',
    page        INTEGER     NOT NULL DEFAULT 0,
    content     TEXT        NOT NULL,
    embedding   VECTOR(384),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─── HNSW cosine index on embeddings ─────────────────────────────────────────
-- HNSW gives ~10-100× faster ANN queries vs ivfflat for small-to-medium tables.
-- m=16, ef_construction=64 are solid defaults; tune for your dataset size.
CREATE INDEX IF NOT EXISTS book_chunks_embedding_hnsw
    ON book_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ─── Convenience index for filtering chunks by book ──────────────────────────
CREATE INDEX IF NOT EXISTS book_chunks_book_id_idx
    ON book_chunks (book_id);
