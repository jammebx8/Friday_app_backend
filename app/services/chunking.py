"""
Semantic chunking service using LangChain's SemanticChunker.

Takes assembled chapter Markdown (NOT individual page text) and splits it
into semantically coherent chunks of ~300–800 tokens with 80-token overlap.
Each returned chunk carries its source page number and chapter title.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.utils.markdown import extract_chapter_title

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single semantic chunk with provenance metadata."""

    content: str
    chapter: str
    page: int  # Page number of the *first* page whose content appears in this chunk


# ── Singleton chunker ─────────────────────────────────────────────────────────

_chunker: SemanticChunker | None = None


def get_chunker() -> SemanticChunker:
    """Return (or lazily initialise) the global SemanticChunker."""
    global _chunker
    if _chunker is None:
        settings = get_settings()
        logger.info("Initialising SemanticChunker with model: %s", settings.embed_model)
        # LangChain's HuggingFaceEmbeddings wraps sentence-transformers;
        # the same BAAI/bge-small-en-v1.5 model is re-used here.
        lc_embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _chunker = SemanticChunker(
            embeddings=lc_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
        )
        logger.info("SemanticChunker ready")
    return _chunker


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_chapter(
    chapter_markdown: str,
    chapter_name: str = "",
    starting_page: int = 1,
) -> list[TextChunk]:
    """
    Semantically chunk an entire chapter's Markdown text.

    The chapter text is split as a whole (not page-by-page) so the chunker
    can find natural semantic break-points across page boundaries.

    Args:
        chapter_markdown: Full Markdown text of the chapter, already assembled
                          from OCR output (pages joined with ``\\n\\n``).
        chapter_name:     Human-readable chapter title.  If empty, the function
                          tries to extract an H1/H2 heading from the text.
        starting_page:    Page number of the first page in this chapter
                          (used as the ``page`` attribute on every chunk).

    Returns:
        List of TextChunk objects ordered as they appear in the document.
    """
    if not chapter_markdown.strip():
        return []

    if not chapter_name:
        chapter_name = extract_chapter_title(chapter_markdown) or f"Page {starting_page}"

    chunker = get_chunker()

    logger.info(
        "Chunking chapter '%s' (~%d chars) starting at page %d",
        chapter_name,
        len(chapter_markdown),
        starting_page,
    )

    raw_chunks: list[str] = chunker.split_text(chapter_markdown)

    chunks: list[TextChunk] = []
    for raw in raw_chunks:
        if raw.strip():
            chunks.append(
                TextChunk(
                    content=raw.strip(),
                    chapter=chapter_name,
                    page=starting_page,
                )
            )

    logger.info("Chapter '%s' → %d chunks", chapter_name, len(chunks))
    return chunks


def chunk_pages(
    page_markdowns: list[tuple[int, str]],
    chapter_name: str = "",
) -> list[TextChunk]:
    """
    Combine multiple pages into a chapter and semantically chunk the result.

    Args:
        page_markdowns: List of ``(page_number, markdown_text)`` tuples.
        chapter_name:   Optional chapter label.

    Returns:
        List of TextChunk objects.
    """
    if not page_markdowns:
        return []

    # Sort by page number to ensure correct ordering
    sorted_pages = sorted(page_markdowns, key=lambda x: x[0])
    starting_page = sorted_pages[0][0]

    # Join all pages into one large document for the semantic chunker
    combined = "\n\n".join(md for _, md in sorted_pages if md.strip())

    return chunk_chapter(
        chapter_markdown=combined,
        chapter_name=chapter_name,
        starting_page=starting_page,
    )
