"""
Semantic chunking service — Vercel-safe, zero heavy-ML dependencies.

Replaces the previous LangChain SemanticChunker (which required
sentence-transformers + torch) with LangChain's
RecursiveCharacterTextSplitter.  This is a pure-Python, rule-based splitter
that targets ~600-token chunks with 80-token overlap and respects Markdown
structure (headings, paragraphs, sentences) as natural break-points.

The quality trade-off is minimal for educational textbooks: semantic meaning
is already encoded in the Markdown structure produced by the OCR step, and the
embedding model still captures cross-chunk semantics at retrieval time.

Each returned chunk carries its source page number and chapter title.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.utils.markdown import extract_chapter_title

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single chunk with provenance metadata."""

    content: str
    chapter: str
    page: int  # First-page of the chapter whose content appears in this chunk


# ── Splitter configuration ────────────────────────────────────────────────────
# ~600 tokens × ~4 chars/token ≈ 2 400 chars; overlap 80 tokens ≈ 320 chars.
# Separators are ordered from coarsest to finest so the splitter tries to
# preserve Markdown structure before falling back to sentences/words.

_SEPARATORS = [
    "\n\n## ",   # H2 heading — strong section boundary
    "\n\n### ",  # H3 heading
    "\n\n",      # Paragraph boundary
    "\n",        # Line break
    ". ",        # Sentence boundary
    " ",         # Word boundary
    "",          # Character (last resort)
]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2400,
    chunk_overlap=320,
    separators=_SEPARATORS,
    length_function=len,
    is_separator_regex=False,
)


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_chapter(
    chapter_markdown: str,
    chapter_name: str = "",
    starting_page: int = 1,
) -> list[TextChunk]:
    """
    Split a chapter's Markdown text into overlapping chunks.

    Args:
        chapter_markdown: Full Markdown text of the chapter (all pages joined).
        chapter_name:     Human-readable chapter label.  Auto-detected from
                          the first H1/H2 heading when empty.
        starting_page:    Page number of the first page in this chapter.

    Returns:
        List of TextChunk objects in document order.
    """
    if not chapter_markdown.strip():
        return []

    if not chapter_name:
        chapter_name = extract_chapter_title(chapter_markdown) or f"Page {starting_page}"

    logger.info(
        "Chunking chapter '%s' (~%d chars) starting at page %d",
        chapter_name,
        len(chapter_markdown),
        starting_page,
    )

    raw_chunks: list[str] = _splitter.split_text(chapter_markdown)

    chunks: list[TextChunk] = [
        TextChunk(content=raw.strip(), chapter=chapter_name, page=starting_page)
        for raw in raw_chunks
        if raw.strip()
    ]

    logger.info("Chapter '%s' → %d chunks", chapter_name, len(chunks))
    return chunks


def chunk_pages(
    page_markdowns: list[tuple[int, str]],
    chapter_name: str = "",
) -> list[TextChunk]:
    """
    Combine multiple pages into a chapter and chunk the result.

    Args:
        page_markdowns: List of ``(page_number, markdown_text)`` tuples.
        chapter_name:   Optional chapter label.

    Returns:
        List of TextChunk objects.
    """
    if not page_markdowns:
        return []

    sorted_pages = sorted(page_markdowns, key=lambda x: x[0])
    starting_page = sorted_pages[0][0]
    combined = "\n\n".join(md for _, md in sorted_pages if md.strip())

    return chunk_chapter(
        chapter_markdown=combined,
        chapter_name=chapter_name,
        starting_page=starting_page,
    )
