"""
Markdown post-processing utilities.

Cleans raw OCR output before it is passed to the semantic chunker so the
chunker sees consistent, well-formed Markdown rather than model artefacts.
"""
from __future__ import annotations

import re


# ── Public API ────────────────────────────────────────────────────────────────


def clean_ocr_markdown(raw: str) -> str:
    """
    Normalise raw Markdown output from the vision model.

    Operations performed (in order):
    1. Strip leading/trailing whitespace.
    2. Collapse runs of blank lines to at most two.
    3. Remove common page-number patterns (e.g. ``--- 42 ---``).
    4. Remove standalone header/footer lines (page N, chapter lines at
       start/end of content).
    5. Remove HTML comments ``<!-- ... -->``.
    6. Strip zero-width characters.
    7. Normalise Windows line endings.

    Args:
        raw: Raw Markdown string from the OCR model.

    Returns:
        Cleaned Markdown string.
    """
    text = raw

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip zero-width / non-printing characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove common page-number/header/footer artefacts
    # e.g. "--- 12 ---", "Page 12", "12 |", "| 12"
    text = re.sub(r"^---\s*\d+\s*---\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[Pp]age\s+\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ blank lines to exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_chapter_title(markdown: str) -> str:
    """
    Heuristically extract a chapter title from a Markdown document.

    Looks for the first H1 (``# Title``) or H2 (``## Title``) heading.
    Returns an empty string if none is found.

    Args:
        markdown: Cleaned Markdown text for a chapter.

    Returns:
        Chapter title string (stripped), or ``""`` if not found.
    """
    match = re.search(r"^#{1,2}\s+(.+)$", markdown, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def markdown_to_plain_text(markdown: str) -> str:
    """
    Strip Markdown syntax to produce plain text for token counting.

    Not a full parser — removes headings, bold/italic, links, code fences,
    and horizontal rules so token estimates are accurate.

    Args:
        markdown: Markdown-formatted string.

    Returns:
        Approximate plain-text equivalent.
    """
    text = markdown
    # Remove fenced code blocks (keep content)
    text = re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove links but keep display text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()
