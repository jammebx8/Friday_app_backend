"""
PDF → PNG bytes conversion — fully in-memory, no disk writes.

Vercel serverless functions have a tiny /tmp quota (~512 MB total) that a
large PDF would instantly exhaust when rendered to PNG files.  This module
renders every page to raw PNG bytes in RAM via PyMuPDF's tobytes() method
and returns them directly — nothing is ever written to the filesystem.
"""
from __future__ import annotations

import logging

import fitz  # PyMuPDF

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def pdf_to_image_bytes(
    pdf_bytes: bytes,
    dpi: int | None = None,
) -> list[bytes]:
    """
    Render every page of a PDF to PNG bytes in memory.

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        dpi:       Render resolution.  Defaults to ``settings.pdf_dpi``.
                   150 DPI is sufficient for text OCR and keeps memory low.

    Returns:
        List of PNG byte-strings, one per page, in page order.

    Raises:
        ValueError:  If the PDF cannot be opened or contains no pages.
        RuntimeError: If any page fails to render.
    """
    settings = get_settings()
    dpi = dpi or settings.pdf_dpi

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise ValueError("PDF contains no pages.")

    logger.info("Rendering %d pages at %d DPI (in-memory)", total_pages, dpi)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    page_images: list[bytes] = []

    for page_num in range(total_pages):
        page = doc[page_num]
        try:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            # tobytes() returns PNG bytes without writing to disk
            page_images.append(pixmap.tobytes("png"))
        except Exception as exc:
            doc.close()
            raise RuntimeError(f"Failed to render page {page_num + 1}: {exc}") from exc

    doc.close()
    logger.info("Rendered %d pages to memory (%d bytes total)",
                total_pages,
                sum(len(b) for b in page_images))
    return page_images
