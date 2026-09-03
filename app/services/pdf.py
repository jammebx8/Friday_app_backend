"""
PDF → PNG image conversion service using PyMuPDF (fitz).

Single responsibility: given a raw PDF byte-string, render every page to a
PNG file on disk at the configured DPI and return the ordered list of paths.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import fitz  # PyMuPDF

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def pdf_to_images(
    pdf_bytes: bytes,
    output_dir: str | None = None,
    dpi: int | None = None,
) -> list[str]:
    """
    Convert every page of a PDF to a PNG image file.

    Args:
        pdf_bytes:  Raw bytes of the uploaded PDF file.
        output_dir: Directory where PNG files are written.
                    Defaults to ``settings.tmp_dir / <random-uuid>``.
        dpi:        Render resolution. Defaults to ``settings.pdf_dpi``.

    Returns:
        Sorted list of absolute PNG file paths, one per page.

    Raises:
        ValueError: If the PDF is corrupted or has no pages.
        RuntimeError: If rendering any page fails.
    """
    settings = get_settings()
    dpi = dpi or settings.pdf_dpi

    # Prepare output directory
    if output_dir is None:
        base = Path(settings.tmp_dir)
        output_dir = str(base / uuid.uuid4().hex)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise ValueError("PDF contains no pages.")

    logger.info("Rendering %d pages at %d DPI → %s", total_pages, dpi, output_dir)

    # PyMuPDF uses a transformation matrix; 72 dpi is the base.
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    image_paths: list[str] = []

    for page_num in range(total_pages):
        page = doc[page_num]
        try:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        except Exception as exc:
            doc.close()
            raise RuntimeError(
                f"Failed to render page {page_num + 1}: {exc}"
            ) from exc

        # Zero-padded filename for natural sort order
        filename = os.path.join(output_dir, f"page_{page_num + 1:04d}.png")
        pixmap.save(filename)
        image_paths.append(filename)

    doc.close()
    logger.info("Rendered %d pages successfully", total_pages)
    return image_paths


def cleanup_images(image_paths: list[str]) -> None:
    """
    Delete rendered PNG files and their parent directory (if empty).

    Safe to call even if files no longer exist.
    """
    parent_dirs: set[Path] = set()
    for path in image_paths:
        p = Path(path)
        parent_dirs.add(p.parent)
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    for d in parent_dirs:
        try:
            d.rmdir()  # Only removes if empty
        except Exception:
            pass
