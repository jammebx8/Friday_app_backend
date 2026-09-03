"""
OCR service — converts in-memory page PNG bytes to structured Markdown.

Uses the Groq Vision API (Qwen vision model) to process up to 5 pages per
request.  All image data stays in RAM — no files are written to disk.
Implements exponential back-off retry logic for transient API failures.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from app.core.config import get_settings
from app.services.groq import vision_completion
from app.utils.image import build_image_batch_content_from_bytes
from app.utils.markdown import clean_ocr_markdown

logger = logging.getLogger(__name__)

_OCR_SYSTEM_PROMPT = """
You are an expert OCR engine for scanned educational textbooks.

Your task is to convert the provided scanned page image(s) into clean, structured Markdown.

PRESERVE exactly:
- All headings (use appropriate # ## ### levels)
- Mathematical equations and derivations (use LaTeX delimiters: $...$ inline, $$...$$ display)
- Numbered lists and bullet points
- Tables (use Markdown table syntax)
- Chemical formulas, symbols, and special notation
- Code snippets (use fenced code blocks)
- Figure captions and labels

REMOVE completely:
- Page numbers (standalone numerals at top/bottom)
- Running headers and footers
- Watermarks
- Decorative dividers that carry no content

OUTPUT FORMAT:
Return a JSON array. Each element corresponds to one page image in the order given.
Each element must be an object with exactly two keys:
  "page": <integer page number starting from the first page in this batch>
  "markdown": "<clean markdown string for that page>"

If a page is blank or contains only noise, set "markdown" to an empty string.
Return ONLY valid JSON — no preamble, no explanation.
""".strip()


@dataclass
class PageOCRResult:
    """OCR result for a single page."""
    page: int
    markdown: str


def _parse_ocr_response(raw: str, batch_start_page: int, batch_size: int) -> list[PageOCRResult]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = []
        else:
            logger.warning("Could not parse OCR JSON; treating batch as empty")
            return [PageOCRResult(page=batch_start_page + i, markdown="") for i in range(batch_size)]

    if not isinstance(data, list):
        data = [data]

    results: list[PageOCRResult] = []
    for i, item in enumerate(data):
        page_num = item.get("page", batch_start_page + i)
        md = clean_ocr_markdown(item.get("markdown", ""))
        results.append(PageOCRResult(page=page_num, markdown=md))
    return results


async def _ocr_batch_with_retry(
    page_images: list[bytes],       # PNG bytes, one per page
    batch_start_page: int,
    max_retries: int,
    base_delay: float,
) -> list[PageOCRResult]:
    settings = get_settings()
    user_instruction = (
        f"These are pages {batch_start_page} to "
        f"{batch_start_page + len(page_images) - 1} "
        "of a scanned educational textbook. Extract their text as instructed."
    )
    content_items = build_image_batch_content_from_bytes(page_images, leading_text=user_instruction)

    for attempt in range(1, max_retries + 1):
        try:
            raw = await vision_completion(
                image_contents=content_items,
                system_prompt=_OCR_SYSTEM_PROMPT,
                model=settings.vision_model,
                temperature=0.05,
                max_tokens=8192,
            )
            return _parse_ocr_response(raw, batch_start_page, len(page_images))
        except Exception as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "OCR batch pages %d-%d attempt %d/%d failed: %s — retry in %.1fs",
                    batch_start_page, batch_start_page + len(page_images) - 1,
                    attempt, max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "OCR batch pages %d-%d failed after %d attempts: %s",
                    batch_start_page, batch_start_page + len(page_images) - 1,
                    max_retries, exc,
                )

    return [PageOCRResult(page=batch_start_page + i, markdown="") for i in range(len(page_images))]


async def ocr_pages(
    page_images: list[bytes],         # PNG bytes list — no file paths
    batch_size: int | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
) -> list[PageOCRResult]:
    """
    OCR all pages from in-memory PNG bytes, batching up to *batch_size* per call.

    Args:
        page_images: List of raw PNG byte-strings, one per page (page 1 first).
        batch_size:  Pages per API request. Defaults to ``settings.ocr_batch_size``.
        max_retries: Retry attempts per batch.
        base_delay:  Exponential back-off base (seconds).

    Returns:
        Flat list of PageOCRResult ordered by page number.
    """
    settings = get_settings()
    batch_size  = batch_size  or settings.ocr_batch_size
    max_retries = max_retries or settings.ocr_max_retries
    base_delay  = base_delay  or settings.ocr_retry_base_delay

    if not page_images:
        return []

    total = len(page_images)
    all_results: list[PageOCRResult] = []
    n_batches = -(-total // batch_size)  # ceiling div

    for batch_idx, start in enumerate(range(0, total, batch_size)):
        batch = page_images[start : start + batch_size]
        batch_start_page = start + 1
        logger.info(
            "OCR batch %d/%d — pages %d-%d",
            batch_idx + 1, n_batches,
            batch_start_page, batch_start_page + len(batch) - 1,
        )
        results = await _ocr_batch_with_retry(batch, batch_start_page, max_retries, base_delay)
        all_results.extend(results)

        # Release batch bytes from memory immediately
        del batch

    all_results.sort(key=lambda r: r.page)
    return all_results
