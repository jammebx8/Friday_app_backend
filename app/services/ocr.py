"""
OCR service — converts rendered page images to structured Markdown.

Uses the Groq Vision API (Qwen vision model) to process up to 5 pages per
request. Implements exponential back-off retry logic for transient failures.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from app.core.config import get_settings
from app.services.groq import vision_completion
from app.utils.image import build_image_batch_content
from app.utils.markdown import clean_ocr_markdown

logger = logging.getLogger(__name__)

# ── OCR system prompt ─────────────────────────────────────────────────────────

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


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class PageOCRResult:
    """OCR result for a single page."""

    page: int
    markdown: str


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_ocr_response(raw: str, batch_start_page: int, batch_size: int) -> list[PageOCRResult]:
    """
    Parse the JSON array returned by the vision model.

    Falls back gracefully if the model wraps the JSON in markdown fences or
    prefixes it with text.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to extract a JSON array from inside the string
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            logger.warning("Could not parse OCR JSON; treating entire batch as empty")
            return [
                PageOCRResult(page=batch_start_page + i, markdown="")
                for i in range(batch_size)
            ]

    if not isinstance(data, list):
        data = [data]

    results: list[PageOCRResult] = []
    for i, item in enumerate(data):
        page_num = item.get("page", batch_start_page + i)
        md = clean_ocr_markdown(item.get("markdown", ""))
        results.append(PageOCRResult(page=page_num, markdown=md))

    return results


async def _ocr_batch_with_retry(
    image_paths: list[str],
    batch_start_page: int,
    max_retries: int,
    base_delay: float,
) -> list[PageOCRResult]:
    """
    Run OCR on a batch of images with exponential back-off retries.

    Args:
        image_paths:       Paths to PNG files in this batch (max 5).
        batch_start_page:  1-based page number of the first image in the batch.
        max_retries:       Total attempts before giving up.
        base_delay:        Seconds to wait before first retry; doubles each time.

    Returns:
        List of PageOCRResult, one per image.
    """
    settings = get_settings()
    user_instruction = (
        f"These are pages {batch_start_page} to {batch_start_page + len(image_paths) - 1} "
        "of a scanned educational textbook. Extract their text as instructed."
    )
    content_items = build_image_batch_content(image_paths, leading_text=user_instruction)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = await vision_completion(
                image_contents=content_items,
                system_prompt=_OCR_SYSTEM_PROMPT,
                model=settings.vision_model,
                temperature=0.05,
                max_tokens=8192,
            )
            return _parse_ocr_response(raw, batch_start_page, len(image_paths))
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "OCR batch (pages %d-%d) attempt %d/%d failed: %s — retrying in %.1fs",
                    batch_start_page,
                    batch_start_page + len(image_paths) - 1,
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "OCR batch (pages %d-%d) failed after %d attempts: %s",
                    batch_start_page,
                    batch_start_page + len(image_paths) - 1,
                    max_retries,
                    exc,
                )

    # Return empty markdown for all pages in this batch rather than crashing
    return [
        PageOCRResult(page=batch_start_page + i, markdown="")
        for i in range(len(image_paths))
    ]


# ── Public API ────────────────────────────────────────────────────────────────

async def ocr_pages(
    image_paths: list[str],
    batch_size: int | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
) -> list[PageOCRResult]:
    """
    OCR all page images, batching up to *batch_size* images per API call.

    Processing is sequential across batches to respect rate limits; within a
    batch all images are sent in one vision request.

    Args:
        image_paths: Ordered list of PNG file paths (page 1 first).
        batch_size:  Images per request. Defaults to ``settings.ocr_batch_size`` (5).
        max_retries: Retry attempts per batch. Defaults to ``settings.ocr_max_retries``.
        base_delay:  Back-off base delay. Defaults to ``settings.ocr_retry_base_delay``.

    Returns:
        Flat list of PageOCRResult ordered by page number.
    """
    settings = get_settings()
    batch_size = batch_size or settings.ocr_batch_size
    max_retries = max_retries or settings.ocr_max_retries
    base_delay = base_delay or settings.ocr_retry_base_delay

    if not image_paths:
        return []

    all_results: list[PageOCRResult] = []
    total = len(image_paths)

    for batch_idx, start in enumerate(range(0, total, batch_size)):
        batch = image_paths[start : start + batch_size]
        batch_start_page = start + 1  # 1-based
        logger.info(
            "OCR batch %d/%d — pages %d-%d",
            batch_idx + 1,
            -(-total // batch_size),  # ceiling division
            batch_start_page,
            batch_start_page + len(batch) - 1,
        )
        results = await _ocr_batch_with_retry(batch, batch_start_page, max_retries, base_delay)
        all_results.extend(results)

    # Sort by page number (should already be sorted, but be safe)
    all_results.sort(key=lambda r: r.page)
    return all_results
