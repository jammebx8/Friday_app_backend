"""
OCR service — rate-limit-aware Groq Vision OCR.

Model: meta-llama/llama-4-scout-17b-16e-instruct
Free-tier limits:
  • 30 requests / minute
  • 8 000 tokens / minute
  • 1 000 requests / day

Strategy
--------
* Send ONE page per request (safest under 8K tokens/min).
* Enforce a minimum gap of `ocr_min_gap_seconds` (default 3 s) between
  requests → max ~20 req/min, well under the 30 req/min wall.
* On HTTP 429 (rate limited): parse the `retry-after` header if present,
  otherwise back off with exponential delay (capped at 60 s).
* On any other transient error: standard exponential back-off.
* Empty / unparseable responses are logged and skipped (not fatal).

All image data stays in RAM — no disk writes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Awaitable

from groq import RateLimitError

from app.core.config import get_settings
from app.services.groq import vision_completion
from app.utils.image import build_image_batch_content_from_bytes
from app.utils.markdown import clean_ocr_markdown

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_OCR_SYSTEM_PROMPT = """
You are an expert OCR engine for scanned educational textbooks.

Convert the provided page image into clean, structured Markdown.

PRESERVE:
- Headings (# ## ###)
- Math equations (LaTeX: $...$ inline, $$...$$ display)
- Numbered lists and bullet points
- Tables (Markdown syntax)
- Code snippets (fenced blocks)
- Figure captions

REMOVE:
- Page numbers
- Running headers / footers
- Watermarks

Return ONLY the Markdown text — no JSON, no explanation.
If the page is blank or unreadable, return an empty string.
""".strip()


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PageOCRResult:
    """OCR result for a single page."""
    page: int       # 1-based
    markdown: str


# ── Rate-limit token bucket ───────────────────────────────────────────────────

class _RateLimiter:
    """
    Simple sliding-window rate limiter.

    Tracks the timestamps of recent requests and sleeps when the next
    request would exceed `max_per_minute`.  Also enforces a minimum gap
    between consecutive requests to spread load evenly.
    """

    def __init__(self, max_per_minute: int, min_gap_seconds: float) -> None:
        self._max_per_minute = max_per_minute
        self._min_gap = min_gap_seconds
        self._timestamps: list[float] = []
        self._last_call: float = 0.0

    async def acquire(self) -> None:
        """Wait until a new request is allowed, then record the timestamp."""
        now = time.monotonic()

        # Enforce minimum gap between consecutive calls
        gap_wait = self._min_gap - (now - self._last_call)
        if gap_wait > 0:
            logger.debug("Rate limiter: gap wait %.2fs", gap_wait)
            await asyncio.sleep(gap_wait)
            now = time.monotonic()

        # Enforce requests-per-minute window
        window_start = now - 60.0
        self._timestamps = [t for t in self._timestamps if t > window_start]
        if len(self._timestamps) >= self._max_per_minute:
            oldest = self._timestamps[0]
            window_wait = (oldest + 60.0) - now + 0.1  # +100 ms headroom
            if window_wait > 0:
                logger.info("Rate limiter: RPM window full, sleeping %.2fs", window_wait)
                await asyncio.sleep(window_wait)
                now = time.monotonic()
                # Prune again after sleep
                window_start = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > window_start]

        self._timestamps.append(now)
        self._last_call = now


# ── Single-page OCR with retry ────────────────────────────────────────────────

async def _ocr_single_page(
    png_bytes: bytes,
    page_num: int,        # 1-based, for logging
    rate_limiter: _RateLimiter,
    max_retries: int,
    base_delay: float,
) -> str:
    """
    OCR a single page image with rate-limit-aware retry.

    Returns the Markdown string (may be empty if the page is blank).
    Never raises — returns "" on permanent failure.
    """
    settings = get_settings()
    content_items = build_image_batch_content_from_bytes(
        [png_bytes],
        leading_text=f"This is page {page_num} of a scanned educational textbook.",
    )

    for attempt in range(1, max_retries + 1):
        await rate_limiter.acquire()
        try:
            raw = await vision_completion(
                image_contents=content_items,
                system_prompt=_OCR_SYSTEM_PROMPT,
                model=settings.vision_model,
                temperature=0.05,
                max_tokens=2048,   # one page fits comfortably in 2K tokens
            )
            md = clean_ocr_markdown(raw or "")
            logger.debug("Page %d OCR ok (%d chars)", page_num, len(md))
            return md

        except RateLimitError as exc:
            # Parse retry-after from the exception or default to escalating delay
            retry_after = _parse_retry_after(str(exc)) or min(base_delay * (2 ** attempt), 60.0)
            logger.warning(
                "Page %d: 429 rate limited (attempt %d/%d) — sleeping %.1fs",
                page_num, attempt, max_retries, retry_after,
            )
            await asyncio.sleep(retry_after)

        except Exception as exc:
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Page %d: OCR error attempt %d/%d: %s — retry in %.1fs",
                    page_num, attempt, max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Page %d: OCR failed after %d attempts: %s", page_num, max_retries, exc)

    return ""


def _parse_retry_after(error_message: str) -> float | None:
    """Extract a numeric retry-after value from a rate-limit error message."""
    match = re.search(r"retry.?after[:\s]+([0-9.]+)", error_message, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # Also look for "Please try again in Xs" pattern Groq uses
    match = re.search(r"try again in ([0-9.]+)s", error_message, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 0.5   # small headroom
        except ValueError:
            pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────

async def ocr_pages(
    page_images: list[bytes],
    batch_size: int | None = None,           # kept for API compat, ignored (always 1)
    max_retries: int | None = None,
    base_delay: float | None = None,
    progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
) -> list[PageOCRResult]:
    """
    OCR all pages sequentially, one page per API call, with rate limiting.

    Args:
        page_images:       PNG bytes, one per page, in order.
        batch_size:        Ignored — always sends 1 page per request to stay
                           within the 8K tokens/min limit.
        max_retries:       Retries per page (including 429 back-off).
        base_delay:        Base exponential back-off delay in seconds.
        progress_callback: Optional async callable(pages_done, total) called
                           after each page completes, for live progress updates.

    Returns:
        List of PageOCRResult sorted by page number.
        Pages that fail after all retries return an empty markdown string.
    """
    settings = get_settings()
    max_retries = max_retries or settings.ocr_max_retries
    base_delay  = base_delay  or settings.ocr_retry_base_delay

    if not page_images:
        return []

    total = len(page_images)
    rate_limiter = _RateLimiter(
        max_per_minute=settings.ocr_rpm_limit,
        min_gap_seconds=settings.ocr_min_gap_seconds,
    )

    logger.info(
        "Starting OCR: %d pages, model=%s, rpm_limit=%d, gap=%.1fs",
        total, settings.vision_model,
        settings.ocr_rpm_limit, settings.ocr_min_gap_seconds,
    )

    results: list[PageOCRResult] = []

    for idx, png_bytes in enumerate(page_images):
        page_num = idx + 1  # 1-based
        logger.info("OCR page %d/%d", page_num, total)

        md = await _ocr_single_page(
            png_bytes=png_bytes,
            page_num=page_num,
            rate_limiter=rate_limiter,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        results.append(PageOCRResult(page=page_num, markdown=md))

        if progress_callback:
            await progress_callback(page_num, total)

        # Free the image bytes immediately after OCR
        page_images[idx] = b""

    non_empty = sum(1 for r in results if r.markdown.strip())
    logger.info("OCR complete: %d/%d pages with content", non_empty, total)

    return results
