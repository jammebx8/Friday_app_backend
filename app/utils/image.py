"""
Image utility helpers — in-memory version.

Builds Groq multimodal content items directly from PNG bytes without
touching the filesystem.
"""
from __future__ import annotations

import base64


def png_bytes_to_base64(png_bytes: bytes) -> str:
    """Base64-encode raw PNG bytes."""
    return base64.b64encode(png_bytes).decode("utf-8")


def build_image_content_item_from_bytes(png_bytes: bytes) -> dict:
    """
    Build a Groq multimodal content item from in-memory PNG bytes.

    Returns:
        ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``
    """
    b64 = png_bytes_to_base64(png_bytes)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


def build_image_batch_content_from_bytes(
    page_images: list[bytes],
    leading_text: str = "",
) -> list[dict]:
    """
    Build the complete ``content`` list for a vision request from PNG bytes.

    Args:
        page_images:  List of raw PNG byte-strings.
        leading_text: Optional instruction text prepended before the images.

    Returns:
        List of content items ready for a Groq vision message.
    """
    content: list[dict] = []
    if leading_text:
        content.append({"type": "text", "text": leading_text})
    for png_bytes in page_images:
        content.append(build_image_content_item_from_bytes(png_bytes))
    return content
