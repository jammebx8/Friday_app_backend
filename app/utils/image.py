"""
Image utility helpers.

Provides base-64 encoding of PNG files for embedding in Groq multimodal
message payloads, plus a helper to build the content-item dict the API expects.
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """
    Read an image file and return its base64-encoded content (no data-URI prefix).

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Plain base64 string (UTF-8 encoded).

    Raises:
        FileNotFoundError: If the image file does not exist.
        IOError: If the file cannot be read.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_image_content_item(image_path: str) -> dict:
    """
    Build a Groq multimodal content item for a local image file.

    The Groq vision API accepts base64 data URIs in the same format as
    OpenAI's vision API.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Dict with ``type`` and ``image_url`` keys ready for insertion into
        the ``content`` list of a chat message.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"

    b64 = image_to_base64(image_path)
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{b64}",
        },
    }


def build_image_batch_content(
    image_paths: list[str],
    leading_text: str = "",
) -> list[dict]:
    """
    Build the complete ``content`` list for a vision request covering multiple images.

    Args:
        image_paths:  List of paths to include in this batch.
        leading_text: Optional text instruction prepended before the images.

    Returns:
        List of content items (text + images) ready for a Groq vision message.
    """
    content: list[dict] = []
    if leading_text:
        content.append({"type": "text", "text": leading_text})
    for path in image_paths:
        content.append(build_image_content_item(path))
    return content
