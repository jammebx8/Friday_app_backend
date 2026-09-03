"""
Async Groq client wrapper.

Provides a thin, typed interface over the official `groq` Python SDK so the
rest of the codebase never imports the SDK directly. Both text generation and
vision (multimodal) calls go through this module.
"""
from __future__ import annotations

import logging
from typing import Any

from groq import AsyncGroq

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton initialised lazily on first use
_groq_client: AsyncGroq | None = None


def get_groq_client() -> AsyncGroq:
    """Return (or create) the shared async Groq client."""
    global _groq_client
    if _groq_client is None:
        settings = get_settings()
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
        logger.info("Groq async client initialised")
    return _groq_client


async def chat_completion(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    response_format: dict[str, str] | None = None,
) -> str:
    """
    Send a chat-completion request and return the assistant content string.

    Args:
        messages:        OpenAI-format message list.
        model:           Override the default generation model.
        temperature:     Sampling temperature.
        max_tokens:      Maximum tokens in the response.
        response_format: Optional ``{"type": "json_object"}`` to enforce JSON.

    Returns:
        The assistant message content as a plain string.
    """
    settings = get_settings()
    client = get_groq_client()

    kwargs: dict[str, Any] = {
        "model": model or settings.openai_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    logger.debug(
        "Groq chat_completion model=%s tokens_used=%s",
        kwargs["model"],
        response.usage.total_tokens if response.usage else "?",
    )
    return content


async def vision_completion(
    image_contents: list[dict[str, Any]],
    system_prompt: str,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """
    Send a vision (multimodal) request with one or more images.

    Args:
        image_contents:  List of Groq multimodal content items.  Each item is
                         either ``{"type": "text", "text": "..."}`` or
                         ``{"type": "image_url", "image_url": {"url": "..."}}``.
        system_prompt:   System instruction for the vision model.
        model:           Override the default vision model.
        temperature:     Sampling temperature.
        max_tokens:      Maximum tokens in the response.

    Returns:
        The assistant message content as a plain string.
    """
    settings = get_settings()
    client = get_groq_client()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_contents},
    ]

    response = await client.chat.completions.create(
        model=model or settings.vision_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    logger.debug(
        "Groq vision_completion model=%s tokens_used=%s",
        model or settings.vision_model,
        response.usage.total_tokens if response.usage else "?",
    )
    return content
