"""
Embedding service — BGE-small-en-v1.5 singleton.

The model is loaded exactly once during application startup and reused for
every embedding request. Never reloads per request.

Dimensions: 384
Backend:    HuggingFace sentence-transformers (local CPU/GPU inference)
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Dedicated thread-pool so CPU-bound embedding doesn't block the event loop
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")

# ── Singleton model ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model (called once at startup)."""
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embed_model)
    model = SentenceTransformer(settings.embed_model)
    logger.info(
        "Embedding model loaded — dimension=%d",
        model.get_sentence_embedding_dimension(),
    )
    return model


def warm_up_embeddings() -> None:
    """
    Force model loading synchronously.

    Call this from the FastAPI lifespan startup so the first real request
    does not pay the cold-start cost.
    """
    _load_model()


# ── Internal sync helpers (run in thread pool) ────────────────────────────────

def _embed_texts_sync(texts: list[str], normalize: bool = True) -> list[list[float]]:
    """Encode a list of texts synchronously and return float vectors."""
    model = _load_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=normalize,
        batch_size=32,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def _embed_single_sync(text: str, normalize: bool = True) -> list[float]:
    """Encode a single text synchronously."""
    return _embed_texts_sync([text], normalize=normalize)[0]


# ── Public async API ──────────────────────────────────────────────────────────

async def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts asynchronously.

    Offloads CPU-bound work to a thread pool so the event loop stays free.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of 384-dimensional float vectors, one per input text.
    """
    if not texts:
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _embed_texts_sync, texts)


async def embed_query(text: str) -> list[float]:
    """
    Embed a single query string asynchronously.

    Args:
        text: Query string.

    Returns:
        384-dimensional float vector.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _embed_single_sync, text)
