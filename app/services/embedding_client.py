"""
HTTP client for the external embedding service.

The Vercel deployment contains no PyTorch or sentence-transformers.
All embedding work is delegated to a separate service (e.g. a Modal function,
a Railway worker, or any HTTP server that runs BAAI/bge-small-en-v1.5).

Expected service contract
──────────────────────────
POST /embed
  Body:  { "texts": ["text1", "text2", ...] }
  Reply: { "embeddings": [[0.1, ...], [0.2, ...], ...] }
         Each inner list is a 384-dimensional float vector.

The service URL and an optional bearer-token API key are read from env vars:
  EMBEDDING_SERVICE_URL     — e.g. https://my-embed-worker.modal.run
  EMBEDDING_SERVICE_API_KEY — passed as "Authorization: Bearer <key>"
                               (leave blank if the service uses no auth)
"""
from __future__ import annotations

import logging

import httpx
from fastapi import HTTPException, status
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Module-level httpx client (reused across requests) ────────────────────────
# Created lazily on first call so that config is fully loaded before access.
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Return (or lazily create) a shared async HTTP client."""
    global _http_client
    if _http_client is None:
        settings = get_settings()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.embedding_service_api_key:
            headers["Authorization"] = f"Bearer {settings.embedding_service_api_key}"
        _http_client = httpx.AsyncClient(
            base_url=settings.embedding_service_url,
            headers=headers,
            timeout=httpx.Timeout(60.0),  # embedding batches can be slow
        )
        logger.info(
            "Embedding HTTP client initialised → %s",
            settings.embedding_service_url,
        )
    return _http_client


async def close_embedding_client() -> None:
    """Close the shared HTTP client (call from FastAPI lifespan shutdown)."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        logger.info("Embedding HTTP client closed")


# ── Retry decorator ───────────────────────────────────────────────────────────

def _make_retry():
    """Build a tenacity retry decorator: 3 attempts, exponential back-off."""
    return retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1.0, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )


# ── Core request helper ───────────────────────────────────────────────────────

@_make_retry()
async def _post_embed(texts: list[str]) -> list[list[float]]:
    """
    POST ``texts`` to the embedding service and return the vector list.

    Raises:
        HTTPException(502): If the service returns a non-200 status.
        HTTPException(503): If the service is unreachable after all retries.
    """
    client = _get_http_client()
    try:
        response = await client.post("/embed", json={"texts": texts})
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        logger.error("Embedding service unreachable: %s", exc)
        raise  # tenacity will retry; after max attempts we raise below

    if response.status_code != 200:
        logger.error(
            "Embedding service returned %d: %s",
            response.status_code,
            response.text[:200],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Embedding service returned HTTP {response.status_code}. "
                "Check EMBEDDING_SERVICE_URL and that the service is running."
            ),
        )

    data = response.json()
    embeddings: list[list[float]] = data.get("embeddings", [])

    if len(embeddings) != len(texts):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Embedding service returned {len(embeddings)} vectors "
                f"for {len(texts)} input texts."
            ),
        )

    return embeddings


# ── Public API ────────────────────────────────────────────────────────────────

async def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts via the remote embedding service.

    Sends texts in a single batch; the service is responsible for internal
    batching if needed.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of 384-dimensional float vectors, one per input text.

    Raises:
        HTTPException(502): Embedding service error.
        HTTPException(503): Embedding service unreachable.
    """
    if not texts:
        return []

    logger.debug("embed_documents: sending %d texts to embedding service", len(texts))
    try:
        return await _post_embed(texts)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error calling embedding service")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {exc}",
        ) from exc


async def embed_query(text: str) -> list[float]:
    """
    Embed a single query string via the remote embedding service.

    Args:
        text: Query string.

    Returns:
        384-dimensional float vector.

    Raises:
        HTTPException(502 / 503): If the embedding service fails.
    """
    vectors = await embed_documents([text])
    return vectors[0]
