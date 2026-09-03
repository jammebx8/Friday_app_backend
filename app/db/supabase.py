"""
Async Supabase client — single module, lazily initialised.

Uses the official `supabase` Python SDK which wraps `httpx` under the hood.
All public helpers are thin async wrappers so callers never import the SDK
directly; swap the transport layer here if needed.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from supabase import AsyncClient, acreate_client

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton (set during app lifespan startup)
_client: AsyncClient | None = None


async def init_supabase() -> None:
    """Initialise the global async Supabase client.

    Call this once from the FastAPI lifespan startup hook.
    """
    global _client
    if _client is not None:
        return
    settings = get_settings()
    _client = await acreate_client(settings.supabase_url, settings.supabase_key)
    logger.info("Supabase async client initialised (url=%s)", settings.supabase_url)


async def close_supabase() -> None:
    """Close the Supabase HTTP session on application shutdown."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()  # type: ignore[attr-defined]
        except Exception:
            pass
        _client = None
        logger.info("Supabase client closed")


def get_client() -> AsyncClient:
    """Return the initialised Supabase client.

    Raises RuntimeError if called before ``init_supabase()``.
    """
    if _client is None:
        raise RuntimeError(
            "Supabase client not initialised. "
            "Ensure init_supabase() is awaited during app startup."
        )
    return _client


# ── Low-level helpers ─────────────────────────────────────────────────────────


async def db_insert(table: str, data: dict[str, Any]) -> dict[str, Any]:
    """Insert a single row and return it."""
    client = get_client()
    response = await client.table(table).insert(data).execute()
    return response.data[0]


async def db_insert_many(table: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bulk-insert rows (no return data by default to keep payload small)."""
    client = get_client()
    response = await client.table(table).insert(rows).execute()
    return response.data


async def db_select(
    table: str,
    columns: str = "*",
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Generic SELECT with optional equality filters."""
    client = get_client()
    query = client.table(table).select(columns)
    for col, val in (filters or {}).items():
        query = query.eq(col, val)
    if limit is not None:
        query = query.limit(limit)
    response = await query.execute()
    return response.data


async def db_update(
    table: str,
    data: dict[str, Any],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """UPDATE rows matching *filters* with *data*."""
    client = get_client()
    query = client.table(table).update(data)
    for col, val in filters.items():
        query = query.eq(col, val)
    response = await query.execute()
    return response.data


async def db_rpc(fn_name: str, params: dict[str, Any]) -> Any:
    """Call a Supabase / Postgres RPC function."""
    client = get_client()
    response = await client.rpc(fn_name, params).execute()
    return response.data
