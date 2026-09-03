"""
Friday RAG — FastAPI application entry point.

Lifespan:
  startup  — init Supabase client
  shutdown — close HTTP sessions

Routers:
  /books  — PDF process + status
  /chat   — RAG question answering
  /debug  — deployment diagnostics (safe, read-only)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.books import router as books_router
from app.api.chat import router as chat_router
from app.core.config import get_settings
from app.db.supabase import close_supabase, init_supabase
from app.services.embedding_client import close_embedding_client

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Request timing middleware ─────────────────────────────────────────────────

_APP_START_TIME = time.time()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info("=== Friday RAG starting up (Python %s) ===", sys.version.split()[0])
    logger.info("CORS origins: %s", settings.cors_origins)
    logger.info("Embedding service: %s", settings.embedding_service_url)

    await init_supabase()
    logger.info("Supabase client ready")

    logger.info("=== Friday RAG ready ===")
    yield

    logger.info("=== Friday RAG shutting down ===")
    await close_embedding_client()
    await close_supabase()


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Friday RAG API",
        version="1.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS — must be the FIRST middleware ───────────────────────────────────
    # allow_origins=["*"] is intentionally broad so every preflight succeeds.
    # Tighten to specific origins once everything is working.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # override with settings.cors_origins in prod
        allow_credentials=False,      # must be False when allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # ── Request logging middleware ────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        origin = request.headers.get("origin", "-")
        logger.info(
            "→ %s %s  origin=%s  content-length=%s",
            request.method,
            request.url.path,
            origin,
            request.headers.get("content-length", "?"),
        )
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "← %s %s  status=%d  %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(books_router)
    app.include_router(chat_router)

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["meta"])
    async def health() -> JSONResponse:
        """Liveness probe — also confirms CORS headers are present."""
        return JSONResponse({
            "status": "ok",
            "service": "friday-rag",
            "version": "1.1.0",
            "uptime_seconds": round(time.time() - _APP_START_TIME),
        })

    # ── Debug endpoint ────────────────────────────────────────────────────────
    @app.get("/debug", tags=["meta"])
    async def debug(request: Request) -> JSONResponse:
        """
        Safe read-only endpoint that confirms the deployment is running the
        latest code and shows which endpoints exist.

        Hit this first when diagnosing issues:
          curl https://friday-app-backend.vercel.app/debug
        """
        settings = get_settings()

        routes = [
            {"method": list(r.methods or []), "path": r.path}
            for r in app.routes
            if hasattr(r, "methods")
        ]

        return JSONResponse({
            "deployment": "ok",
            "version": "1.1.0",
            "python": sys.version.split()[0],
            "endpoints": routes,
            "cors_origins_configured": settings.cors_origins,
            "cors_middleware": "allow_origins=['*']",
            "embedding_service_url": settings.embedding_service_url or "NOT SET",
            "supabase_url": settings.supabase_url[:40] + "..." if settings.supabase_url else "NOT SET",
            "request_headers": dict(request.headers),
            "uptime_seconds": round(time.time() - _APP_START_TIME),
        })

    # ── CORS preflight debug ──────────────────────────────────────────────────
    @app.options("/{rest_of_path:path}", tags=["meta"])
    async def options_handler(rest_of_path: str) -> JSONResponse:
        """
        Explicit OPTIONS handler so Vercel never intercepts preflight requests
        before FastAPI can respond with CORS headers.
        """
        return JSONResponse(
            content={"ok": True},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            },
        )

    return app


app = create_app()


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        reload=False,
    )
