"""
Friday RAG — FastAPI application entry point.

Lifespan:
  startup  — init Supabase client, warm up BGE embedding model
  shutdown — close Supabase HTTP session

Routers:
  /books  — PDF upload + status
  /chat   — RAG question answering
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.books import router as books_router
from app.api.chat import router as chat_router
from app.core.config import get_settings
from app.db.supabase import close_supabase, init_supabase
from app.services.embeddings import warm_up_embeddings

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context — startup / shutdown hooks."""
    settings = get_settings()
    logger.info("=== Friday RAG starting up ===")

    # Initialise Supabase async client
    await init_supabase()

    # Pre-load BGE model so first request has no cold-start lag
    logger.info("Pre-loading BGE embedding model: %s", settings.embed_model)
    warm_up_embeddings()
    logger.info("Embedding model warm-up complete")

    logger.info("=== Friday RAG ready — listening on %s:%d ===", settings.host, settings.port)
    yield  # Application runs here

    # Shutdown
    logger.info("=== Friday RAG shutting down ===")
    await close_supabase()


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Friday RAG API",
        description=(
            "Production RAG backend for scanned educational textbooks. "
            "Ingests PDFs via Groq Vision OCR, embeds with BGE, stores in "
            "Supabase pgvector, and answers student questions with GPT-OSS-120B."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(books_router)
    app.include_router(chat_router)

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["meta"], summary="Health check")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "friday-rag"})

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
