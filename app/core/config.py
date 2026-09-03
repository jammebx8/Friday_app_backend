"""
Core configuration using pydantic-settings.
Loads from .env file and validates all required environment variables.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq Cloud API key")
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq OpenAI-compatible base URL",
    )

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_key: str = Field(..., description="Supabase service-role or anon key")

    # ── Model identifiers ─────────────────────────────────────────────────────
    openai_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Generation model served via Groq",
    )
    vision_model: str = Field(
        default="qwen/qwen3-72b",
        description="Vision / OCR model served via Groq",
    )
    embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local HuggingFace embedding model",
    )

    # ── RAG tuning ────────────────────────────────────────────────────────────
    top_k_chunks: int = Field(default=5, description="Number of chunks to retrieve")
    chunk_size_tokens: int = Field(default=600, description="Target chunk size in tokens")
    chunk_overlap_tokens: int = Field(default=80, description="Overlap between chunks")

    # ── PDF processing ────────────────────────────────────────────────────────
    pdf_dpi: int = Field(default=150, description="DPI for page-to-image rendering (150 sufficient for OCR, saves memory)")
    ocr_batch_size: int = Field(default=5, description="Pages per OCR request")
    ocr_max_retries: int = Field(default=3, description="OCR retry attempts")
    ocr_retry_base_delay: float = Field(
        default=1.5, description="Base delay (seconds) for exponential back-off"
    )

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    log_level: Literal["debug", "info", "warning", "error"] = Field(default="info")

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins (restrict in production)",
    )

    # ── External embedding service ────────────────────────────────────────────
    embedding_service_url: str = Field(
        ...,
        description=(
            "Base URL of the external embedding service, e.g. "
            "https://my-embed-worker.modal.run"
        ),
    )
    embedding_service_api_key: str = Field(
        default="",
        description=(
            "Optional bearer token sent as 'Authorization: Bearer <key>'. "
            "Leave empty if the service requires no authentication."
        ),
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    tmp_dir: str = Field(
        default="/tmp/friday_rag",
        description="Temporary directory for PDF images",
    )

    @field_validator("groq_api_key", "supabase_url", "supabase_key", "embedding_service_url", mode="before")
    @classmethod
    def _must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Required environment variable must not be empty")
        return v.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
