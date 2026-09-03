"""
embeddings.py — STUB for the Vercel deployment.

All embedding logic has been moved to an external service.
The Vercel FastAPI app calls that service through
``app.services.embedding_client``.

This file is kept so that any accidental import of the old module fails
with a clear, actionable error rather than a silent AttributeError.
"""

raise ImportError(
    "app.services.embeddings is no longer available on the Vercel deployment. "
    "Import from app.services.embedding_client instead:\n"
    "  from app.services.embedding_client import embed_documents, embed_query"
)
