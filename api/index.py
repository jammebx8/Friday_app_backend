"""
Vercel Python serverless entry point.

Vercel looks for `api/index.py` and expects an ASGI `app` object.
This module simply re-exports the FastAPI application from app/main.py.
"""
import sys
import os

# Ensure the project root is on the path so `app.*` imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: F401  — Vercel picks this up as the ASGI handler
