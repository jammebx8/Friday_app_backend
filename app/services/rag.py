"""
RAG (Retrieval Augmented Generation) service.

Full pipeline:
  1. Embed the student question with BGE.
  2. Retrieve top-K semantically similar chunks from pgvector.
  3. Build a structured prompt with context + question.
  4. Call GPT-OSS-120B via Groq.
  5. Parse the JSON response and return a typed result.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.core.config import get_settings
from app.models.chunk import RetrievedChunk
from app.services.embeddings import embed_query
from app.services.groq import chat_completion
from app.services.vector_store import similarity_search

logger = logging.getLogger(__name__)


# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert educational tutor.

Rules:
- Answer ONLY using the provided context excerpts.
- If the answer is not present in the context, explicitly say: "This information is not available in the provided textbook sections."
- Always cite the page numbers where you found the information.
- Use clear, structured explanations suitable for students.
- For mathematical content, use LaTeX notation ($...$ inline, $$...$$ display).

Response format — return ONLY valid JSON with exactly these two keys:
{
  "answer": "<your detailed answer>",
  "citations": [<page_number>, ...]
}

No markdown fences, no preamble, no explanation outside the JSON object.
"""

_USER_TEMPLATE = """CONTEXT (retrieved textbook excerpts):
{context}

---
STUDENT QUESTION:
{question}"""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Excerpt {i} | Chapter: {chunk.chapter} | Page: {chunk.page}]"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def _parse_rag_response(raw: str) -> tuple[str, list[int]]:
    """
    Parse the model's JSON response.

    Returns:
        (answer_text, citation_page_list)

    Falls back gracefully if the model does not return valid JSON.
    """
    # Strip markdown code fences if the model wraps the JSON anyway
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
        answer = str(data.get("answer", "")).strip()
        raw_citations = data.get("citations", [])
        citations = sorted(set(int(c) for c in raw_citations if str(c).isdigit() or isinstance(c, int)))
        return answer, citations
    except (json.JSONDecodeError, ValueError, TypeError):
        # If JSON parsing fails, attempt to extract citations with a regex
        logger.warning("RAG response was not valid JSON; falling back to raw text")
        citations: list[int] = [int(m) for m in re.findall(r"\bpage\s+(\d+)\b", cleaned, re.IGNORECASE)]
        return cleaned, sorted(set(citations))


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """Structured output from the RAG pipeline."""

    answer: str
    citations: list[int]
    retrieved_chunks: list[RetrievedChunk]


async def answer_question(
    question: str,
    book_id: str,
    top_k: int | None = None,
) -> RAGResult:
    """
    Full RAG pipeline: embed → retrieve → prompt → generate → parse.

    Args:
        question: The student's natural-language question.
        book_id:  UUID of the book to search.
        top_k:    Number of context chunks to retrieve.
                  Defaults to ``settings.top_k_chunks``.

    Returns:
        RAGResult with answer, citations, and the retrieved chunks.

    Raises:
        ValueError: If question is empty.
        RuntimeError: If retrieval or generation fails fatally.
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")

    settings = get_settings()
    top_k = top_k or settings.top_k_chunks

    # Step 1 — Embed query
    logger.info("RAG: embedding question for book_id=%s", book_id)
    query_vector = await embed_query(question)

    # Step 2 — Retrieve top-K chunks
    chunks = await similarity_search(
        query_embedding=query_vector,
        book_id=book_id,
        top_k=top_k,
    )

    if not chunks:
        logger.warning("RAG: no chunks found for book_id=%s", book_id)
        return RAGResult(
            answer="No relevant content was found in this textbook for your question.",
            citations=[],
            retrieved_chunks=[],
        )

    # Step 3 — Build context + prompt
    context_block = _build_context_block(chunks)
    user_message = _USER_TEMPLATE.format(context=context_block, question=question.strip())

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Step 4 — Generate answer
    logger.info("RAG: calling generation model with %d context chunks", len(chunks))
    raw_response = await chat_completion(
        messages=messages,
        model=settings.openai_model,
        temperature=0.2,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    # Step 5 — Parse and return
    answer, citations = _parse_rag_response(raw_response)

    logger.info(
        "RAG: answer generated — citations=%s chunks_used=%d",
        citations,
        len(chunks),
    )

    return RAGResult(
        answer=answer,
        citations=citations,
        retrieved_chunks=chunks,
    )
