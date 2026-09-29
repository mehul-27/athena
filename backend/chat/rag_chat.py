"""RAG Chat.

User -> Retriever -> relevant chunks -> grounded context -> LLM -> Answer + Sources.

Retrieval is mandatory. There is no web-search fallback and no agent fallback.
If no chunk clears the relevance threshold the LLM is never called and a clear
"not found in the documents" message is returned instead — the model is not
given the chance to answer from general knowledge.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.chat.history import clean_history
from backend.chat.prompts import GROUNDING_SYSTEM_PROMPT, NO_CONTEXT_MESSAGE, resolve_identity
from backend.providers.router import LLMRouter
from rag.engine import RagEngine
from rag.retrieval import build_context, normalize_query, retrieve, sources

logger = logging.getLogger(__name__)


def answer(
    query: str,
    engine: RagEngine,
    llm: LLMRouter,
    *,
    k: int = 5,
    threshold: float = 0.0,
    prefer: Optional[str] = None,
    model: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized = normalize_query(query)
    results, diag = retrieve(engine, normalized, k=k, threshold=threshold)
    source_names = [s["filename"] for s in sources(results)]

    logger.info(
        'RAG query="%s" retrieved=%d candidates=%d best_score=%.4f threshold=%.4f',
        normalized, len(results), diag["candidates"], diag["best_score"], diag["threshold"],
    )

    if not results:
        logger.info("RAG context_injected=false sources=[] (no chunk cleared threshold)")
        return {
            "mode": "rag",
            "answer": NO_CONTEXT_MESSAGE,
            "sources": [],
            "grounded": False,
            "context_injected": False,
            "retrieval": {
                "performed": True,
                "candidates": diag["candidates"],
                "kept": 0,
                "best_score": diag["best_score"],
                "threshold": diag["threshold"],
            },
            "provider": None,
            "provider_label": None,
            "model": None,
        }

    context = build_context(results)
    # The grounding instruction stays the system message untouched; the model's
    # identity rides along in the user turn so "which model are you" is still
    # answerable without weakening the context-only contract.
    identity = resolve_identity(llm, prefer)
    identity_note = f"(You are running on {identity}.)\n\n" if identity else ""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
        *clean_history(history),
        {
            "role": "user",
            "content": f"{identity_note}Document context:\n\n{context}\n\nQuestion: {query}",
        },
    ]

    logger.info(
        "RAG context_injected=true chunks=%d sources=%s",
        len(results), source_names,
    )
    routing: Dict[str, Any] = {}
    if prefer:
        routing["prefer"] = prefer
    if model:
        routing["model"] = model
    result = llm.chat(messages, **routing)
    return {
        "mode": "rag",
        "answer": result.content,
        "sources": sources(results),
        "grounded": True,
        "context_injected": True,
        "retrieval": {
            "performed": True,
            "candidates": diag["candidates"],
            "kept": len(results),
            "best_score": diag["best_score"],
            "threshold": diag["threshold"],
        },
        "provider": result.provider,
        "provider_label": result.label,
        "model": result.model,
        "requested_provider": result.requested_provider,
        "fallback": bool(result.fallback),
        "usage": result.usage_summary(),
    }
