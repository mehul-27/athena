"""Normal Chat.

User -> LLM -> Answer. Nothing else. No retrieval, no documents, no agent loop,
no automatic escalation. This is the deliberate opposite of Odysseus's
chat -> RAG -> agent-intent -> agent_loop chain (see MIGRATION_AUDIT.md §0).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.chat.history import clean_history
from backend.chat.prompts import normal_system_prompt, resolve_identity
from backend.providers.router import LLMRouter

logger = logging.getLogger(__name__)


def answer(
    query: str,
    llm: LLMRouter,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    prefer: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": normal_system_prompt(resolve_identity(llm, prefer))}
    ]
    messages.extend(clean_history(history))
    messages.append({"role": "user", "content": query})

    logger.info('Chat mode query=%r retrieval=false', query[:200])
    routing: Dict[str, Any] = {}
    if prefer:
        routing["prefer"] = prefer
    if model:
        routing["model"] = model
    result = llm.chat(messages, **routing)
    return {
        "mode": "chat",
        "answer": result.content,
        "sources": [],
        "grounded": False,
        "retrieval": {"performed": False},
        "provider": result.provider,
        "provider_label": result.label,
        "model": result.model,
        "requested_provider": result.requested_provider,
        "fallback": bool(result.fallback),
        "usage": result.usage_summary(),
    }
