"""Chat capabilities.

One conversation, with optional capabilities the user toggles in the composer:

    Chat
     ├── (none)              → normal chat, no retrieval at all
     ├── web_search          → the existing Search pipeline
     ├── rag                 → the existing RAG pipeline
     └── future: mcp, ...

This module is the single dispatch point. It does **not** reimplement any
retrieval: `rag` calls the same `rag_chat.answer` that `/api/rag/chat` uses, and
`web_search` calls the same `search_chat.answer` that `/api/search` uses. The
implementation boundaries stay internal; the UI never sees them.

Combinations that the backend does not safely support are rejected explicitly
rather than faked. `SUPPORTED_COMBINATIONS` is the one place to change when a
combined retrieval path is implemented.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from backend.chat import normal, rag_chat, search_chat
from backend.config import Settings
from backend.providers.router import LLMRouter
from rag.engine import RagEngine

# Capability ids the UI may send. Keep in sync with the composer's buttons.
CAPABILITIES = ("web_search", "rag")

# Exactly which capability sets are implemented today. `{web_search, rag}`
# (combined retrieval) is deliberately absent: the UI keeps that combination
# disabled until there is a backend path that merges both contexts honestly.
SUPPORTED_COMBINATIONS: FrozenSet[FrozenSet[str]] = frozenset({
    frozenset(),
    frozenset({"web_search"}),
    frozenset({"rag"}),
})


class UnsupportedCapabilityCombination(ValueError):
    """Raised for a capability set the backend does not implement."""

    def __init__(self, capabilities: Iterable[str]) -> None:
        self.capabilities = sorted(capabilities)
        super().__init__(
            "Unsupported capability combination: "
            + (", ".join(self.capabilities) or "none")
            + ". Web Search and RAG cannot be enabled together yet."
        )


def normalize(capabilities: Optional[Iterable[str]]) -> FrozenSet[str]:
    """Clean an incoming capability list, ignoring unknown/blank entries."""
    if not capabilities:
        return frozenset()
    known = {str(cap).strip().lower() for cap in capabilities}
    return frozenset(cap for cap in known if cap in CAPABILITIES)


def is_supported(capabilities: Optional[Iterable[str]]) -> bool:
    return normalize(capabilities) in SUPPORTED_COMBINATIONS


def describe() -> List[Dict[str, Any]]:
    """Capability metadata for the UI (no fake tools — only real ones)."""
    return [
        {"id": "web_search", "label": "Web Search",
         "description": "Ground the answer in live web search results."},
        {"id": "rag", "label": "RAG",
         "description": "Ground the answer in your uploaded documents."},
    ]


def answer(
    message: str,
    *,
    llm: LLMRouter,
    engine: RagEngine,
    settings: Settings,
    document_service=None,
    capabilities: Optional[Iterable[str]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    prefer: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Route one chat message according to the enabled capabilities."""
    caps = normalize(capabilities)
    if caps not in SUPPORTED_COMBINATIONS:
        raise UnsupportedCapabilityCombination(caps)

    if not caps:
        result = normal.answer(message, llm, history, prefer=prefer, model=model)
    elif caps == frozenset({"rag"}):
        result = rag_chat.answer(
            message,
            engine,
            llm,
            k=settings.rag_top_k,
            threshold=settings.rag_relevance_threshold,
            prefer=prefer,
            model=model,
            history=history,
        )
    elif caps == frozenset({"web_search"}):
        result = search_chat.answer(
            message,
            llm=llm,
            settings=settings,
            prefer=prefer,
            model=model,
            history=history,
        )
    else:  # pragma: no cover - guarded by SUPPORTED_COMBINATIONS above
        raise UnsupportedCapabilityCombination(caps)

    result["capabilities"] = sorted(caps)
    return result
