"""Normal Chat API — the single conversation entry point.

`POST /api/chat` is normal chat by default and stays compatible with the old
contract (`message`, `history`). An optional `capabilities` list turns on
grounding for the message:

    []              -> normal chat (no retrieval — unchanged behaviour)
    ["rag"]         -> the existing RAG pipeline
    ["web_search"]  -> the existing Search pipeline

Every message is persisted against a conversation, and the LLM usage behind it
(provider, model, tokens, cost) is recorded against that conversation, so the chat
header and its menu have real data to show. Deep Research started from the chat
joins the same conversation id.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.chat import capabilities as chat_capabilities
from backend.conversations.store import DEFAULT_TITLE

router = APIRouter(tags=["chat"])

# How many words of the first message become the working title.
TITLE_WORDS = 6
TITLE_MAX_CHARS = 60


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    # Prior turns. Deliberately loose: history comes from the client and is
    # filtered by `backend/chat/history.py`, so one malformed turn must not
    # reject the user's actual message.
    history: List[Any] = Field(default_factory=list)
    # Optional capability ids (see GET /api/chat/capabilities).
    capabilities: List[str] = Field(default_factory=list)
    # The conversation this message belongs to; created when absent.
    conversation_id: Optional[str] = Field(default=None, max_length=200)


def _derive_title(message: str) -> str:
    """A sensible working title from the first message (no extra LLM call)."""
    words = (message or "").strip().split()
    if not words:
        return DEFAULT_TITLE
    title = " ".join(words[:TITLE_WORDS])
    if len(title) > TITLE_MAX_CHARS:
        title = title[:TITLE_MAX_CHARS].rstrip()
    if len(words) > TITLE_WORDS:
        title += "…"
    return title


def _conversation_summary(conv: Dict[str, Any]) -> Dict[str, Any]:
    from backend.api.conversations import _summary  # single formatting source

    return _summary(conv)


@router.get("/api/chat/capabilities")
def list_capabilities() -> dict:
    """The capabilities Chat can actually perform (no fake tools)."""
    return {
        "capabilities": chat_capabilities.describe(),
        "supported_combinations": [
            sorted(combo) for combo in chat_capabilities.SUPPORTED_COMBINATIONS
        ],
    }


@router.post("/api/chat")
def chat(req: ChatRequest, request: Request) -> dict:
    store = request.app.state.conversation_store
    conv = store.ensure(req.conversation_id)
    conversation_id = conv["id"]
    is_first_user_turn = (
        conv.get("message_count", 0) == 0 and conv.get("title") == DEFAULT_TITLE
    )

    # Server-side history is authoritative; the client's copy is a fallback for
    # requests that predate the conversation existing.
    history = store.history(conversation_id) or req.history
    store.add_message(conversation_id, "user", req.message)

    selection = request.app.state.provider_registry.mode_model("chat")
    try:
        payload = chat_capabilities.answer(
            req.message,
            llm=request.app.state.llm,
            engine=request.app.state.rag_engine,
            settings=request.app.state.settings,
            document_service=request.app.state.document_service,
            capabilities=req.capabilities,
            history=history,
            prefer=selection["provider"] or None,
            model=selection["model"] or None,
        )
    except chat_capabilities.UnsupportedCapabilityCombination as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    usage = payload.pop("usage", None) or {}
    message = store.add_message(
        conversation_id,
        "assistant",
        payload.get("answer") or "",
        provider=usage.get("provider") or (payload.get("provider") or ""),
        model=usage.get("model") or (payload.get("model") or ""),
        capabilities=payload.get("capabilities") or [],
        sources=payload.get("sources") or [],
        meta={
            "fallback": bool(payload.get("fallback")),
            "requested_provider": payload.get("requested_provider"),
            "grounded": payload.get("grounded"),
            "mode": payload.get("mode"),
        },
    )
    if usage:
        # Only real provider-reported usage is recorded — no estimation.
        store.record_usage(
            conversation_id,
            message_id=message["id"],
            provider=usage.get("provider") or "",
            model=usage.get("model") or "",
            input_tokens=usage.get("input_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            cost_usd=usage.get("cost_usd"),
            cost_known=bool(usage.get("cost_known")),
            latency_ms=usage.get("latency_ms") or 0,
            source="chat",
        )

    if is_first_user_turn:
        store.rename(conversation_id, _derive_title(req.message))

    refreshed = store.get(conversation_id) or conv
    refreshed["usage"] = store.usage(conversation_id)
    payload["conversation"] = _conversation_summary(refreshed)
    return payload
