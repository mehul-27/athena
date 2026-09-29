"""Conversations API — persistence + usage for the chat header and its menu.

Implements the Athena versions of Odysseus's header actions:

* **Rename** — persists to the conversation row (never DOM-only).
* **Compact** — summarises the older half into a `[Conversation summary]` system
  message and keeps the recent tail, exactly as Odysseus does, with one
  difference: the original messages are **archived, not deleted**, so compaction
  never destroys the user's transcript.
* **Copy Chat / PDF** — served by `GET /api/conversations/{id}/transcript`
  (plain text; the browser prints it to PDF, which is how Odysseus does it too).
* **Save to Documents** — persists a real document row, retrievable later.
* **Delete Chat** — removes the conversation, its messages and its usage, and
  detaches (never deletes) saved documents and research jobs. Never touches the
  RAG knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from backend.providers.pricing import format_cost

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["conversations"])

# Odysseus skips compaction on short threads; same guard here.
MIN_MESSAGES_TO_COMPACT = 6


class CreateConversation(BaseModel):
    title: Optional[str] = None


class RenameConversation(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class SaveDocument(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None


def _store(request: Request):
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        from backend.conversations.store import ConversationStore

        store = ConversationStore(request.app.state.settings)
        request.app.state.conversation_store = store
    return store


def _summary(conv: Dict[str, Any]) -> Dict[str, Any]:
    """The shape the chat header renders (never fabricates a cost)."""
    usage = conv.get("usage") or {}
    totals = usage.get("totals") or {}
    return {
        "id": conv["id"],
        "title": conv["title"],
        "message_count": conv.get("message_count", 0),
        "request_count": conv.get("request_count", 0),
        "total_tokens": conv.get("total_tokens", 0),
        "input_tokens": conv.get("total_input_tokens", 0),
        "output_tokens": conv.get("total_output_tokens", 0),
        "cost_usd": totals.get("cost_usd") if usage else None,
        "cost_known": totals.get("cost_known", False) if usage else False,
        "cost_display": format_cost(totals.get("cost_usd")) if totals.get("cost_known") else "$—",
        "unpriced_requests": totals.get("unpriced_requests", 0),
        "by_model": usage.get("by_model", []),
        "created_at": conv.get("created_at"),
        "updated_at": conv.get("updated_at"),
    }


def _require_conversation(request: Request, conversation_id: str) -> Dict[str, Any]:
    conv = _store(request).get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


# ----------------------------------------------------------------------
# conversations
# ----------------------------------------------------------------------
@router.post("/conversations")
def create_conversation(body: CreateConversation, request: Request) -> dict:
    from backend.conversations.store import DEFAULT_TITLE

    conv = _store(request).create(body.title or DEFAULT_TITLE)
    conv["usage"] = _store(request).usage(conv["id"])
    return {"conversation": _summary(conv)}


@router.get("/conversations")
def list_conversations(request: Request, limit: int = 100) -> dict:
    return {"conversations": _store(request).list(limit=limit)}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, request: Request) -> dict:
    store = _store(request)
    conv = _require_conversation(request, conversation_id)
    conv["usage"] = store.usage(conversation_id)
    return {
        "conversation": _summary(conv),
        "messages": store.messages(conversation_id),
    }


@router.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, body: RenameConversation, request: Request) -> dict:
    store = _store(request)
    _require_conversation(request, conversation_id)
    if not store.rename(conversation_id, body.title):
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    conv = store.get(conversation_id)
    conv["usage"] = store.usage(conversation_id)
    return {"conversation": _summary(conv)}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request) -> dict:
    store = _store(request)
    _require_conversation(request, conversation_id)
    result = store.delete(conversation_id)
    # Research jobs keep their reports; only the association is cleared.
    try:
        from backend.research.store import ResearchStore

        detached = ResearchStore(request.app.state.settings).detach_conversation(conversation_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not detach research jobs from %s: %s", conversation_id, exc)
        detached = 0
    return {**result, "detached_research": detached}


# ----------------------------------------------------------------------
# transcript (Copy Chat / PDF)
# ----------------------------------------------------------------------
def _render_transcript(conv: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    """Readable plain-text transcript — user/assistant turns plus sources."""
    lines: List[str] = [f"# {conv['title']}", ""]
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            lines += ["**User**", msg.get("content", ""), ""]
        elif role == "assistant":
            attribution = ""
            if msg.get("provider") or msg.get("model"):
                attribution = f" _(via {msg.get('provider') or '?'} · {msg.get('model') or '?'})_"
            lines += [f"**Athena**{attribution}", msg.get("content", "")]
            sources = msg.get("sources") or []
            if sources:
                lines.append("Sources:")
                for src in sources:
                    label = src.get("title") or src.get("filename") or src.get("url") or "source"
                    extra = []
                    if src.get("chunk_id") is not None:
                        extra.append(f"chunk {src['chunk_id']}")
                    if isinstance(src.get("score"), (int, float)):
                        extra.append(f"score {src['score']:.2f}")
                    if src.get("url"):
                        extra.append(src["url"])
                    lines.append(f"- {label}" + (f" ({', '.join(extra)})" if extra else ""))
            lines.append("")
        # system/note rows are internal noise; kept out of the transcript
    return "\n".join(lines).strip() + "\n"


@router.get("/conversations/{conversation_id}/transcript", response_class=PlainTextResponse)
def conversation_transcript(conversation_id: str, request: Request) -> PlainTextResponse:
    store = _store(request)
    conv = _require_conversation(request, conversation_id)
    text = _render_transcript(conv, store.messages(conversation_id))
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


# ----------------------------------------------------------------------
# compact
# ----------------------------------------------------------------------
_COMPACT_PROMPT = (
    "Summarise the conversation so far so it can replace the earlier turns.\n"
    "Capture the user's goals, decisions, facts and any unresolved questions.\n"
    "Write a dense factual summary; no preamble, no headings, no bullet nesting.\n"
    f"There are {{count}} earlier messages to summarise."
)


@router.post("/conversations/{conversation_id}/compact")
async def compact_conversation(conversation_id: str, request: Request) -> dict:
    import asyncio

    store = _store(request)
    _require_conversation(request, conversation_id)
    messages = store.messages(conversation_id)
    if len(messages) < MIN_MESSAGES_TO_COMPACT:
        raise HTTPException(status_code=400, detail="Not enough messages to compact")

    recent_keep = min(8, max(4, len(messages) // 4))
    older = messages[:-recent_keep]
    recent = messages[-recent_keep:]

    llm = request.app.state.llm
    if not getattr(llm, "configured", False):
        raise HTTPException(status_code=503, detail="No LLM provider is configured")

    convo_text = "\n".join(
        f"{m['role'].upper()}: {(m.get('content') or '')[:2000]}" for m in older
    )
    selection = request.app.state.provider_registry.mode_model("chat")
    routing: Dict[str, Any] = {}
    if selection["provider"]:
        routing["prefer"] = selection["provider"]
    if selection["model"]:
        routing["model"] = selection["model"]

    try:
        result = await asyncio.to_thread(
            llm.chat,
            [
                {"role": "system", "content": _COMPACT_PROMPT.replace("{count}", str(len(older)))},
                {"role": "user", "content": convo_text},
            ],
            max_tokens=1024,
            **routing,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Compaction failed: {exc}") from exc

    summary = (getattr(result, "content", "") or "").strip()
    if not summary:
        raise HTTPException(status_code=502, detail="Compaction produced no summary")

    # The summary call is itself usage against this conversation.
    store.record_usage(
        conversation_id,
        provider=getattr(result, "provider", "") or "",
        model=getattr(result, "model", "") or "",
        input_tokens=getattr(result, "input_tokens", 0) or 0,
        output_tokens=getattr(result, "output_tokens", 0) or 0,
        cost_usd=getattr(result, "cost_usd", None),
        cost_known=bool(getattr(result, "cost_known", False)),
        latency_ms=getattr(result, "latency_ms", 0) or 0,
        source="compact",
    )

    store.add_message(
        conversation_id, "system", f"[Conversation summary]\n{summary}",
        provider=getattr(result, "provider", "") or "",
        model=getattr(result, "model", "") or "",
        meta={"compacted": True, "summarized_count": len(older)},
    )
    # Archive (not delete) the turns the summary replaces.
    archived = store.mark_compacted([m["id"] for m in older])
    store.touch(conversation_id)

    conv = store.get(conversation_id)
    conv["usage"] = store.usage(conversation_id)
    return {
        "conversation": _summary(conv),
        "summarized": len(older),
        "kept": len(recent),
        "archived": archived,
    }


# ----------------------------------------------------------------------
# save to documents
# ----------------------------------------------------------------------
@router.post("/conversations/{conversation_id}/save")
def save_conversation_document(conversation_id: str, request: Request,
                               body: Optional[SaveDocument] = None) -> dict:
    store = _store(request)
    conv = _require_conversation(request, conversation_id)
    payload = body or SaveDocument()
    content = payload.content if payload.content else _render_transcript(conv, store.messages(conversation_id))
    doc = store.save_document(conversation_id, payload.title or conv["title"], content)
    logger.info("Saved conversation %s to document %s", conversation_id, doc["id"])
    return {"ok": True, "document": doc}


@router.get("/saved-documents")
def list_saved_documents(request: Request) -> dict:
    return {"documents": _store(request).saved_documents()}


@router.get("/saved-documents/{doc_id}", response_class=PlainTextResponse)
def get_saved_document(doc_id: str, request: Request) -> PlainTextResponse:
    doc = _store(request).saved_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return PlainTextResponse(doc["content"], media_type="text/plain; charset=utf-8")


@router.delete("/saved-documents/{doc_id}")
def delete_saved_document(doc_id: str, request: Request) -> dict:
    if not _store(request).delete_saved_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True}
