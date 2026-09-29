"""Search API — a standalone capability kept for API compatibility.

The primary UI now exposes web search as a Chat toggle; this endpoint remains so
existing scripts/installs keep working. Both paths share the same implementation
(`backend/chat/search_chat.py`), so nothing is duplicated.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.chat import search_chat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = "web"  # web | news
    answer: Optional[bool] = None
    fetch: bool = True


class FetchRequest(BaseModel):
    url: str = Field(..., min_length=1)


@router.post("")
def search_endpoint(req: SearchRequest, request: Request) -> dict:
    settings = request.app.state.settings
    mode = (req.mode or "web").lower()
    results, sources = search_chat.collect_results(
        req.query, settings=settings, mode=mode, fetch=req.fetch
    )

    want_answer = settings.search_answer if req.answer is None else req.answer
    selection = request.app.state.provider_registry.mode_model("search")
    answer_text = provider = provider_label = used_model = requested_provider = None
    fallback = False
    if want_answer and results:
        result = search_chat.synthesize(
            req.query,
            results,
            request.app.state.llm,
            prefer=selection["provider"] or None,
            model=selection["model"] or None,
        )
        answer_text = result.content
        provider = result.provider
        provider_label = result.label
        used_model = result.model
        requested_provider = result.requested_provider
        fallback = bool(result.fallback)

    return {
        "mode": "search",
        "query": req.query,
        "search_mode": mode,
        "answer": answer_text,
        "provider": provider,
        "provider_label": provider_label,
        "model": used_model,
        "requested_provider": requested_provider,
        "fallback": fallback,
        "results": results,
        "sources": sources,
    }


@router.post("/fetch")
def fetch_endpoint(req: FetchRequest, request: Request) -> dict:
    from search.fetch import fetch_webpage_content

    page = fetch_webpage_content(req.url, settings=request.app.state.settings)
    if not page.get("success"):
        raise HTTPException(status_code=400, detail=page.get("error", "Fetch failed"))
    return page
