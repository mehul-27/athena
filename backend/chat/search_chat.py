"""Web Search as a Chat capability (and the engine behind `POST /api/search`).

The search implementation itself is unchanged — this module only factors the
existing search-then-answer flow out of the HTTP layer so it can be reused by
both the standalone Search endpoint and the unified Chat capability dispatch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.chat.history import clean_history
from backend.chat.prompts import resolve_identity, search_system_prompt
from backend.config import Settings
from backend.providers.router import LLMRouter
from search.fetch import fetch_webpage_content
from search.news import news_search
from search.web import web_search

logger = logging.getLogger(__name__)


def collect_results(
    query: str,
    *,
    settings: Settings,
    mode: str = "web",
    fetch: bool = True,
) -> Tuple[List[dict], List[dict]]:
    """Run the configured search provider(s) and optionally fetch page content.

    Returns `(results, sources)`. Identical behaviour to the standalone Search
    page — including the top-N page fetch used for grounding.
    """
    mode = (mode or "web").lower()
    if mode == "news":
        results = news_search(query, settings=settings)
    else:
        results = web_search(query, settings=settings)

    if fetch and results:
        for row in results[: settings.search_fetch_pages]:
            try:
                page = fetch_webpage_content(row["url"], settings=settings)
                if page.get("success"):
                    row["content"] = page.get("content")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Fetch for %s failed: %s", row.get("url"), exc)

    sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]
    return results, sources


def synthesize(
    query: str,
    results: List[dict],
    llm: LLMRouter,
    *,
    prefer: Optional[str] = None,
    model: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
):
    """Ask the LLM to answer from the supplied search results."""
    context_parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        url = r.get("url") or ""
        body = r.get("content") or r.get("snippet") or ""
        context_parts.append(f"[{i}] {title} ({url})\n{body[:3000]}")
    context = "\n\n".join(context_parts)

    identity = resolve_identity(llm, prefer)
    messages = [
        {"role": "system", "content": search_system_prompt(identity)},
        *clean_history(history),
        {"role": "user", "content": f"Search results:\n\n{context}\n\nQuestion: {query}"},
    ]
    logger.info("Search answer synthesis sources=%d", len(results))
    routing: Dict[str, Any] = {}
    if prefer:
        routing["prefer"] = prefer
    if model:
        routing["model"] = model
    return llm.chat(messages, **routing)


def answer(
    query: str,
    *,
    llm: LLMRouter,
    settings: Settings,
    prefer: Optional[str] = None,
    model: Optional[str] = None,
    mode: str = "web",
    fetch: bool = True,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Search + grounded answer, shaped for a Chat message."""
    results, sources = collect_results(query, settings=settings, mode=mode, fetch=fetch)

    answer_text = None
    provider = provider_label = used_model = None
    requested_provider = None
    fallback = False
    usage = None
    if results:
        result = synthesize(query, results, llm, prefer=prefer, model=model, history=history)
        answer_text = result.content
        provider = result.provider
        provider_label = result.label
        used_model = result.model
        requested_provider = result.requested_provider
        fallback = bool(result.fallback)
        usage = result.usage_summary()

    return {
        "mode": "search",
        "query": query,
        "search_mode": mode,
        "answer": answer_text,
        "provider": provider,
        "provider_label": provider_label,
        "model": used_model,
        "requested_provider": requested_provider,
        "fallback": fallback,
        "usage": usage,
        "results": results,
        "sources": sources,
    }
