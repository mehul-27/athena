"""News search.

Odysseus has no standalone news module — "news" is a recency-biased mode of web
search (`services/search/providers.py::searxng_search_api` switches to
`categories=news`). Athena mirrors that: use the news category when the provider
supports it (SearXNG), otherwise fall back to a recency-filtered web search.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from backend.config import Settings, get_settings
from search.web import _searxng, web_search

logger = logging.getLogger(__name__)

_NEWS_HINTS = ("news", "headlines", "breaking", "latest", "today", "update")


def news_search(
    query: str,
    *,
    count: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> List[dict]:
    settings = settings or get_settings()
    provider = (settings.search_provider or "duckduckgo").lower()
    count = count or settings.search_result_count

    if provider == "searxng":
        try:
            results = _searxng(query, count, settings, time_filter="week", categories="news")
            if results:
                logger.info("News search (searxng) query=%r results=%d", query, len(results))
                return results
        except Exception as exc:
            logger.warning("SearXNG news search failed for %r: %s", query, exc)

    # Providers without a dedicated news endpoint: recency-filtered web search.
    results = web_search(query, count=count, settings=settings, time_filter="week")
    logger.info("News search (fallback web) query=%r results=%d", query, len(results))
    return results
