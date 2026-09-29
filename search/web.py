"""Web search providers with a primary→fallback chain.

Adapted from Odysseus `services/search/providers.py` + `services/search/core.py`.
Provider set and result shape (`{title, url, snippet, age}`) are preserved; the
file-based cache, analytics and admin-settings surface are dropped.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

FALLBACK_ORDER = ["duckduckgo"]
VALID_PROVIDERS = {
    "searxng", "brave", "duckduckgo", "google_pse", "tavily", "serper", "disabled",
}
_TIMEOUT = 15.0


def _chain(primary: str) -> List[str]:
    chain = [primary]
    for fb in FALLBACK_ORDER:
        if fb not in chain:
            chain.append(fb)
    return [p for p in chain if p != "disabled"]


# ----------------------------------------------------------------------
# DuckDuckGo (no API key)
# ----------------------------------------------------------------------
def _is_ddg_host(host: str) -> bool:
    host = (host or "").lower()
    return host == "duckduckgo.com" or host.endswith(".duckduckgo.com")


def _resolve_ddg_redirect(raw: str) -> str:
    if not raw:
        return raw
    resolved = raw
    if resolved.startswith("//"):
        resolved = "https:" + resolved
    elif resolved.startswith("/"):
        resolved = urljoin("https://html.duckduckgo.com", resolved)
    try:
        parsed = urlparse(resolved)
        if _is_ddg_host(parsed.hostname) and parsed.path.rstrip("/") == "/l":
            qs = parse_qs(parsed.query)
            if "uddg" in qs:
                return qs["uddg"][0]
    except Exception:
        pass
    return resolved


def _duckduckgo(query: str, count: int, settings: Settings, time_filter: Optional[str] = None) -> List[dict]:
    resp = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": settings.web_fetch_user_agent},
        timeout=_TIMEOUT,
        follow_redirects=True,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for block in soup.select(".result")[:count]:
        link = block.select_one(".result__a")
        if not link:
            continue
        url = _resolve_ddg_redirect(link.get("href", ""))
        if not url:
            continue
        snippet = block.select_one(".result__snippet")
        results.append(
            {
                "title": link.get_text(" ", strip=True),
                "url": url,
                "snippet": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
    return results


# ----------------------------------------------------------------------
# SearXNG (self-hosted, JSON API)
# ----------------------------------------------------------------------
def _searxng(query: str, count: int, settings: Settings, time_filter: Optional[str] = None,
             categories: str = "general") -> List[dict]:
    instance = (settings.searxng_instance or "").rstrip("/")
    if not instance:
        return []
    params: Dict[str, Any] = {"q": query, "format": "json", "language": "en", "categories": categories}
    if time_filter in ("day", "week", "month", "year"):
        params["time_range"] = time_filter
    resp = httpx.get(
        f"{instance}/search",
        params=params,
        headers={"User-Agent": settings.web_fetch_user_agent},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])[:count]
        if r.get("url")
    ]


# ----------------------------------------------------------------------
# Keyed providers
# ----------------------------------------------------------------------
def _brave(query: str, count: int, settings: Settings, time_filter: Optional[str] = None) -> List[dict]:
    if not settings.brave_api_key:
        return []
    params: Dict[str, Any] = {"q": query, "count": count}
    if time_filter in ("day", "week", "month", "year"):
        params["freshness"] = time_filter
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": settings.brave_api_key, "Accept": "application/json"},
        params=params,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    items = (resp.json().get("web") or {}).get("results", [])
    return [
        {"title": i.get("title", ""), "url": i.get("url", ""),
         "snippet": i.get("description", ""), "age": i.get("date", "")}
        for i in items[:count] if i.get("url")
    ]


def _tavily(query: str, count: int, settings: Settings, time_filter: Optional[str] = None) -> List[dict]:
    if not settings.tavily_api_key:
        return []
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={"query": query, "max_results": count, "include_answer": False},
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return [
        {"title": i.get("title", ""), "url": i.get("url", ""),
         "snippet": i.get("content", ""), "age": i.get("published_date", "")}
        for i in resp.json().get("results", [])[:count] if i.get("url")
    ]


def _serper(query: str, count: int, settings: Settings, time_filter: Optional[str] = None) -> List[dict]:
    if not settings.serper_api_key:
        return []
    resp = httpx.post(
        "https://google.serper.dev/search",
        json={"q": query, "num": count},
        headers={"X-API-KEY": settings.serper_api_key},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return [
        {"title": i.get("title", ""), "url": i.get("link", ""), "snippet": i.get("snippet", "")}
        for i in resp.json().get("organic", [])[:count] if i.get("link")
    ]


def _google_pse(query: str, count: int, settings: Settings, time_filter: Optional[str] = None) -> List[dict]:
    if not settings.google_pse_key or not settings.google_pse_cx:
        return []
    resp = httpx.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": settings.google_pse_key,
            "cx": settings.google_pse_cx,
            "q": query,
            "num": min(count, 10),
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return [
        {"title": i.get("title", ""), "url": i.get("link", ""), "snippet": i.get("snippet", "")}
        for i in resp.json().get("items", [])[:count] if i.get("link")
    ]


_CALLERS = {
    "duckduckgo": _duckduckgo,
    "searxng": _searxng,
    "brave": _brave,
    "tavily": _tavily,
    "serper": _serper,
    "google_pse": _google_pse,
}


def call_provider(
    provider: str,
    query: str,
    count: int = 10,
    *,
    settings: Optional[Settings] = None,
    time_filter: Optional[str] = None,
) -> List[dict]:
    """Call one named search provider once, without applying a fallback chain.

    Deep Research needs to try providers itself so it can record which one
    actually produced a result. Ordinary web search continues to use
    `web_search()` below and keeps its existing chain behavior.
    """
    settings = settings or get_settings()
    provider = (provider or "").strip().lower()
    if provider == "disabled":
        return []
    caller = _CALLERS.get(provider)
    if caller is None:
        raise ValueError(f"Unknown search provider: {provider}")
    return caller(query, max(1, count), settings, time_filter)


def provider_chain(primary: str) -> List[str]:
    """Return Athena's configured provider order for a research run."""
    primary = (primary or "").strip().lower()
    if not primary or primary == "disabled":
        return []
    return _chain(primary)


def web_search(
    query: str,
    *,
    count: Optional[int] = None,
    settings: Optional[Settings] = None,
    time_filter: Optional[str] = None,
) -> List[dict]:
    """Search the web using the configured provider, with fallback."""
    settings = settings or get_settings()
    provider = (settings.search_provider or "duckduckgo").lower()
    if provider == "disabled":
        logger.info("Web search disabled by configuration")
        return []
    count = count or settings.search_result_count

    for name in _chain(provider):
        caller = _CALLERS.get(name)
        if not caller:
            continue
        try:
            results = caller(query, count, settings, time_filter)
        except Exception as exc:
            logger.warning("Search provider %s failed for %r: %s", name, query, exc)
            continue
        if results:
            logger.info("Search provider=%s query=%r results=%d", name, query, len(results))
            return results

    logger.warning("All search providers returned no results for %r", query)
    return []
