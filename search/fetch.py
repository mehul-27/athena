"""Webpage fetching with an SSRF guard.

Adapted from Odysseus `services/search/content.py` (HTML→text heuristics) and
`src/outbound_fetch.py` (public-URL enforcement). The SSRF guard is security
critical — it is ported deliberately, not simplified away: Athena fetches URLs
that originate from search results, so private/loopback targets must be refused.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

FETCH_SOFT_MAX_BYTES = 2_000_000
FETCH_HARD_MAX_BYTES = 20_000_000


class UnsafeURLError(ValueError):
    pass


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _resolve_ips(host: str) -> List[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve host {host!r}: {exc}") from exc
    return list({info[4][0] for info in infos})


def assert_public_url(url: str) -> str:
    """Raise UnsafeURLError unless `url` is http(s) and resolves to public IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(f"Only http/https URLs are allowed (got {parsed.scheme!r})")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no host")
    ips = _resolve_ips(host)
    if not ips or any(_is_private_ip(ip) for ip in ips):
        raise UnsafeURLError(f"Refusing to fetch non-public host: {host}")
    return url


def _empty(url: str, error: str) -> Dict[str, Any]:
    return {"url": url, "title": "", "content": "", "success": False, "error": error}


def _extract_og_image(soup: BeautifulSoup, page_url: str) -> str:
    """Return the first usable Open Graph/Twitter image URL."""
    candidates = []
    for prop in ("og:image", "og:image:url", "og:image:secure_url"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content", "").strip():
            candidates.append(tag["content"].strip())
    for name in ("twitter:image", "thumbnail"):
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content", "").strip():
            candidates.append(tag["content"].strip())
    for candidate in candidates:
        image_url = urljoin(page_url, candidate)
        parsed = urlparse(image_url)
        if parsed.scheme in ("http", "https") and parsed.netloc and not image_url.lower().endswith((".svg", ".ico")):
            return image_url
    return ""


def fetch_webpage_content(
    url: str,
    *,
    settings: Settings | None = None,
    timeout: float | None = None,
    max_bytes: int | None = None,
) -> Dict[str, Any]:
    """Fetch a URL and extract readable text. Never raises on network errors."""
    settings = settings or get_settings()
    timeout = timeout or settings.web_fetch_timeout
    cap = min(max_bytes or settings.web_fetch_max_bytes or FETCH_SOFT_MAX_BYTES, FETCH_HARD_MAX_BYTES)

    try:
        assert_public_url(url)
    except UnsafeURLError as exc:
        logger.warning("Blocked fetch %s: %s", url, exc)
        return _empty(url, f"Blocked: {exc}")

    headers = {
        "User-Agent": settings.web_fetch_user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "identity",
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code >= 400:
                    return _empty(url, f"HTTP {resp.status_code}")
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body.extend(chunk)
                    if len(body) >= cap:
                        break
                declared = resp.headers.get("Content-Type", "")
        html = bytes(body).decode("utf-8", errors="replace")
    except httpx.HTTPError as exc:
        logger.warning("Fetch failed %s: %s", url, exc)
        return _empty(url, f"NetworkError: {exc}")

    if "html" not in declared.lower():
        text = html.strip()
        return {
            "url": url,
            "title": url.rsplit("/", 1)[-1] or url,
            "content": text,
            "success": bool(text),
            "error": "" if text else "Empty body",
        }

    soup = BeautifulSoup(html, "html.parser")
    for noise in soup.find_all(["script", "style", "noscript", "template", "nav", "header", "footer", "aside"]):
        noise.extract()
    title = soup.title.get_text(strip=True) if soup.title else ""
    content = " ".join(soup.get_text(separator=" ", strip=True).split())
    og_image = _extract_og_image(soup, url)
    return {
        "url": url,
        "title": title,
        "content": content,
        "og_image": og_image,
        "success": bool(content),
        "error": "",
    }
