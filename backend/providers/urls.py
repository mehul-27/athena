"""URL helpers for arbitrary OpenAI-compatible endpoints.

Adapted (own implementation) from Odysseus's endpoint normalisation behaviour in
`src/endpoint_resolver.py`: users paste whatever the provider's docs show, so we
normalise it before appending paths, and we never assume a provider identity to
decide the URLs.
"""

from __future__ import annotations

import ipaddress
import logging
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

# Suffixes a user may paste that are already a concrete API path, not a base.
_PATH_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/v1/messages",
    "/responses",
    "/models",
    "/embeddings",
)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}

# Friendly labels for well-known hosts (display only — never used to branch logic).
_HOST_LABELS = (
    ("api.openai.com", "OpenAI"),
    ("api.anthropic.com", "Anthropic"),
    ("api.deepseek.com", "DeepSeek"),
    ("api.mistral.ai", "Mistral"),
    ("api.groq.com", "Groq"),
    ("api.cerebras.ai", "Cerebras"),
    ("api.together.xyz", "Together"),
    ("api.fireworks.ai", "Fireworks"),
    ("openrouter.ai", "OpenRouter"),
    ("integrate.api.nvidia.com", "NVIDIA"),
    ("generativelanguage.googleapis.com", "Google"),
    ("api.x.ai", "xAI"),
    ("api.perplexity.ai", "Perplexity"),
)


def normalize_base(url: str) -> str:
    """Strip a pasted endpoint path back to a base URL.

    ``https://api.openai.com/v1/chat/completions`` -> ``https://api.openai.com/v1``
    """
    base = (url or "").strip().rstrip("/")
    if not base:
        return ""
    lowered = base.lower()
    for suffix in _PATH_SUFFIXES:
        if lowered.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    return base


def validate_base(url: str) -> str:
    """Normalise and reject grossly invalid bases. Raises ValueError."""
    base = normalize_base(url)
    if not base:
        raise ValueError("Endpoint URL is required")
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Endpoint URL must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("Endpoint URL must include a host")
    if parsed.query or parsed.fragment:
        raise ValueError("Endpoint URL must not include a query string or fragment")
    return urlunparse(parsed._replace(query="", fragment="")).rstrip("/")


def is_local_url(url: str) -> bool:
    """True for loopback / private-LAN endpoints (a local model server)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host or host in _LOOPBACK_HOSTS:
        return bool(host)
    if host.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host  # bare hostname like "mybox"
    return addr.is_private or addr.is_loopback or addr.is_link_local


def host_label(url: str) -> str:
    """A friendly provider label derived from the endpoint host (display only)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    if is_local_url(url):
        return "Local"
    for known, label in _HOST_LABELS:
        if host == known or host.endswith("." + known):
            return label
    return host.replace("api.", "", 1) if host.startswith("api.") else host


def chat_url(base_url: str) -> str:
    """The OpenAI-compatible chat completions URL for a base."""
    base = normalize_base(base_url)
    if not base:
        return ""
    return f"{base}/chat/completions"


def models_url(base_url: str) -> str:
    """The model-list URL for a base.

    Local servers are conventionally reachable at a bare host (e.g.
    ``http://localhost:1234`` for LM Studio) but expose ``/v1/models``; insert
    the ``/v1`` segment only when the base has no path at all, so explicit
    prefixes (``/openai``, ``/api/v1``) are preserved.
    """
    base = normalize_base(base_url)
    if not base:
        return ""
    parsed = urlparse(base)
    if not (parsed.path or "").strip("/") and is_local_url(base):
        base = f"{base}/v1"
    return f"{base}/models"


def auth_headers(api_key: str, auth_type: str = "bearer") -> dict:
    """Auth headers for the configured style. No key -> no auth header."""
    if not api_key:
        return {}
    auth_type = (auth_type or "bearer").strip().lower()
    if auth_type in ("none", ""):
        return {}
    if auth_type in ("header", "x-api-key", "api-key", "anthropic"):
        headers = {"x-api-key": api_key}
        if auth_type == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        return headers
    return {"Authorization": f"Bearer {api_key}"}
