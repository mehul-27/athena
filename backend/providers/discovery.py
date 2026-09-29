"""Generic provider probing: model discovery and a live connectivity test.

Both work against *any* OpenAI-compatible endpoint — there is no provider table
here. Discovery tries `GET {base}/models`; when a provider doesn't implement it,
the UI falls back to manual model entry. The connectivity test sends one minimal
completion so a provider is only ever reported "available" if it really answered.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from backend.providers.llm import LLMClient, LLMError
from backend.providers.providers import ProviderSpec
from backend.providers.urls import auth_headers, models_url

logger = logging.getLogger(__name__)

DISCOVERY_TIMEOUT = 15.0
PROBE_TIMEOUT = 25.0
MAX_DISCOVERED_MODELS = 400

# Human-readable outcomes for the Settings "Test" button.
AVAILABLE = "available"
RATE_LIMITED = "rate_limited"
AUTH_FAILED = "auth_failed"
ENDPOINT_UNAVAILABLE = "endpoint_unavailable"
MODEL_UNAVAILABLE = "model_unavailable"
TIMEOUT = "timeout"
ERROR = "error"


def _headers(spec: ProviderSpec) -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(spec.api_key, spec.auth_type))
    headers.update(spec.extra_headers)
    return headers


def fetch_models(spec: ProviderSpec, *, timeout: float = DISCOVERY_TIMEOUT) -> List[str]:
    """Return the model ids the endpoint advertises, or [] when unsupported.

    Accepts both `{"data": [...]}` and a bare JSON list, which is what local
    servers (LM Studio, llama.cpp, vLLM) variously return.
    """
    url = models_url(spec.base_url)
    if not url:
        return []
    try:
        resp = httpx.get(url, headers=_headers(spec), timeout=timeout)
    except httpx.HTTPError as exc:
        logger.warning("Model discovery failed for %s (%s): %s", spec.name, url, type(exc).__name__)
        return []
    if resp.status_code >= 400:
        logger.warning("Model discovery for %s (%s) returned HTTP %s", spec.name, url, resp.status_code)
        return []
    try:
        payload = resp.json()
    except ValueError:
        logger.warning("Model discovery for %s returned a non-JSON body", spec.name)
        return []

    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    ids: List[str] = []
    for item in items:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("name")
        else:
            model_id = item
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id.strip())
    seen = set()
    unique = [m for m in ids if not (m in seen or seen.add(m))]
    return unique[:MAX_DISCOVERED_MODELS]


def _classify_http(status: int) -> str:
    if status in (401, 403):
        return AUTH_FAILED
    if status == 404:
        return MODEL_UNAVAILABLE
    if status == 429:
        return RATE_LIMITED
    if status >= 500:
        return ENDPOINT_UNAVAILABLE
    return ERROR


def probe(
    spec: ProviderSpec,
    *,
    model: Optional[str] = None,
    timeout: float = PROBE_TIMEOUT,
) -> Dict[str, Any]:
    """Send one minimal completion to prove the endpoint really works.

    One attempt, no retries: this is a user-triggered check, so it should fail
    fast instead of hiding behind the router's transient-error retry budget.
    """
    target = spec
    if model and model != spec.model:
        target = ProviderSpec(
            name=spec.name,
            label=spec.label,
            base_url=spec.base_url,
            api_key=spec.api_key,
            model=model,
            enabled=spec.enabled,
            api_key_required=spec.api_key_required,
            auth_type=spec.auth_type,
            provider_type=spec.provider_type,
            preset=spec.preset,
            extra_payload=dict(spec.extra_payload),
            extra_headers=dict(spec.extra_headers),
        )

    if not target.model:
        return {"ok": False, "status": ERROR, "label": "No model configured",
                "model": "", "error": "Set a model id (or run model discovery) first."}

    client = LLMClient(
        base_url=target.base_url,
        api_key=target.api_key,
        model=target.model,
        provider=target.name,
        auth_type=target.auth_type,
        temperature=0.0,
        max_tokens=16,
        timeout=timeout,
        max_attempts=1,
        retry_base_delay=0.0,
        extra_payload=dict(target.extra_payload),
        extra_headers=dict(target.extra_headers),
    )
    try:
        content = client.chat([{"role": "user", "content": "Reply with the single word: OK"}])
    except LLMError as exc:
        status = exc.status
        kind = _classify_http(status) if status else ERROR
        if status is None and "timed out" in str(exc).lower():
            kind = TIMEOUT
        return {
            "ok": False,
            "status": kind,
            "http_status": status,
            "model": target.model,
            "error": str(exc)[:300],
        }
    except httpx.TimeoutException as exc:  # pragma: no cover - defensive
        return {"ok": False, "status": TIMEOUT, "http_status": None, "model": target.model,
                "error": str(exc)[:300]}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "status": ERROR, "http_status": None, "model": target.model,
                "error": str(exc)[:300]}
    return {"ok": True, "status": AVAILABLE, "http_status": 200, "model": target.model,
            "reply": content[:80]}
