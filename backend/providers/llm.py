"""Low-level OpenAI-compatible chat client — one instance per provider endpoint.

Owns the HTTP call, that provider's payload/header tweaks, and the retry policy
for transient failures. Which provider gets used, and what happens when one
fails, is `backend/providers/router.py`'s job.

Deliberately tiny. Odysseus's `src/llm_core.py` (~4000 LOC) bundles Ollama-native
handling, Anthropic payload shaping, response caching, a local model gate and
Kimi UA juggling; copying it would re-import exactly the entanglement Athena is
migrating away from (see MIGRATION_AUDIT.md §6).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from backend.providers.urls import auth_headers
from backend.text_utils import strip_thinking

logger = logging.getLogger(__name__)

# Retry policy, applied per provider.
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
# 429 = rate limited; 5xx = transient provider errors. Other 4xx (auth, config,
# malformed request) are permanent — retrying them wastes quota.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """A failed LLM call. `status` carries the HTTP status when there was one."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class CallOutcome:
    """One successful call: the answer, the provider-reported token usage, latency."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_usage(self) -> bool:
        return bool(self.input_tokens or self.output_tokens)


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        *,
        provider: str = "openai",
        auth_type: str = "bearer",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
        max_attempts: int = MAX_ATTEMPTS,
        retry_base_delay: float = RETRY_BASE_DELAY,
        extra_payload: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = (provider or "openai").lower()
        self.auth_type = (auth_type or "bearer").lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.extra_payload = dict(extra_payload or {})
        self.extra_headers = dict(extra_headers or {})

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        # Auth style comes from the provider's config (bearer / x-api-key /
        # anthropic / none) — an endpoint with no key simply sends none.
        headers.update(auth_headers(self.api_key, self.auth_type))
        headers.update(self.extra_headers)
        return headers

    def _scrub(self, text: str) -> str:
        """Keep credentials out of logs and exceptions, whatever upstream echoes."""
        if self.api_key and self.api_key in text:
            text = text.replace(self.api_key, "***")
        return text

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Send a chat completion and return the assistant's final answer text."""
        return self.chat_with_usage(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, timeout=timeout,
        ).content

    def chat_with_usage(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> CallOutcome:
        """Same as `chat`, but also report provider usage (tokens) and latency."""
        url = f"{self.base_url}/chat/completions"
        model_name = model or self.model
        request_timeout = self.timeout if timeout is None else timeout

        payload: Dict[str, Any] = {"model": model_name, "messages": messages}
        temp = self.temperature if temperature is None else temperature
        if temp is not None:
            payload["temperature"] = temp
        mt = self.max_tokens if max_tokens is None else max_tokens
        if mt:
            payload["max_tokens"] = mt
        payload.update(self.extra_payload)

        headers = self._headers()

        for attempt in range(1, self.max_attempts + 1):
            logger.info(
                "LLM provider=%s model=%s attempt=%d/%d",
                self.provider, model_name, attempt, self.max_attempts,
            )
            started = time.perf_counter()
            try:
                resp = httpx.post(url, json=payload, headers=headers, timeout=request_timeout)
            except httpx.HTTPError as exc:
                # Connection resets and timeouts are transient — retry them.
                if attempt < self.max_attempts:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM provider=%s model=%s attempt=%d/%d error=%s retrying_in=%.1fs",
                        self.provider, model_name, attempt, self.max_attempts,
                        type(exc).__name__, delay,
                    )
                    time.sleep(delay)
                    continue
                logger.error(
                    "LLM provider=%s model=%s failed attempts=%d error=%s",
                    self.provider, model_name, attempt, type(exc).__name__,
                )
                raise LLMError(f"LLM request failed: {self._scrub(str(exc))}") from exc

            if resp.status_code < 400:
                latency_ms = int((time.perf_counter() - started) * 1000)
                content = self._parse_response(resp, model_name)
                input_tokens, output_tokens = self._extract_usage(resp)
                return CallOutcome(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                )

            if resp.status_code in RETRYABLE_STATUS and attempt < self.max_attempts:
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                reason = "rate limited" if resp.status_code == 429 else "transient error"
                logger.warning(
                    "LLM provider=%s model=%s attempt=%d/%d status=%s %s retrying_in=%.1fs",
                    self.provider, model_name, attempt, self.max_attempts,
                    resp.status_code, reason, delay,
                )
                time.sleep(delay)
                continue

            logger.error(
                "LLM provider=%s model=%s failed attempts=%d status=%s",
                self.provider, model_name, attempt, resp.status_code,
            )
            raise LLMError(
                f"LLM returned HTTP {resp.status_code}: {self._scrub(self._error_snippet(resp))}",
                status=resp.status_code,
            )

        # Not reachable: the loop either returns a response or raises.
        raise LLMError("LLM request failed: no attempt was made")

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_usage(resp: Any) -> tuple[int, int]:
        """Read provider-reported token usage, or (0, 0) when it isn't supplied.

        Providers that don't return `usage` simply contribute no tokens — we
        never estimate them, so the numbers on screen are real.
        """
        try:
            data = resp.json()
        except (ValueError, AttributeError):
            return 0, 0
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            return 0, 0
        raw_in = usage.get("prompt_tokens", usage.get("input_tokens"))
        raw_out = usage.get("completion_tokens", usage.get("output_tokens"))
        try:
            input_tokens = max(int(raw_in or 0), 0)
            output_tokens = max(int(raw_out or 0), 0)
        except (TypeError, ValueError):
            return 0, 0
        return input_tokens, output_tokens

    # ------------------------------------------------------------------
    @staticmethod
    def _error_snippet(resp: Any) -> str:
        return (getattr(resp, "text", "") or "")[:500]

    @staticmethod
    def _parse_response(resp: Any, model_name: str) -> str:
        """Return the final answer only — never a reasoning trace."""
        try:
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"Malformed LLM response: {exc}") from exc

        if isinstance(content, list):
            # Some providers return content as a list of parts.
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

        raw = (content or "").strip()
        if not raw:
            # Reasoning-capable models can return content=None with the trace in
            # `reasoning`. That is not an answer — fail so the router moves on
            # rather than showing the user raw reasoning.
            raise LLMError(f"LLM returned no answer content (model={model_name})")

        # Some models (e.g. Gemma/Gemini) inline their reasoning in the content
        # as <thought>…</thought> or channel markup. Strip it here so no caller
        # can accidentally surface a reasoning trace as the answer.
        text = strip_thinking(raw)
        if not text:
            raise LLMError(f"LLM returned only reasoning, no answer (model={model_name})")
        return text
