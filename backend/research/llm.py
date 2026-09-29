"""Async adapter from the research engine to Athena's shared LLM router.

The engine only distinguishes two kinds of work:

* ``role="fast"``  — control calls (plan, category, query generation, page
  extraction, stop/continue decisions). Frequent, and the cheapest adequate
  model is fine.
* ``role="strong"`` — the report itself (round synthesis and the final report).

Each role maps to a configured model id. Both default to blank, which means
"whatever model the active provider is configured with" — i.e. today's
behaviour. Set `RESEARCH_FAST_MODEL` / `RESEARCH_STRONG_MODEL` (or the Settings
UI) to split them.

The adapter also counts calls and records which provider answered each one, so a
research run can report its LLM usage and provider mix.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from backend.providers.router import LLMRouter
from backend.research.text import strip_thinking

logger = logging.getLogger(__name__)


class ResearchLLM:
    def __init__(
        self,
        router: LLMRouter,
        *,
        fast_model: Optional[str] = None,
        strong_model: Optional[str] = None,
        prefer_provider: Optional[str] = None,
        mode_model: Optional[str] = None,
        usage_sink: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self.router = router
        self.fast_model = (fast_model or "").strip() or None
        self.strong_model = (strong_model or "").strip() or None
        # Model chosen for the Research tab in the UI (blank = provider default).
        self.prefer_provider = (prefer_provider or "").strip().lower() or None
        self.mode_model = (mode_model or "").strip() or None
        # Called with every LLMResult so a run's many calls can be attributed to
        # the conversation it was launched from.
        self.usage_sink = usage_sink
        self.last_provider: Optional[str] = None
        self.last_model: Optional[str] = None
        self.llm_calls = 0
        self.providers: List[str] = []

    async def call(
        self,
        messages: List[Dict[str, Any]],
        *,
        role: str = "fast",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> str:
        # Role model (FAST/STRONG) wins over the per-mode selection; the
        # preferred provider always routes first (with fallback behind it).
        role_model = self.strong_model if role == "strong" else self.fast_model
        model = role_model or self.mode_model
        kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if model:
            kwargs["model"] = model
        if self.prefer_provider:
            kwargs["prefer"] = self.prefer_provider

        result = await asyncio.to_thread(self.router.chat, messages, **kwargs)
        self.llm_calls += 1
        self.last_provider = result.provider
        self.last_model = result.model
        if result.provider:
            self.providers.append(result.provider)
        if self.usage_sink is not None:
            # Bookkeeping must never break a research run.
            try:
                await asyncio.to_thread(self.usage_sink, result)
            except Exception:  # pragma: no cover - defensive
                logger.warning("Could not record research LLM usage", exc_info=True)
        return strip_thinking(result.content)
