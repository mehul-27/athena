"""LLM router: ordered provider failover.

Callers — normal chat, RAG chat, search and Deep Research — only ever call
`router.chat(messages)`. Which vendor answers, and whether it took a retry or a
fallback to get there, is entirely this layer's problem.

The provider list can be *live*: when constructed with a `loader` (the
`ProviderRegistry`), the router re-reads the configured chain on every request,
so a provider/model change made in Settings applies to the next call without a
restart. Passing a plain sequence (as the unit tests do) keeps it static.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from backend.providers.health import HealthTracker
from backend.providers.llm import LLMClient, LLMError, MAX_ATTEMPTS, RETRY_BASE_DELAY
from backend.providers.pricing import estimate_cost
from backend.providers.providers import ProviderSpec, build_providers
from backend.providers.urls import is_local_url

if TYPE_CHECKING:  # avoid a config <-> providers import cycle
    from backend.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    """An answer plus the provider that actually produced it — and its usage."""

    content: str
    provider: str
    model: str
    label: str
    # Set when the caller asked for a specific provider (per-tab model choice)
    # but a different one answered — so the UI can say so instead of silently
    # returning another model's answer.
    requested_provider: Optional[str] = None
    fallback: bool = False
    # Real usage for this call. Tokens come from the provider; `cost_usd` is
    # None when we have no price for the model (never a guess).
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    cost_known: bool = False
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def usage_summary(self) -> Dict[str, Any]:
        """The usage record for this call (provider/model kept per request)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "cost_known": self.cost_known,
            "latency_ms": self.latency_ms,
        }

    def __str__(self) -> str:  # convenient when logged or interpolated
        return self.content


class LLMRouter:
    def __init__(
        self,
        providers: Sequence[ProviderSpec],
        *,
        loader: Optional[Callable[[], Sequence[ProviderSpec]]] = None,
        health: Optional[HealthTracker] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
        max_attempts: int = MAX_ATTEMPTS,
        retry_base_delay: float = RETRY_BASE_DELAY,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.health = health or HealthTracker()

        self._source: List[ProviderSpec] = list(providers)
        self._loader = loader
        self._clients: Dict[tuple, LLMClient] = {}
        # Monotonic counters; Deep Research snapshots these to attribute
        # fallbacks to a single run.
        self.stats: Dict[str, int] = {"requests": 0, "fallbacks": 0}

        self._log_skips(*self._resolve())

    # ------------------------------------------------------------------
    # provider resolution
    # ------------------------------------------------------------------
    def _resolve(self) -> tuple[List[ProviderSpec], List[str]]:
        specs = list(self._loader()) if self._loader else list(self._source)
        active: List[ProviderSpec] = []
        skipped: List[str] = []
        for spec in specs:
            if not spec.enabled:
                skipped.append(spec.name)
            elif spec.api_key_required and not spec.api_key:
                skipped.append(spec.name)
            else:
                active.append(spec)
        return active, skipped

    def _log_skips(self, active: List[ProviderSpec], skipped: List[str]) -> None:
        for spec in (self._loader() if self._loader else self._source):
            if not spec.enabled:
                logger.info("LLM provider=%s skipped (disabled in Settings)", spec.name)
            elif spec.api_key_required and not spec.api_key:
                logger.info("LLM provider=%s skipped (no API key configured)", spec.name)

    def _order_for_attempt(self, providers: List[ProviderSpec]) -> List[ProviderSpec]:
        """Try providers that are *not* currently rate-limited first.

        A provider that just returned 429 stays in cooldown for a short window
        (see `HealthTracker`). During that window a request would otherwise pay
        its retry delay again on every call — which, on a research run issuing
        dozens of calls, makes an exhausted free provider feel like a hang. The
        cooling provider is still appended, so a single-provider install never
        loses its only option.
        """
        if not providers:
            return providers
        healthy: List[ProviderSpec] = []
        cooling: List[ProviderSpec] = []
        for spec in providers:
            (cooling if self.health.get(spec.name).rate_limited() else healthy).append(spec)
        if not healthy or not cooling:
            return list(providers)
        logger.info(
            "LLM routing order adjusted: rate-limited %s deferred behind %s",
            ",".join(s.name for s in cooling),
            ",".join(s.name for s in healthy),
        )
        return healthy + cooling

    def _with_preferred(self, providers: List[ProviderSpec], prefer: Optional[str]) -> List[ProviderSpec]:
        """Move an explicitly chosen provider to the front of the attempt order.

        Used by the per-mode model selector (Chat / RAG / Search / Research): the
        user's explicit pick is tried first, but the rest of the chain stays
        behind it so a rate-limited or failing choice still falls back.
        """
        if not prefer:
            return providers
        chosen = [spec for spec in providers if spec.name == prefer]
        if not chosen:
            return providers
        others = [spec for spec in providers if spec.name != prefer]
        return chosen + others

    @property
    def configured(self) -> bool:
        return bool(self._resolve()[0])

    @property
    def provider_names(self) -> List[str]:
        return [spec.name for spec in self._resolve()[0]]

    @property
    def skipped_names(self) -> List[str]:
        return list(self._resolve()[1])

    def describe(self) -> List[Dict[str, str]]:
        return [
            {"name": spec.name, "model": spec.model, "label": spec.label}
            for spec in self._resolve()[0]
        ]

    def _client(self, spec: ProviderSpec) -> LLMClient:
        key = spec.signature
        client = self._clients.get(key)
        if client is None:
            client = LLMClient(
                base_url=spec.base_url,
                api_key=spec.api_key,
                model=spec.model,
                provider=spec.name,
                auth_type=spec.auth_type,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_attempts=self.max_attempts,
                retry_base_delay=self.retry_base_delay,
                extra_payload=spec.extra_payload,
                extra_headers=spec.extra_headers,
            )
            self._clients[key] = client
        return client

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        prefer: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        providers, _ = self._resolve()
        if not providers:
            raise LLMError(
                "No LLM provider is configured — add one in Athena Settings "
                "(Add Models → Add API Model)."
            )

        self.stats["requests"] += 1
        providers = self._order_for_attempt(providers)
        prefer = (prefer or "").strip().lower() or None
        providers = self._with_preferred(providers, prefer)
        logger.info(
            "LLM request providers=%s model=%s%s",
            ",".join(spec.name for spec in providers),
            model or providers[0].model,
            f" prefer={prefer}" if prefer else "",
        )

        failures: List[str] = []
        for index, spec in enumerate(providers):
            # `model` alone overrides the model for every provider (global
            # override). With `prefer`, it applies only to the chosen provider so
            # the fallbacks still use their own configured models.
            if model and (not prefer or spec.name == prefer):
                use_model = model
            else:
                use_model = spec.model
            try:
                outcome = self._client(spec).chat_with_usage(
                    messages,
                    model=use_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            except LLMError as exc:
                status = exc.status if exc.status is not None else "n/a"
                failures.append(f"{spec.name} ({status}): {exc}")
                self.health.record_failure(spec.name, status=exc.status, message=str(exc))
                nxt = providers[index + 1].name if index + 1 < len(providers) else None
                if nxt:
                    self.stats["fallbacks"] += 1
                    logger.warning(
                        "LLM provider=%s failed status=%s; falling back to %s",
                        spec.name, status, nxt,
                    )
                else:
                    logger.error(
                        "LLM provider=%s failed status=%s; no further providers",
                        spec.name, status,
                    )
                continue

            self.health.record_success(spec.name, model=use_model)
            fallback = bool(prefer) and spec.name != prefer
            if fallback:
                logger.warning(
                    "LLM provider=%s was requested but %s answered (fallback)",
                    prefer, spec.name,
                )
            # Cost is attributed to the model that actually answered, using the
            # provider-reported token counts. Local endpoints are never billed.
            is_local = spec.provider_type == "local" or is_local_url(spec.base_url)
            cost, cost_known = estimate_cost(
                use_model or spec.model,
                outcome.input_tokens,
                outcome.output_tokens,
                is_local=is_local,
            )
            logger.info(
                "LLM success provider=%s model=%s tokens=%d/%d latency_ms=%d cost=%s",
                spec.name, use_model, outcome.input_tokens, outcome.output_tokens,
                outcome.latency_ms, "unknown" if cost is None else f"{cost:.6f}",
            )
            return LLMResult(
                content=outcome.content,
                provider=spec.name,
                # Report the model actually sent, not just the provider's
                # configured default — a per-tab/per-call override must show up
                # here or the UI would attribute the answer to the wrong model.
                model=use_model,
                label=spec.label,
                requested_provider=prefer,
                fallback=fallback,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                cost_usd=cost,
                cost_known=cost_known,
                latency_ms=outcome.latency_ms,
            )

        raise LLMError("All configured LLM providers failed — " + "; ".join(failures))


def build_llm_router(settings: "Settings", registry: Optional[Any] = None) -> LLMRouter:
    """Build the router from env, or from a live registry when one is supplied."""
    if registry is not None:
        return LLMRouter(
            registry.specs(),
            loader=registry.specs,
            health=registry.health,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens or None,
            timeout=settings.llm_timeout,
        )
    return LLMRouter(
        build_providers(settings),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens or None,
        timeout=settings.llm_timeout,
    )
