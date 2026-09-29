"""`ProviderSpec` — the router's generic view of one configured provider.

Everything here is OpenAI-compatible: the only variations are the base URL, the
auth style, and an optional extra body/headers dict. There is deliberately **no
per-provider branching** — a provider that exists only in the user's
`providers.json` is described by exactly the same structure as a preset one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.providers import presets
from backend.providers.presets import (  # re-exported for convenience
    GOOGLE_BASE_URL,
    GROQ_BASE_URL,
    NVIDIA_BASE_URL,
    OPENROUTER_BASE_URL,
)
from backend.providers.urls import host_label

if TYPE_CHECKING:  # avoid a config <-> providers import cycle
    from backend.config import Settings

logger = logging.getLogger(__name__)

# Env-only fallback order (used when building a router without a registry).
DEFAULT_PROVIDER_ORDER = ("groq", "nvidia", "openrouter")


@dataclass(frozen=True)
class ProviderSpec:
    name: str                                   # provider id (unique)
    label: str                                  # display name
    base_url: str
    api_key: str = ""
    model: str = ""
    enabled: bool = True
    api_key_required: bool = True
    auth_type: str = "bearer"
    provider_type: str = "openai_compatible"
    preset: str = ""
    extra_payload: Dict[str, Any] = field(default_factory=dict)
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        """True when this provider has what it needs to be called."""
        return bool(self.api_key) or not self.api_key_required

    @property
    def usable(self) -> bool:
        """A provider the router will actually try."""
        return self.enabled and self.configured

    @property
    def signature(self) -> tuple:
        """Identity of a live client — changes invalidate the cached client."""
        return (
            self.name,
            self.base_url,
            self.model,
            self.api_key,
            self.auth_type,
            tuple(sorted(self.extra_payload.items())),
            tuple(sorted(self.extra_headers.items())),
        )

    def describe(self) -> Dict[str, str]:
        return {"name": self.name, "model": self.model, "label": self.label}


def spec_from_record(settings: "Settings", store: Any, record: Dict[str, Any]) -> Optional[ProviderSpec]:
    """Build a spec from a persisted provider record (any user-defined provider)."""
    provider_id = record.get("id")
    if not provider_id:
        return None
    preset = presets.get(record.get("preset") or "")
    base_url = record.get("base_url") or (preset.base_url if preset else "")
    if not base_url:
        logger.warning("Provider %s has no endpoint URL; skipping", provider_id)
        return None

    api_key = store.api_key(provider_id) or (presets.env_key(settings, preset) if preset else "")
    model = (
        record.get("model")
        or (presets.env_model(settings, preset) if preset else "")
        or (preset.default_model if preset else "")
    )
    label = record.get("name") or (preset.label if preset else provider_id)

    return ProviderSpec(
        name=provider_id,
        label=label,
        base_url=base_url,
        api_key=api_key,
        model=model,
        enabled=bool(record.get("enabled", True)),
        api_key_required=bool(record.get("api_key_required", True)),
        auth_type=record.get("auth_type") or (preset.auth_type if preset else "bearer"),
        provider_type=record.get("provider_type") or (preset.provider_type if preset else "openai_compatible"),
        preset=record.get("preset") or "",
        extra_payload=dict(record.get("extra_body") or (preset.extra_body if preset else {})),
        extra_headers=dict(record.get("extra_headers") or (preset.extra_headers if preset else {})),
    )


def spec_from_preset(
    preset_key: str,
    *,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    name: str = "",
    enabled: bool = True,
    api_key_required: Optional[bool] = None,
    auth_type: str = "",
) -> ProviderSpec:
    """A spec straight from a preset (used for connectivity tests before saving)."""
    preset = presets.get(preset_key)
    return ProviderSpec(
        name=name or (preset.key if preset else "provider"),
        label=name or (preset.label if preset else "Provider"),
        base_url=base_url or (preset.base_url if preset else ""),
        api_key=api_key,
        model=model or (preset.default_model if preset else ""),
        enabled=enabled,
        api_key_required=preset.api_key_required if api_key_required is None and preset else bool(api_key_required),
        auth_type=auth_type or (preset.auth_type if preset else "bearer"),
        provider_type=preset.provider_type if preset else "openai_compatible",
        preset=preset.key if preset else "",
        extra_payload=dict(preset.extra_body) if preset else {},
        extra_headers=dict(preset.extra_headers) if preset else {},
    )


def build_provider(name: str, settings: "Settings") -> Optional[ProviderSpec]:
    """Env-only bootstrap spec for one legacy provider (no store involved)."""
    preset = presets.get(name)
    if preset is None or name not in presets.BOOTSTRAP_PRESET_KEYS:
        return None
    return spec_from_preset(
        preset.key,
        api_key=presets.env_key(settings, preset),
        model=presets.env_model(settings, preset),
    )


def build_providers(settings: "Settings") -> List[ProviderSpec]:
    """Legacy `.env`-only provider chain (used when no registry is supplied)."""
    raw = settings.athena_llm_providers or ""
    order = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not order:
        order = list(DEFAULT_PROVIDER_ORDER)

    specs: List[ProviderSpec] = []
    seen = set()
    for name in order:
        spec = build_provider(name, settings)
        if spec is None:
            logger.warning("Ignoring unknown LLM provider %r in ATHENA_LLM_PROVIDERS", name)
            continue
        if spec.name in seen:
            continue
        seen.add(spec.name)
        specs.append(spec)
    return specs


def display_label(name: str, base_url: str) -> str:
    """A display name for a provider that has none (host-derived)."""
    return host_label(base_url) or name
