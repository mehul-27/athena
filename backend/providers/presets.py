"""Provider presets.

Presets are **defaults only** — convenience values for the name, endpoint, auth
style and discovery method that the Add API Model form pre-fills. They do NOT
constrain which providers Athena can use: a provider that appears nowhere in this
file can be added entirely from the UI (type it into the form), and any preset
field can be edited afterwards.

This replaces the old fixed `catalog.py`, where the provider list *was* the
source of truth and adding a provider required a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

from backend.providers.urls import is_local_url

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    label: str
    base_url: str = ""
    default_model: str = ""
    provider_type: str = "cloud"          # cloud | openai_compatible | local
    auth_type: str = "bearer"             # bearer | header | anthropic | none
    api_key_required: bool = True
    supports_model_discovery: bool = True
    docs_url: str = ""
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    # Bootstrap-only: Settings attribute names read from .env on a fresh install.
    env_key_fields: Tuple[str, ...] = ()
    env_model_field: str = ""
    # Hosts used to auto-detect the preset from a pasted endpoint (UI convenience).
    hosts: Tuple[str, ...] = ()


def _presets() -> Dict[str, ProviderPreset]:
    entries = [
        ProviderPreset(
            key="nvidia", label="NVIDIA NIM", base_url=NVIDIA_BASE_URL,
            default_model="nvidia/nemotron-3.5-lightning-30b-a3b",
            env_key_fields=("nvidia_api_key",), env_model_field="athena_nvidia_model",
            docs_url="https://build.nvidia.com", hosts=("integrate.api.nvidia.com",),
        ),
        ProviderPreset(
            key="groq", label="Groq", base_url=GROQ_BASE_URL,
            default_model="qwen/qwen3.8-27b",
            extra_body=MappingProxyType({"reasoning_format": "hidden"}),
            env_key_fields=("groq_api_key",), env_model_field="athena_groq_model",
            docs_url="https://console.groq.com/keys", hosts=("api.groq.com",),
        ),
        ProviderPreset(
            key="openrouter", label="OpenRouter", base_url=OPENROUTER_BASE_URL,
            default_model="nvidia/nemotron-3.5-lightning:free",
            env_key_fields=("openrouter_api_key", "llm_api_key"),
            env_model_field="athena_openrouter_model",
            docs_url="https://openrouter.ai/keys", hosts=("openrouter.ai",),
            # Attribution headers OpenRouter uses for rankings; harmless elsewhere.
            extra_headers=MappingProxyType({"HTTP-Referer": "http://localhost", "X-Title": "Athena"}),
        ),
        ProviderPreset(
            key="google", label="Google Gemini", base_url=GOOGLE_BASE_URL,
            default_model="gemini-3.6-flash",
            env_key_fields=("google_api_key", "gemini_api_key"),
            env_model_field="athena_google_model",
            docs_url="https://aistudio.google.com/app/apikey",
            hosts=("generativelanguage.googleapis.com",),
        ),
        ProviderPreset(
            key="openai", label="OpenAI", base_url="https://api.openai.com/v1",
            default_model="gpt-4o-mini", docs_url="https://platform.openai.com/api-keys",
            hosts=("api.openai.com",),
        ),
        ProviderPreset(
            key="anthropic", label="Anthropic", base_url="https://api.anthropic.com/v1",
            default_model="claude-sonnet-4-5", auth_type="anthropic",
            docs_url="https://console.anthropic.com/settings/keys",
            hosts=("api.anthropic.com",),
        ),
        ProviderPreset(
            key="deepseek", label="DeepSeek", base_url="https://api.deepseek.com/v1",
            default_model="deepseek-chat", docs_url="https://platform.deepseek.com/api_keys",
            hosts=("api.deepseek.com",),
        ),
        ProviderPreset(
            key="mistral", label="Mistral", base_url="https://api.mistral.ai/v1",
            default_model="mistral-large-latest", docs_url="https://console.mistral.ai/api-keys",
            hosts=("api.mistral.ai",),
        ),
        ProviderPreset(
            key="together", label="Together", base_url="https://api.together.xyz/v1",
            default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            docs_url="https://api.together.xyz/settings/api-keys", hosts=("api.together.xyz",),
        ),
        ProviderPreset(
            key="fireworks", label="Fireworks", base_url="https://api.fireworks.ai/inference/v1",
            default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            docs_url="https://fireworks.ai/account/api-keys", hosts=("api.fireworks.ai",),
        ),
        ProviderPreset(
            key="cerebras", label="Cerebras", base_url="https://api.cerebras.ai/v1",
            default_model="llama3.1-8b", docs_url="https://cloud.cerebras.ai",
            hosts=("api.cerebras.ai",),
        ),
        ProviderPreset(
            key="xai", label="xAI", base_url="https://api.x.ai/v1",
            default_model="grok-2-latest", docs_url="https://console.x.ai", hosts=("api.x.ai",),
        ),
        # A gateway that fronts many vendors (and exposes them OpenAI-compatibly,
        # so discovery lists e.g. openai/…, anthropic/…, google/… model ids).
        ProviderPreset(
            key="vercel", label="Vercel AI Gateway",
            base_url="https://ai-gateway.vercel.sh/v1",
            default_model="openai/gpt-4o-mini",
            docs_url="https://vercel.com/docs/ai-gateway",
            hosts=("ai-gateway.vercel.sh",),
        ),
        # --- Local servers: no API key by default -------------------------
        ProviderPreset(
            key="ollama", label="Ollama (local)", base_url="http://localhost:11434/v1",
            default_model="llama3.2", provider_type="local", auth_type="none",
            api_key_required=False, docs_url="https://ollama.com",
            hosts=("localhost",),
        ),
        ProviderPreset(
            key="lmstudio", label="LM Studio (local)", base_url="http://localhost:1234/v1",
            provider_type="local", auth_type="none", api_key_required=False,
            docs_url="https://lmstudio.ai", hosts=("localhost",),
        ),
        ProviderPreset(
            key="vllm", label="vLLM (local)", base_url="http://localhost:8000/v1",
            provider_type="local", auth_type="none", api_key_required=False,
            docs_url="https://docs.vllm.ai",
        ),
        ProviderPreset(
            key="llamacpp", label="llama.cpp (local)", base_url="http://localhost:8080/v1",
            provider_type="local", auth_type="none", api_key_required=False,
            docs_url="https://github.com/ggerganov/llama.cpp",
        ),
        # --- Always-available escape hatch --------------------------------
        ProviderPreset(
            key="custom", label="Custom / OpenAI-compatible",
            provider_type="openai_compatible",
            docs_url="https://platform.openai.com/docs/api-reference",
        ),
    ]
    return {preset.key: preset for preset in entries}


PRESETS: Dict[str, ProviderPreset] = _presets()

CUSTOM_PRESET_KEY = "custom"

# The presets bootstrapped from .env on a fresh install (backwards compatibility).
BOOTSTRAP_PRESET_KEYS: Tuple[str, ...] = ("nvidia", "groq", "openrouter", "google")

# Default priority for a brand-new store that has no .env keys at all.
DEFAULT_PROVIDER_ORDER: Tuple[str, ...] = BOOTSTRAP_PRESET_KEYS


def get(key: str) -> Optional[ProviderPreset]:
    return PRESETS.get((key or "").strip().lower())


def is_known(key: str) -> bool:
    return (key or "").strip().lower() in PRESETS


def find_by_url(base_url: str) -> Optional[ProviderPreset]:
    """Best-effort preset match for a pasted endpoint (UI hint only)."""
    if not base_url:
        return None
    host = ""
    try:
        from urllib.parse import urlparse

        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    for preset in PRESETS.values():
        if any(host == h or host.endswith("." + h) for h in preset.hosts):
            return preset
    if is_local_url(base_url):
        return PRESETS.get("custom")
    return None


def as_dict(preset: ProviderPreset) -> Dict[str, Any]:
    """Serialisable preset for the Settings UI."""
    return {
        "key": preset.key,
        "label": preset.label,
        "base_url": preset.base_url,
        "default_model": preset.default_model,
        "provider_type": preset.provider_type,
        "auth_type": preset.auth_type,
        "api_key_required": preset.api_key_required,
        "supports_model_discovery": preset.supports_model_discovery,
        "docs_url": preset.docs_url,
        "extra_body": dict(preset.extra_body),
        "extra_headers": dict(preset.extra_headers),
        "is_local": preset.provider_type == "local",
    }


def env_key(settings: Any, preset: ProviderPreset) -> str:
    for field_name in preset.env_key_fields:
        value = getattr(settings, field_name, "") or ""
        if value:
            return value
    return ""


def env_model(settings: Any, preset: ProviderPreset) -> str:
    if not preset.env_model_field:
        return ""
    return getattr(settings, preset.env_model_field, "") or ""


def all_presets() -> List[ProviderPreset]:
    """Presets in a stable, display-friendly order (custom last)."""
    ordered = [p for k, p in PRESETS.items() if k != CUSTOM_PRESET_KEY]
    ordered.append(PRESETS[CUSTOM_PRESET_KEY])
    return ordered
