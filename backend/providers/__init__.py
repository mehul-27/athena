"""LLM provider layer.

`LLMRouter` is the only entry point the rest of Athena uses; provider details
stay inside this package. Providers are **user-defined** and persisted by
`ProviderStore`, then served live to the router through `ProviderRegistry` —
presets in `presets.py` are defaults only and never define the supported set.
"""

from backend.providers.health import HealthTracker
from backend.providers.llm import LLMClient, LLMError
from backend.providers.presets import (
    DEFAULT_PROVIDER_ORDER,
    GOOGLE_BASE_URL,
    GROQ_BASE_URL,
    NVIDIA_BASE_URL,
    OPENROUTER_BASE_URL,
    PRESETS,
)
from backend.providers.providers import ProviderSpec, build_providers
from backend.providers.registry import ProviderRegistry
from backend.providers.router import LLMResult, LLMRouter, build_llm_router
from backend.providers.store import MODES, ProviderStore, resolve_specs

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResult",
    "LLMRouter",
    "ProviderSpec",
    "build_llm_router",
    "build_providers",
    "GROQ_BASE_URL",
    "NVIDIA_BASE_URL",
    "OPENROUTER_BASE_URL",
    "GOOGLE_BASE_URL",
    "PRESETS",
    "DEFAULT_PROVIDER_ORDER",
    "HealthTracker",
    "ProviderRegistry",
    "ProviderStore",
    "MODES",
    "resolve_specs",
]
