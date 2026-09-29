"""Provider registry — the live, mutable view the router and Settings share.

One instance lives on `app.state`. It owns the persistent store and the runtime
health tracker, and it is the single place the Settings API mutates provider
configuration. There is no fixed provider list: `create()` accepts anything the
user types, and presets only supply default field values.

`LLMRouter` is constructed with `registry.specs` as its loader, so a change made
in Settings applies to the **next** request with no restart and no `.env` edit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.providers import presets
from backend.providers.discovery import fetch_models, probe
from backend.providers.health import HealthTracker
from backend.providers.providers import ProviderSpec, spec_from_preset, spec_from_record
from backend.providers.store import MODES, ProviderStore, provider_statuses, resolve_specs
from backend.providers.urls import host_label, validate_base

if TYPE_CHECKING:  # avoid a config <-> providers import cycle
    from backend.config import Settings

logger = logging.getLogger(__name__)


class ProviderRegistry:
    def __init__(
        self,
        settings: "Settings",
        *,
        store: Optional[ProviderStore] = None,
        health: Optional[HealthTracker] = None,
    ) -> None:
        self.settings = settings
        self.store = store or ProviderStore(settings)
        self.health = health or HealthTracker()
        self._specs: List[ProviderSpec] = []
        self._token: Optional[tuple] = None
        self.store.ensure_bootstrapped()
        self.reload()

    # ------------------------------------------------------------------
    # live reload (mtime + size change detection)
    # ------------------------------------------------------------------
    def _store_token(self) -> Optional[tuple]:
        try:
            stat = self.store.path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def reload(self) -> None:
        self._specs = resolve_specs(self.settings, self.store)
        self._token = self._store_token()

    def _refresh_if_changed(self) -> None:
        """Re-resolve when the persistent store changed on disk.

        Runs on the router's loader for every request, so the router reads the
        *current* persistent configuration — including edits made externally,
        not only through this API.
        """
        if self._store_token() != self._token:
            logger.info("Provider configuration changed on disk; reloading")
            self.reload()

    def specs(self) -> List[ProviderSpec]:
        self._refresh_if_changed()
        return list(self._specs)

    def active_specs(self) -> List[ProviderSpec]:
        return [spec for spec in self.specs() if spec.usable]

    def chain(self) -> List[str]:
        return [spec.name for spec in self.specs() if spec.usable]

    def spec_for(self, provider_id: str) -> Optional[ProviderSpec]:
        for spec in self.specs():
            if spec.name == provider_id:
                return spec
        return None

    # ------------------------------------------------------------------
    # read views
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        specs = self.specs()
        rows = provider_statuses(self.settings, self.store, self.health)
        chain = [spec.name for spec in specs if spec.usable]
        active = chain[0] if chain else None
        active_model = next((s.model for s in specs if s.name == active), None)
        available = sorted(
            {row["model"] for row in rows if row["model"]}
            | {model for row in rows for model in row.get("models", [])}
        )
        return {
            "providers": rows,
            "order": self.store.effective_order(),
            "chain": chain,
            "active": active,
            "active_model": active_model,
            "mode_models": self.mode_models(),
            "research_models": self.store.research_models(),
            "available_models": available,
            "presets": [presets.as_dict(p) for p in presets.all_presets()],
            "encryption_available": self.store.box.encryption_available,
        }

    def presets(self) -> List[Dict[str, Any]]:
        return [presets.as_dict(p) for p in presets.all_presets()]

    def row(self, provider_id: str) -> Dict[str, Any]:
        for row in provider_statuses(self.settings, self.store, self.health):
            if row["id"] == provider_id:
                return row
        raise KeyError(provider_id)

    # ------------------------------------------------------------------
    # mutations (all UI-driven)
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        name: str = "",
        preset: str = "",
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        auth_type: str = "",
        api_key_required: Optional[bool] = None,
        provider_type: str = "",
        enabled: bool = True,
    ) -> Dict[str, Any]:
        preset_key = (preset or "").strip().lower()
        preset_def = presets.get(preset_key)
        display = (name or "").strip() or (preset_def.label if preset_def else "")
        candidate_base = (base_url or "").strip() or (preset_def.base_url if preset_def else "")
        if not candidate_base:
            raise ValueError("Endpoint URL is required")
        resolved_base = validate_base(candidate_base)
        if not display:
            display = host_label(resolved_base) or "Provider"

        provider_id = self.store.create_provider(
            name=display,
            preset=preset_def.key if preset_def else presets.CUSTOM_PRESET_KEY,
            base_url=resolved_base,
            api_key=api_key,
            model=(model or "").strip() or (preset_def.default_model if preset_def else ""),
            auth_type=auth_type or "",
            api_key_required=api_key_required,
            provider_type=provider_type or "",
            enabled=enabled,
        )
        self.reload()
        return self.row(provider_id)

    def update(
        self,
        provider_id: str,
        *,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enabled: Optional[bool] = None,
        auth_type: Optional[str] = None,
        api_key_required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self._require(provider_id)
        fields: Dict[str, Any] = {}
        if name is not None:
            fields["name"] = name.strip() or provider_id
        if base_url is not None:
            fields["base_url"] = base_url.strip()
        if api_key is not None:
            fields["api_key"] = api_key.strip()
        if model is not None:
            fields["model"] = model.strip()
        if enabled is not None:
            fields["enabled"] = bool(enabled)
        if auth_type is not None:
            fields["auth_type"] = auth_type.strip()
        if api_key_required is not None:
            fields["api_key_required"] = bool(api_key_required)
        self.store.set_provider(provider_id, **fields)
        self.reload()
        logger.info("Provider %s updated (model=%s enabled=%s)", provider_id, model, enabled)
        return self.row(provider_id)

    def remove(self, provider_id: str) -> bool:
        self._require(provider_id)
        removed = self.store.delete_provider(provider_id)
        self.health.reset_provider(provider_id)
        self.reload()
        logger.info("Provider %s removed (credential deleted)", provider_id)
        return removed

    def set_order(self, order: List[str]) -> List[str]:
        for provider_id in order:
            self._require(provider_id)
        self.store.set_order(order)
        self.reload()
        logger.info("Provider order updated: %s", ",".join(self.store.effective_order()))
        return self.store.effective_order()

    # ------------------------------------------------------------------
    # probing
    # ------------------------------------------------------------------
    def discover_models(self, provider_id: str, *, api_key: Optional[str] = None) -> List[str]:
        spec = self._spec_with_overrides(provider_id, api_key=api_key)
        if spec is None:
            return []
        models = fetch_models(spec)
        if models:
            self.store.set_models(provider_id, models)
            self.reload()
        return models

    def discover_models_draft(
        self,
        *,
        preset: str = "",
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        auth_type: str = "",
        api_key_required: Optional[bool] = None,
    ) -> List[str]:
        """Discover models for a provider that has not been saved yet."""
        try:
            resolved_base = validate_base(base_url)
        except ValueError:
            return []
        spec = spec_from_preset(
            (preset or "").strip().lower(),
            api_key=api_key,
            model=model,
            base_url=resolved_base,
            auth_type=auth_type,
            api_key_required=api_key_required,
        )
        return fetch_models(spec)

    def test(self, provider_id: str, *, api_key: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        self._require(provider_id)
        spec = self._spec_with_overrides(provider_id, api_key=api_key, model=model)
        return self._run_probe(provider_id, spec)

    def test_draft(
        self,
        *,
        preset: str = "",
        name: str = "",
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        auth_type: str = "",
        api_key_required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Test a provider that has not been saved yet (the Add form's Test)."""
        try:
            resolved_base = validate_base(base_url)
        except ValueError as exc:
            return {"ok": False, "status": "error", "error": str(exc), "model": model}
        spec = spec_from_preset(
            (preset or "").strip().lower(),
            api_key=api_key,
            model=model,
            base_url=resolved_base,
            name=name or (presets.get(preset).label if presets.get(preset) else "Provider"),
            api_key_required=api_key_required,
            auth_type=auth_type,
        )
        return self._run_probe(None, spec)

    def _spec_with_overrides(
        self,
        provider_id: str,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[ProviderSpec]:
        record = self.store.record(provider_id)
        if not record:
            return None
        spec = spec_from_record(self.settings, self.store, record)
        if spec is None:
            return None
        effective_key = (api_key or "").strip() or spec.api_key
        effective_model = (model or "").strip() or spec.model
        if effective_key == spec.api_key and effective_model == spec.model:
            return spec
        return ProviderSpec(
            name=spec.name,
            label=spec.label,
            base_url=spec.base_url,
            api_key=effective_key,
            model=effective_model,
            enabled=spec.enabled,
            api_key_required=spec.api_key_required,
            auth_type=spec.auth_type,
            provider_type=spec.provider_type,
            preset=spec.preset,
            extra_payload=dict(spec.extra_payload),
            extra_headers=dict(spec.extra_headers),
        )

    def _run_probe(self, provider_id: Optional[str], spec: Optional[ProviderSpec]) -> Dict[str, Any]:
        if spec is None:
            return {"ok": False, "status": "error", "error": "Provider not found", "model": ""}
        result = probe(spec, model=spec.model)
        if provider_id:
            if result.get("ok"):
                self.health.record_success(provider_id, model=spec.model)
            else:
                # 401/403/404/400 are configuration problems, not provider
                # availability problems — record them so the UI can explain,
                # but do not put the provider into a rate-limit cooldown.
                self.health.record_failure(
                    provider_id,
                    status=result.get("http_status"),
                    message=result.get("error", ""),
                )
        return result

    # ------------------------------------------------------------------
    # per-mode / role model selection
    # ------------------------------------------------------------------
    def mode_models(self) -> Dict[str, Dict[str, str]]:
        return {mode: self.mode_model(mode) for mode in MODES}

    def mode_model(self, mode: str) -> Dict[str, str]:
        saved = self.store.mode_models().get(mode) or {}
        provider = (saved.get("provider") or "").strip()
        model = (saved.get("model") or "").strip()
        if not provider:
            return {"provider": "", "model": ""}
        spec = self.spec_for(provider)
        if spec is None or not spec.usable:
            return {"provider": "", "model": ""}
        return {"provider": provider, "model": model or spec.model}

    def set_mode_model(self, mode: str, provider: str = "", model: str = "") -> Dict[str, Dict[str, str]]:
        provider = (provider or "").strip()
        if provider:
            self._require(provider)
        self.store.set_mode_model(mode, provider=provider, model=model)
        self.reload()
        logger.info("Model selection for mode=%s -> provider=%r model=%r", mode, provider, model)
        return self.mode_models()

    def research_models(self) -> Dict[str, str]:
        return self.store.research_models()

    def set_research_models(self, *, fast: Optional[str] = None, strong: Optional[str] = None) -> Dict[str, str]:
        updated = self.store.set_research_models(fast=fast, strong=strong)
        logger.info("Research role models updated: fast=%r strong=%r", updated["fast"], updated["strong"])
        return updated

    # ------------------------------------------------------------------
    def has(self, provider_id: str) -> bool:
        cleaned = (provider_id or "").strip()
        return bool(cleaned) and self.store.has_record(cleaned)

    def _require(self, provider_id: str) -> str:
        cleaned = (provider_id or "").strip()
        if not cleaned or not self.store.has_record(cleaned):
            raise KeyError(provider_id)
        return cleaned
