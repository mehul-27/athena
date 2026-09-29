"""Persistent provider configuration — user-defined, not source-defined.

`data/providers.json` is the source of truth for which LLM providers Athena can
use, their endpoints, keys, models and priority. Nothing here constrains the
*set* of providers: a provider whose name appears nowhere in Athena's source can
be created at runtime through the Settings API/UI.

Schema (version 2)::

    {
      "version": 2,
      "order": ["<id>", ...],                # priority, first = highest
      "providers": {
        "<id>": {
          "id", "name", "preset", "provider_type", "base_url",
          "api_key_enc", "auth_type", "api_key_required",
          "model", "models", "enabled", "extra_body", "metadata"
        }
      },
      "mode_models": {"chat": {"provider": "<id>", "model": "..."}, ...},
      "research_models": {"fast": "...", "strong": "..."},
      "bootstrapped": true
    }

Version-1 files (the old fixed catalogue: nvidia/groq/openrouter/google keyed by
name) migrate automatically on load — the old name becomes the provider id, so
existing `mode_models`, priority and keys keep working unchanged.

API keys are encrypted at rest (`backend/security/secrets.py`); only masked hints
and a non-secret fingerprint ever leave this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.providers import presets
from backend.providers.health import HealthTracker, health_summary
from backend.providers.providers import ProviderSpec, spec_from_record
from backend.providers.urls import is_local_url, validate_base
from backend.security.secrets import SecretBox, is_encrypted

if TYPE_CHECKING:  # avoid a config <-> providers import cycle
    from backend.config import Settings

logger = logging.getLogger(__name__)

STORE_VERSION = 2
STORE_FILENAME = "providers.json"
KEY_FILENAME = ".provider_key"

# Per-mode model selection offered in the UI.
MODES = ("chat", "rag", "search", "research")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def mask_key(api_key: str) -> str:
    """A display-only hint: bullets plus the last four characters."""
    key = (api_key or "").strip()
    if not key:
        return ""
    if len(key) <= 4:
        return "•" * len(key)
    return "•" * 8 + key[-4:]


def key_fingerprint(api_key: str) -> str:
    """Stable, non-secret label for distinguishing two keys for the same provider."""
    key = (api_key or "").strip()
    if not key:
        return ""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or "provider"


class ProviderStore:
    def __init__(self, settings: "Settings", *, box: Optional[SecretBox] = None) -> None:
        self.settings = settings
        data_dir = settings.resolved_data_dir()
        self.path = data_dir / STORE_FILENAME
        self.box = box or SecretBox(data_dir / KEY_FILENAME)

    # ------------------------------------------------------------------
    # raw persistence
    # ------------------------------------------------------------------
    def _empty(self) -> Dict[str, Any]:
        return {"version": STORE_VERSION, "order": [], "providers": {}}

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s: %s", self.path, exc)
            return self._empty()
        if not isinstance(data, dict):
            return self._empty()

        data = self._migrate(data)

        providers = data.get("providers")
        data["providers"] = providers if isinstance(providers, dict) else {}
        order = data.get("order")
        data["order"] = [str(x) for x in order] if isinstance(order, list) else []
        return data

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data["version"] = STORE_VERSION
        fd, temp_name = tempfile.mkstemp(prefix="providers-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
            Path(temp_name).replace(self.path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # migration
    # ------------------------------------------------------------------
    def _migrate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Bring a version-1 (fixed-catalogue) file up to version 2."""
        version = data.get("version")
        providers = data.get("providers")
        if version == STORE_VERSION or not isinstance(providers, dict):
            return data

        order = data.get("order") if isinstance(data.get("order"), list) else []
        migrated: Dict[str, Any] = {}
        for old_name, record in providers.items():
            if not isinstance(record, dict):
                continue
            pid = str(old_name)
            preset = presets.get(pid)
            migrated[pid] = {
                "id": pid,
                "name": preset.label if preset else pid,
                "preset": pid if preset else presets.CUSTOM_PRESET_KEY,
                "provider_type": preset.provider_type if preset else "openai_compatible",
                "base_url": record.get("base_url") or (preset.base_url if preset else ""),
                "api_key_enc": record.get("api_key_enc", ""),
                "auth_type": preset.auth_type if preset else "bearer",
                "api_key_required": preset.api_key_required if preset else True,
                "model": record.get("model", ""),
                "models": list(record.get("models") or []),
                "enabled": bool(record.get("enabled", True)),
                "extra_body": dict(preset.extra_body) if preset else {},
                "metadata": {},
                "created_at": record.get("created_at", time.time()),
                "updated_at": record.get("updated_at", time.time()),
            }
        data["providers"] = migrated
        data["order"] = [pid for pid in order if pid in migrated]
        data["version"] = STORE_VERSION
        logger.info("Migrated %d provider(s) from store schema v1 to v2", len(migrated))
        try:
            self.save(data)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not persist the migrated provider store: %s", exc)
        return data

    # ------------------------------------------------------------------
    # bootstrap from .env (runs once per store file)
    # ------------------------------------------------------------------
    def _bootstrap_order(self) -> List[str]:
        """Bootstrap priority: honour ATHENA_LLM_PROVIDERS, then the defaults."""
        raw = getattr(self.settings, "athena_llm_providers", "") or ""
        env_order = [part.strip().lower() for part in raw.split(",") if part.strip()]
        order = [key for key in env_order if key in presets.BOOTSTRAP_PRESET_KEYS]
        for key in presets.BOOTSTRAP_PRESET_KEYS:
            if key not in order:
                order.append(key)
        return order

    def ensure_bootstrapped(self) -> None:
        """Seed legacy `.env` providers on first run, then never again.

        `.env` is a bootstrap fallback only. Once a store exists it is
        authoritative — a provider the user deletes from Settings must not
        reappear on the next start.
        """
        data = self.load()
        if data.get("bootstrapped"):
            return
        seeded = 0
        for key in self._bootstrap_order():
            preset = presets.get(key)
            if preset is None or key in data["providers"]:
                continue
            api_key = presets.env_key(self.settings, preset)
            model = presets.env_model(self.settings, preset) or preset.default_model
            if not api_key and not model:
                continue
            if preset.api_key_required and not api_key:
                continue
            self._insert(data, name=preset.label, preset_key=key, base_url=preset.base_url,
                         api_key=api_key, model=model, enabled=True)
            seeded += 1
        data["bootstrapped"] = True
        self.save(data)
        if seeded:
            logger.info("Seeded %d provider(s) from .env (bootstrap)", seeded)

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------
    def records(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.load().get("providers", {}))

    def record(self, provider_id: str) -> Dict[str, Any]:
        return dict(self.records().get(provider_id, {}))

    def has_record(self, provider_id: str) -> bool:
        return provider_id in self.records()

    def api_key(self, provider_id: str) -> str:
        stored = self.record(provider_id).get("api_key_enc", "")
        if not stored:
            return ""
        return self.box.decrypt(stored)

    def key_state(self, provider_id: str) -> str:
        """`none` (never saved) | `ok` | `undecryptable` (key file lost/replaced)."""
        stored = self.record(provider_id).get("api_key_enc", "")
        if not stored:
            return "none"
        if not is_encrypted(stored):
            return "ok"
        return "ok" if self.box.decrypt(stored) else "undecryptable"

    # -- id allocation --------------------------------------------------
    def _unique_id(self, data: Dict[str, Any], name: str, preset_key: str = "") -> str:
        base = slugify(preset_key if preset_key and preset_key != presets.CUSTOM_PRESET_KEY else name)
        taken = set(data["providers"])
        if base not in taken:
            return base
        for n in range(2, 1000):
            candidate = f"{base}-{n}"
            if candidate not in taken:
                return candidate
        return f"{base}-{uuid.uuid4().hex[:6]}"

    def _insert(
        self,
        data: Dict[str, Any],
        *,
        name: str,
        preset_key: str,
        base_url: str,
        api_key: str = "",
        model: str = "",
        enabled: bool = True,
        auth_type: str = "",
        api_key_required: Optional[bool] = None,
        provider_type: str = "",
        extra_body: Optional[Dict[str, Any]] = None,
        models: Optional[List[str]] = None,
    ) -> str:
        preset = presets.get(preset_key)
        provider_id = self._unique_id(data, name, preset_key if preset else "")
        record = {
            "id": provider_id,
            "name": (name or (preset.label if preset else provider_id)).strip(),
            "preset": preset.key if preset else presets.CUSTOM_PRESET_KEY,
            "provider_type": provider_type or (preset.provider_type if preset else "openai_compatible"),
            "base_url": base_url or (preset.base_url if preset else ""),
            "api_key_enc": self.box.encrypt(api_key) if api_key else "",
            "auth_type": auth_type or (preset.auth_type if preset else "bearer"),
            "api_key_required": (
                preset.api_key_required if api_key_required is None and preset else
                (True if api_key_required is None else bool(api_key_required))
            ),
            "model": model or (preset.default_model if preset else ""),
            "models": list(models or []),
            "enabled": bool(enabled),
            "extra_body": dict(extra_body if extra_body is not None else (preset.extra_body if preset else {})),
            "metadata": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        data["providers"][provider_id] = record
        data["order"].append(provider_id)
        return provider_id

    def create_provider(self, **fields: Any) -> str:
        """Create a brand-new provider record. Returns its id."""
        data = self.load()
        base_url = validate_base(fields.get("base_url") or "")
        provider_id = self._insert(
            data,
            name=fields.get("name") or "",
            preset_key=fields.get("preset") or presets.CUSTOM_PRESET_KEY,
            base_url=base_url,
            api_key=fields.get("api_key") or "",
            model=fields.get("model") or "",
            enabled=fields.get("enabled", True),
            auth_type=fields.get("auth_type") or "",
            api_key_required=fields.get("api_key_required"),
            provider_type=fields.get("provider_type") or "",
            extra_body=fields.get("extra_body"),
        )
        self.save(data)
        logger.info("Added provider id=%s name=%r base_url=%s", provider_id, fields.get("name"), base_url)
        return provider_id

    def set_provider(self, provider_id: str, **fields: Any) -> Dict[str, Any]:
        """Merge `fields` into a provider record and persist."""
        data = self.load()
        record = data["providers"].get(provider_id)
        if record is None:
            raise KeyError(provider_id)
        for field_name, value in fields.items():
            if value is None:
                continue
            if field_name == "api_key":
                record["api_key_enc"] = self.box.encrypt(value)
            elif field_name == "base_url":
                record["base_url"] = validate_base(value)
            else:
                record[field_name] = value
        record["updated_at"] = time.time()
        self.save(data)
        return dict(record)

    def delete_provider(self, provider_id: str) -> bool:
        data = self.load()
        existed = data["providers"].pop(provider_id, None) is not None
        data["order"] = [pid for pid in data["order"] if pid != provider_id]
        modes = data.get("mode_models")
        if isinstance(modes, dict):
            for mode, selection in list(modes.items()):
                if isinstance(selection, dict) and selection.get("provider") == provider_id:
                    modes[mode] = {"provider": "", "model": ""}
        self.save(data)
        return existed

    def set_order(self, order: List[str]) -> None:
        data = self.load()
        known = set(data["providers"])
        cleaned = [pid for pid in order if pid in known]
        for pid in data["order"]:
            if pid not in cleaned and pid in known:
                cleaned.append(pid)
        data["order"] = cleaned
        self.save(data)

    def set_models(self, provider_id: str, models: List[str]) -> None:
        data = self.load()
        record = data["providers"].get(provider_id)
        if record is None:
            return
        record["models"] = [str(m) for m in models]
        record["updated_at"] = time.time()
        self.save(data)

    # ------------------------------------------------------------------
    # resolved views
    # ------------------------------------------------------------------
    def effective_order(self) -> List[str]:
        data = self.load()
        known = list(data["providers"])
        ordered = [pid for pid in data["order"] if pid in data["providers"]]
        for pid in known:
            if pid not in ordered:
                ordered.append(pid)
        return ordered

    def spec(self, provider_id: str) -> Optional[ProviderSpec]:
        record = self.record(provider_id)
        if not record:
            return None
        return spec_from_record(self.settings, self, record)

    # -- per-mode / role model selection --------------------------------
    def mode_models(self) -> Dict[str, Dict[str, str]]:
        saved = self.load().get("mode_models")
        saved = saved if isinstance(saved, dict) else {}
        out: Dict[str, Dict[str, str]] = {}
        for mode in MODES:
            entry = saved.get(mode)
            entry = entry if isinstance(entry, dict) else {}
            out[mode] = {
                "provider": str(entry.get("provider") or ""),
                "model": str(entry.get("model") or ""),
            }
        return out

    def set_mode_model(self, mode: str, provider: str = "", model: str = "") -> Dict[str, Dict[str, str]]:
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode!r}")
        data = self.load()
        current = data.get("mode_models")
        current = dict(current) if isinstance(current, dict) else {}
        current[mode] = {"provider": (provider or "").strip(), "model": (model or "").strip()}
        data["mode_models"] = current
        self.save(data)
        return self.mode_models()

    def research_models(self) -> Dict[str, str]:
        saved = self.load().get("research_models") or {}
        fast = str(saved.get("fast") or "") or (getattr(self.settings, "research_fast_model", "") or "")
        strong = str(saved.get("strong") or "") or (getattr(self.settings, "research_strong_model", "") or "")
        return {"fast": fast, "strong": strong}

    def set_research_models(self, *, fast: Optional[str] = None, strong: Optional[str] = None) -> Dict[str, str]:
        data = self.load()
        current = data.get("research_models")
        current = dict(current) if isinstance(current, dict) else {}
        if fast is not None:
            current["fast"] = fast.strip()
        if strong is not None:
            current["strong"] = strong.strip()
        data["research_models"] = current
        self.save(data)
        return self.research_models()


# ----------------------------------------------------------------------
# resolution helpers
# ----------------------------------------------------------------------
def resolve_specs(settings: "Settings", store: Optional[ProviderStore] = None) -> List[ProviderSpec]:
    """Every configured provider, in priority order, as a router-usable spec."""
    store = store or ProviderStore(settings)
    specs: List[ProviderSpec] = []
    for provider_id in store.effective_order():
        spec = store.spec(provider_id)
        if spec is not None:
            specs.append(spec)
    return specs


def provider_statuses(
    settings: "Settings",
    store: ProviderStore,
    health: HealthTracker,
) -> List[Dict[str, Any]]:
    """The Settings-UI view: one row per configured provider, secrets masked."""
    rows: List[Dict[str, Any]] = []
    for position, provider_id in enumerate(store.effective_order(), 1):
        record = store.record(provider_id)
        spec = store.spec(provider_id)
        if spec is None:
            continue
        stored_key = store.api_key(provider_id)
        key_state = store.key_state(provider_id)
        key_error = ""
        if key_state == "undecryptable":
            key_error = (
                "Stored API key could not be decrypted — data/.provider_key is missing "
                "or was replaced. Re-enter the key."
            )

        if key_state == "undecryptable" and not spec.api_key:
            status = "error"
        elif not spec.enabled:
            status = "disabled"
        elif spec.api_key_required and not spec.api_key:
            status = "not_configured"
        else:
            status, _ = health_summary(health.get(provider_id))

        rows.append(
            {
                "id": provider_id,
                "name": record.get("name") or provider_id,
                "label": spec.label,
                "preset": record.get("preset", ""),
                "type": record.get("provider_type", "openai_compatible"),
                "is_local": record.get("provider_type") == "local" or is_local_url(spec.base_url),
                "status": status,
                "enabled": spec.enabled,
                "configured": bool(spec.api_key) or not spec.api_key_required,
                "model": spec.model,
                "base_url": spec.base_url,
                "auth_type": record.get("auth_type", "bearer"),
                "api_key_required": spec.api_key_required,
                "has_key": bool(spec.api_key),
                "key_masked": mask_key(spec.api_key),
                "key_fingerprint": key_fingerprint(spec.api_key),
                "key_source": "settings" if stored_key else "none",
                "key_state": key_state,
                "models": list(record.get("models") or []),
                "last_error": key_error or health.get(provider_id).last_error,
                "last_error_at": health.get(provider_id).last_error_at,
                "last_success_at": health.get(provider_id).last_success_at,
                "priority": position,
            }
        )
    return rows
