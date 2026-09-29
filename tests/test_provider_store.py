"""Dynamic provider persistence: user-defined records, encryption, migration."""

from __future__ import annotations

import json

from backend.config import Settings
from backend.providers.health import HealthTracker
from backend.providers.store import (
    ProviderStore,
    key_fingerprint,
    mask_key,
    provider_statuses,
    resolve_specs,
    slugify,
)


def make_settings(tmp_path, **over):
    base = dict(
        _env_file=None,
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "docs"),
        nvidia_api_key="",
        groq_api_key="",
        openrouter_api_key="",
        google_api_key="",
    )
    base.update(over)
    return Settings(**base)


def test_create_any_provider_persists_encrypted_key(tmp_path):
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    provider_id = store.create_provider(
        name="My New Provider",
        base_url="https://example.com/v1",
        api_key="secret-key-9999",
        model="some-model",
    )
    assert provider_id == "my-new-provider"

    on_disk = (settings.resolved_data_dir() / "providers.json").read_text(encoding="utf-8")
    assert "secret-key-9999" not in on_disk, "raw key must never hit disk"
    assert "enc:" in on_disk

    reopened = ProviderStore(settings)
    assert reopened.api_key(provider_id) == "secret-key-9999"
    assert reopened.record(provider_id)["name"] == "My New Provider"
    assert reopened.record(provider_id)["base_url"] == "https://example.com/v1"


def test_provider_that_exists_nowhere_in_source_is_resolved(tmp_path):
    """The architectural acceptance check: an unknown provider name still works."""
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    store.create_provider(
        name="Totally Made Up Co",
        base_url="https://llm.made-up.example/v1",
        api_key="k",
        model="made-up-model",
    )
    specs = {s.name: s for s in resolve_specs(settings, store)}
    spec = specs["totally-made-up-co"]
    assert spec.base_url == "https://llm.made-up.example/v1"
    assert spec.model == "made-up-model"
    assert spec.usable is True


def test_encrypt_creates_key_file_but_decrypt_never_does(tmp_path):
    store = ProviderStore(make_settings(tmp_path))
    assert store.box.key_available is False
    token = store.box.encrypt("secret-value")
    assert store.box.key_available is True
    assert store.box.decrypt(token) == "secret-value"

    store.box.key_path.unlink()
    fresh = ProviderStore(make_settings(tmp_path))
    assert fresh.box.decrypt(token) == ""
    assert fresh.box.key_path.exists() is False


def test_lost_key_file_is_surfaced_not_silently_unconfigured(tmp_path):
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    store.create_provider(name="Google Gemini", preset="google", base_url="https://x/v1", api_key="g-secret")
    assert store.key_state("google") == "ok"

    store.box.key_path.unlink()
    fresh = ProviderStore(settings)
    assert fresh.key_state("google") == "undecryptable"

    row = {r["id"]: r for r in provider_statuses(settings, fresh, HealthTracker())}["google"]
    assert row["status"] == "error"
    assert "provider_key" in row["last_error"]


def test_statuses_mask_secrets(tmp_path):
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    store.create_provider(name="OpenRouter", preset="openrouter",
                          base_url="https://openrouter.ai/api/v1", api_key="sk-or-abcdef1234")
    row = provider_statuses(settings, store, HealthTracker())[0]
    assert "sk-or-abcdef1234" not in json.dumps(row)
    assert row["key_masked"].endswith("1234")
    assert len(row["key_fingerprint"]) == 8
    assert row["has_key"] is True


def test_delete_removes_record_credential_and_mode_reference(tmp_path):
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    pid = store.create_provider(name="Groq", preset="groq", base_url="https://api.groq.com/openai/v1",
                                api_key="gsk_x", model="qwen/qwen3.8-27b")
    store.set_mode_model("chat", provider=pid, model="qwen/qwen3.8-27b")

    assert store.delete_provider(pid) is True
    assert store.has_record(pid) is False
    assert store.api_key(pid) == ""
    assert store.mode_models()["chat"] == {"provider": "", "model": ""}
    assert store.delete_provider(pid) is False


def test_order_is_configurable_and_unknown_ids_ignored(tmp_path):
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    a = store.create_provider(name="A", base_url="https://a/v1", api_key="k")
    b = store.create_provider(name="B", base_url="https://b/v1", api_key="k")
    c = store.create_provider(name="C", base_url="https://c/v1", api_key="k")

    store.set_order([c, "does-not-exist", a])
    assert store.effective_order() == [c, a, b]


def test_api_key_optional_provider_is_usable_without_key(tmp_path):
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    pid = store.create_provider(name="Ollama", preset="ollama", base_url="http://localhost:11434/v1",
                                api_key_required=False, auth_type="none")
    spec = store.spec(pid)
    assert spec.api_key_required is False
    assert spec.usable is True
    assert spec.auth_type == "none"


def test_url_is_normalised_on_create(tmp_path):
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    pid = store.create_provider(name="OpenAI", base_url="https://api.openai.com/v1/chat/completions")
    assert store.record(pid)["base_url"] == "https://api.openai.com/v1"


def test_invalid_base_url_is_rejected(tmp_path):
    store = ProviderStore(make_settings(tmp_path))
    for bad in ("", "not-a-url", "ftp://x/y", "https://api.openai.com/v1?x=1"):
        try:
            store.create_provider(name="X", base_url=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_v1_store_migrates_to_v2(tmp_path):
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    legacy = {
        "version": 1,
        "order": ["groq"],
        "providers": {
            "groq": {
                "model": "qwen/qwen3.8-27b",
                "api_key_enc": store.box.encrypt("legacy-groq-key"),
                "enabled": True,
                "models": ["qwen/qwen3.8-27b"],
            }
        },
    }
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = ProviderStore(settings)
    record = migrated.record("groq")
    assert record["preset"] == "groq"
    assert record["name"] == "Groq"
    assert record["base_url"] == "https://api.groq.com/openai/v1"
    assert migrated.api_key("groq") == "legacy-groq-key"
    assert migrated.effective_order() == ["groq"]
    assert json.loads(store.path.read_text(encoding="utf-8"))["version"] == 2


def test_env_bootstrap_seeds_once_and_does_not_resurrect(tmp_path):
    settings = make_settings(tmp_path, groq_api_key="env-groq-key")
    store = ProviderStore(settings)
    store.ensure_bootstrapped()
    assert store.has_record("groq")
    assert store.api_key("groq") == "env-groq-key"

    # A provider the user deletes must not come back on the next start.
    store.delete_provider("groq")
    fresh = ProviderStore(settings)
    fresh.ensure_bootstrapped()
    assert fresh.has_record("groq") is False


def test_env_bootstrap_honours_the_env_priority_order(tmp_path):
    settings = make_settings(
        tmp_path,
        athena_llm_providers="groq,nvidia",
        groq_api_key="gk", nvidia_api_key="nk", openrouter_api_key="ork",
    )
    store = ProviderStore(settings)
    store.ensure_bootstrapped()
    # groq leads because ATHENA_LLM_PROVIDERS says so; openrouter is appended.
    assert store.effective_order() == ["groq", "nvidia", "openrouter"]


def test_mode_and_role_models_round_trip(tmp_path):
    settings = make_settings(tmp_path)
    store = ProviderStore(settings)
    store.set_mode_model("chat", provider="abc", model="m1")
    assert store.mode_models()["chat"] == {"provider": "abc", "model": "m1"}
    store.set_research_models(fast="f", strong="s")
    assert store.research_models() == {"fast": "f", "strong": "s"}


def test_mask_fingerprint_and_slug_helpers():
    assert mask_key("nvapi-abcdef1234").endswith("1234")
    assert "abcdef1234" not in mask_key("nvapi-abcdef1234")
    assert len(key_fingerprint("x")) == 8
    assert slugify("My New Provider!") == "my-new-provider"
    assert slugify("") == "provider"
