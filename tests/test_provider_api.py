"""Settings API: providers are user-defined, not catalogue-defined.

The headline test is `test_add_brand_new_provider_entirely_through_the_api` —
a provider whose name exists nowhere in Athena's source is created, given an
endpoint/key/model, made top priority and exposed to the router, all via HTTP.

All provider HTTP is mocked — no quota is consumed.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import backend.providers.discovery as discovery_mod
import backend.providers.llm as llm_mod
from backend.config import Settings
from backend.main import create_app
from backend.providers.store import ProviderStore

SECRET = "sk-THIS-SHOULD-NEVER-LEAK-4242"


def make_settings(tmp_path, **over):
    base = dict(
        _env_file=None,
        embedding_provider="hashing",
        chroma_mode="embedded",
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "docs"),
        chroma_path=str(tmp_path / "chroma"),
        rag_collection="providers-api-test",
        nvidia_api_key="",
        groq_api_key="",
        openrouter_api_key="",
        google_api_key="",
    )
    base.update(over)
    return Settings(**base)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


class Resp:
    def __init__(self, status_code, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


def _chat_ok(content="ok"):
    return Resp(200, {"choices": [{"message": {"content": content}}]})


# ----------------------------------------------------------------------
# THE architectural acceptance test
# ----------------------------------------------------------------------
def test_add_brand_new_provider_entirely_through_the_api(client):
    """Provider name/endpoint/model that appear nowhere in Athena's source."""
    # 1. Add it exactly like the Settings form does.
    created = client.post(
        "/api/providers",
        json={
            "name": "My Test Provider",
            "base_url": "https://example.com/v1",
            "api_key": SECRET,
            "model": "my-model",
        },
    )
    assert created.status_code == 200, created.text
    row = created.json()["provider"]
    provider_id = row["id"]
    assert provider_id == "my-test-provider"
    assert row["name"] == "My Test Provider"
    assert row["model"] == "my-model"
    assert row["status"] == "available"

    # 2. It is now a first-class provider in the listing...
    listing = client.get("/api/providers").json()
    assert provider_id in [p["id"] for p in listing["providers"]]
    # 3. ...and in the live router chain (no restart).
    assert provider_id in client.app.state.llm.provider_names
    assert SECRET not in json.dumps(listing)

    # 4. Move it to priority #1.
    reordered = client.put("/api/providers/order", json={"order": [provider_id]}).json()
    assert reordered["order"][0] == provider_id
    assert client.app.state.llm.provider_names[0] == provider_id

    # 5. Select its model for Chat.
    modes = client.put(
        "/api/providers/mode-models",
        json={"mode": "chat", "provider": provider_id, "model": "my-model"},
    ).json()
    assert modes["mode_models"]["chat"] == {"provider": provider_id, "model": "my-model"}
    # (the chat request itself is exercised — with mocked HTTP — in
    # `test_new_provider_round_trips_through_the_router`)


def test_new_provider_round_trips_through_the_router(client, monkeypatch):
    client.post("/api/providers", json={
        "name": "My Test Provider", "base_url": "https://example.com/v1",
        "api_key": "test-key", "model": "my-model",
    })
    seen = {}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen["body"] = kwargs.get("json")
        seen["headers"] = kwargs.get("headers")
        return _chat_ok("answered")

    monkeypatch.setattr(llm_mod.httpx, "post", fake_post)
    body = client.post("/api/chat", json={"message": "hi"}).json()

    assert body["answer"] == "answered"
    assert seen["url"] == "https://example.com/v1/chat/completions"
    assert seen["body"]["model"] == "my-model"
    assert seen["headers"]["Authorization"] == "Bearer test-key"


# ----------------------------------------------------------------------
# presets exist but are not required
# ----------------------------------------------------------------------
def test_presets_are_offered_and_include_custom_and_local(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        presets = {p["key"]: p for p in c.get("/api/providers").json()["presets"]}
    assert "custom" in presets, "a generic OpenAI-compatible option must always exist"
    assert presets["custom"]["api_key_required"] is True
    for key in ("openai", "deepseek", "anthropic", "groq", "google", "ollama", "vercel"):
        assert key in presets
    assert presets["ollama"]["is_local"] is True
    assert presets["ollama"]["api_key_required"] is False


def test_vercel_ai_gateway_preset_is_openai_compatible(client):
    preset = next(p for p in client.get("/api/providers").json()["presets"] if p["key"] == "vercel")
    assert preset["base_url"] == "https://ai-gateway.vercel.sh/v1"
    assert preset["auth_type"] == "bearer"
    assert preset["provider_type"] == "cloud"


def test_preset_defaults_are_prefilled_when_creating(client):
    row = client.post("/api/providers", json={
        "preset": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
        "api_key": "d-key",
    }).json()["provider"]
    assert row["model"] == "deepseek-chat"      # from the preset
    assert row["type"] == "cloud"


def test_preset_is_only_a_default_and_can_be_overridden(client):
    row = client.post("/api/providers", json={
        "preset": "openai", "name": "OpenAI (my proxy)",
        "base_url": "https://proxy.internal/v1", "api_key": "k", "model": "gpt-4o-mini",
    }).json()["provider"]
    assert row["base_url"] == "https://proxy.internal/v1"
    assert row["name"] == "OpenAI (my proxy)"


# ----------------------------------------------------------------------
# edit / enable / reorder / remove
# ----------------------------------------------------------------------
def test_edit_endpoint_model_and_key(client):
    pid = client.post("/api/providers", json={
        "name": "P", "base_url": "https://a.example/v1", "api_key": "k1", "model": "m1",
    }).json()["provider"]["id"]

    updated = client.patch(f"/api/providers/{pid}", json={
        "base_url": "https://b.example/v1/chat/completions", "model": "m2", "api_key": "k2",
    }).json()["provider"]
    assert updated["base_url"] == "https://b.example/v1"   # normalised
    assert updated["model"] == "m2"
    assert updated["key_fingerprint"] != ""


def test_enable_disable_and_reorder(client):
    a = client.post("/api/providers", json={"name": "A", "base_url": "https://a/v1", "api_key": "k"}).json()["provider"]["id"]
    b = client.post("/api/providers", json={"name": "B", "base_url": "https://b/v1", "api_key": "k"}).json()["provider"]["id"]
    assert client.app.state.llm.provider_names == [a, b]

    client.patch(f"/api/providers/{a}", json={"enabled": False})
    assert client.app.state.llm.provider_names == [b]
    client.patch(f"/api/providers/{a}", json={"enabled": True})

    order = client.put("/api/providers/order", json={"order": [b, a]}).json()["order"]
    assert order == [b, a]
    assert client.app.state.llm.provider_names == [b, a]


def test_remove_provider(client):
    pid = client.post("/api/providers", json={
        "name": "Gone", "base_url": "https://g/v1", "api_key": "k",
    }).json()["provider"]["id"]
    assert client.delete(f"/api/providers/{pid}").json()["removed"] is True
    assert pid not in [p["id"] for p in client.get("/api/providers").json()["providers"]]
    assert client.get(f"/api/providers/{pid}/models").status_code == 404


def test_unknown_provider_id_is_404(client):
    assert client.patch("/api/providers/nope", json={"model": "x"}).status_code == 404
    assert client.delete("/api/providers/nope").status_code == 404
    assert client.post("/api/providers/nope/test", json={}).status_code == 404


def test_bad_endpoint_is_rejected(client):
    assert client.post("/api/providers", json={"name": "X", "base_url": "not-a-url"}).status_code == 400
    assert client.post("/api/providers", json={"name": "X", "base_url": ""}).status_code == 422


# ----------------------------------------------------------------------
# secrets
# ----------------------------------------------------------------------
def test_secret_is_encrypted_on_disk_and_never_returned(client, tmp_path):
    pid = client.post("/api/providers", json={
        "name": "Secretive", "base_url": "https://s/v1", "api_key": SECRET,
    }).json()["provider"]["id"]

    on_disk = (tmp_path / "data" / "providers.json").read_text(encoding="utf-8")
    assert SECRET not in on_disk
    assert "enc:" in on_disk

    listing = client.get("/api/providers")
    assert SECRET not in listing.text
    row = next(p for p in listing.json()["providers"] if p["id"] == pid)
    assert row["has_key"] is True
    assert row["key_masked"].endswith("4242")
    assert len(row["key_fingerprint"]) == 8


def test_removing_provider_deletes_its_credential(client, tmp_path):
    pid = client.post("/api/providers", json={
        "name": "Temp", "base_url": "https://t/v1", "api_key": "temp-secret-9999",
    }).json()["provider"]["id"]
    client.delete(f"/api/providers/{pid}")
    store = ProviderStore(client.app.state.settings)
    assert store.api_key(pid) == ""
    assert store.has_record(pid) is False


# ----------------------------------------------------------------------
# connectivity test
# ----------------------------------------------------------------------
def test_test_provider_success_and_failure(client, monkeypatch):
    pid = client.post("/api/providers", json={
        "name": "P", "base_url": "https://p.example/v1", "api_key": "k", "model": "m",
    }).json()["provider"]["id"]

    monkeypatch.setattr(llm_mod.httpx, "post", lambda url, **kw: _chat_ok("OK"))
    ok = client.post(f"/api/providers/{pid}/test", json={}).json()
    assert ok["ok"] is True and ok["status"] == "available"

    monkeypatch.setattr(llm_mod.httpx, "post", lambda url, **kw: Resp(401, text='{"error":"bad key"}'))
    bad = client.post(f"/api/providers/{pid}/test", json={}).json()
    assert bad["ok"] is False
    assert bad["status"] == "auth_failed"


@pytest.mark.parametrize(
    "status,expected",
    [(401, "auth_failed"), (403, "auth_failed"), (404, "model_unavailable"),
     (429, "rate_limited"), (500, "endpoint_unavailable"), (400, "error")],
)
def test_test_provider_maps_status_codes(client, monkeypatch, status, expected):
    pid = client.post("/api/providers", json={
        "name": f"P{status}", "base_url": f"https://p{status}.example/v1", "api_key": "k", "model": "m",
    }).json()["provider"]["id"]
    monkeypatch.setattr(llm_mod.httpx, "post", lambda url, **kw: Resp(status, text='{"error":"x"}'))
    result = client.post(f"/api/providers/{pid}/test", json={}).json()
    assert result["status"] == expected


def test_test_provider_reports_timeout(client, monkeypatch):
    pid = client.post("/api/providers", json={
        "name": "Slow", "base_url": "https://slow.example/v1", "api_key": "k", "model": "m",
    }).json()["provider"]["id"]

    def boom(url, **kw):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(llm_mod.httpx, "post", boom)
    result = client.post(f"/api/providers/{pid}/test", json={}).json()
    assert result["ok"] is False
    assert result["status"] in ("timeout", "error")


def test_draft_test_before_saving(client, monkeypatch):
    """The Add form can test before Add is pressed."""
    monkeypatch.setattr(llm_mod.httpx, "post", lambda url, **kw: _chat_ok("OK"))
    result = client.post("/api/providers/test", json={
        "name": "Draft", "base_url": "https://draft.example/v1", "api_key": "k", "model": "m",
    }).json()
    assert result["ok"] is True
    assert client.get("/api/providers").json()["providers"] == []


def test_provider_without_api_key_can_be_tested(client, monkeypatch):
    pid = client.post("/api/providers", json={
        "name": "Local", "base_url": "http://localhost:11434/v1", "model": "llama3.2",
        "api_key_required": False, "auth_type": "none", "provider_type": "local",
    }).json()["provider"]["id"]

    def fake_post(url, **kwargs):
        assert "Authorization" not in (kwargs.get("headers") or {})
        return _chat_ok("OK")

    monkeypatch.setattr(llm_mod.httpx, "post", fake_post)
    assert client.post(f"/api/providers/{pid}/test", json={}).json()["ok"] is True


# ----------------------------------------------------------------------
# model discovery + manual entry
# ----------------------------------------------------------------------
def test_model_discovery_populates_and_caches(client, monkeypatch):
    pid = client.post("/api/providers", json={
        "name": "Disc", "base_url": "https://disc.example/v1", "api_key": "k",
    }).json()["provider"]["id"]

    monkeypatch.setattr(discovery_mod.httpx, "get",
                        lambda url, **kw: Resp(200, {"data": [{"id": "m-1"}, {"id": "m-2"}]}))
    result = client.post(f"/api/providers/{pid}/models", json={}).json()
    assert result["models"] == ["m-1", "m-2"]
    assert result["supported"] is True
    assert client.get(f"/api/providers/{pid}/models").json()["models"] == ["m-1", "m-2"]
    assert "m-2" in client.get("/api/providers").json()["available_models"]


def test_model_discovery_accepts_a_bare_list(client, monkeypatch):
    """Local servers often return a bare JSON array rather than {"data": ...}."""
    pid = client.post("/api/providers", json={
        "name": "Bare", "base_url": "http://localhost:1234/v1", "api_key_required": False,
    }).json()["provider"]["id"]
    monkeypatch.setattr(discovery_mod.httpx, "get", lambda url, **kw: Resp(200, ["a", "b"]))
    assert client.post(f"/api/providers/{pid}/models", json={}).json()["models"] == ["a", "b"]


def test_model_discovery_unsupported_falls_back_to_manual_entry(client, monkeypatch):
    pid = client.post("/api/providers", json={
        "name": "NoDisc", "base_url": "https://nodisc.example/v1", "api_key": "k", "model": "manual-model",
    }).json()["provider"]["id"]

    monkeypatch.setattr(discovery_mod.httpx, "get", lambda url, **kw: Resp(404, text="nope"))
    result = client.post(f"/api/providers/{pid}/models", json={}).json()
    assert result["models"] == [] and result["supported"] is False
    # manual entry still works
    row = client.patch(f"/api/providers/{pid}", json={"model": "typed-by-hand"}).json()["provider"]
    assert row["model"] == "typed-by-hand"


# ----------------------------------------------------------------------
# persistence + env bootstrap
# ----------------------------------------------------------------------
def test_provider_persists_across_restart(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        c.post("/api/providers", json={
            "name": "Persisted", "base_url": "https://p.example/v1", "api_key": "pk", "model": "pm",
        })

    with TestClient(create_app(make_settings(tmp_path))) as c:
        row = next(p for p in c.get("/api/providers").json()["providers"] if p["id"] == "persisted")
        assert row["model"] == "pm"
        assert row["has_key"] is True


def test_env_key_seeds_a_provider_once(tmp_path):
    settings = make_settings(tmp_path, openrouter_api_key="sk-or-legacy")
    with TestClient(create_app(settings)) as c:
        row = next(p for p in c.get("/api/providers").json()["providers"] if p["id"] == "openrouter")
        assert row["has_key"] is True
        assert "sk-or-legacy" not in c.get("/api/providers").text
        # removing it must stick
        c.delete("/api/providers/openrouter")

    with TestClient(create_app(make_settings(tmp_path, openrouter_api_key="sk-or-legacy"))) as c:
        assert c.get("/api/providers").json()["providers"] == []


def test_provider_api_is_not_cacheable(client):
    assert client.get("/api/providers").headers.get("cache-control") == "no-store"


# ----------------------------------------------------------------------
# role-aware model selection
# ----------------------------------------------------------------------
def test_mode_and_role_models_reference_configured_providers(client):
    pid = client.post("/api/providers", json={
        "name": "R", "base_url": "https://r/v1", "api_key": "k", "model": "rm",
    }).json()["provider"]["id"]

    for mode in ("chat", "rag", "search", "research"):
        client.put("/api/providers/mode-models", json={"mode": mode, "provider": pid, "model": "rm"})
    roles = client.put("/api/providers/research-models", json={"fast": "rm", "strong": "rm"}).json()
    assert roles["research_models"] == {"fast": "rm", "strong": "rm"}

    body = client.get("/api/providers").json()
    assert all(body["mode_models"][m]["provider"] == pid for m in ("chat", "rag", "search", "research"))


def test_mode_model_reverts_to_auto_when_provider_removed(client):
    pid = client.post("/api/providers", json={
        "name": "Temp", "base_url": "https://t/v1", "api_key": "k",
    }).json()["provider"]["id"]
    client.put("/api/providers/mode-models", json={"mode": "chat", "provider": pid, "model": "m"})
    assert client.get("/api/providers").json()["mode_models"]["chat"]["provider"] == pid

    client.delete(f"/api/providers/{pid}")
    assert client.get("/api/providers").json()["mode_models"]["chat"] == {"provider": "", "model": ""}


def test_mode_model_rejects_bad_input(client):
    assert client.put("/api/providers/mode-models", json={"mode": "nope"}).status_code == 400
    assert client.put(
        "/api/providers/mode-models", json={"mode": "chat", "provider": "ghost"}
    ).status_code == 400
