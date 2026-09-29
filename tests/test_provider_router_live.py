"""Router driven live by the provider registry: dynamic providers, failover,
health, and no-restart configuration reload.

All HTTP is mocked — no real provider is contacted.
"""

from __future__ import annotations

import json

import pytest

import backend.providers.llm as llm_mod
from backend.config import Settings
from backend.providers.health import HealthTracker
from backend.providers.llm import LLMError
from backend.providers.presets import GROQ_BASE_URL, NVIDIA_BASE_URL
from backend.providers.registry import ProviderRegistry
from backend.providers.router import build_llm_router
from backend.providers.store import ProviderStore

MESSAGES = [{"role": "user", "content": "hi"}]


class Resp:
    def __init__(self, status_code, content="", text=None):
        self.status_code = status_code
        if status_code < 400:
            self._payload = {"choices": [{"message": {"content": content}}]}
        else:
            self._payload = {"error": {"message": "upstream error"}}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self):
        self.queues = {}
        self.calls = []
        self.payloads = []

    def queue(self, base_url, *responses):
        self.queues[base_url] = list(responses)

    def post(self, url, **kwargs):
        self.calls.append(url)
        self.payloads.append(kwargs.get("json") or {})
        for base, queued in self.queues.items():
            if url.startswith(base):
                if not queued:
                    raise AssertionError(f"unexpected extra request to {url}")
                return queued.pop(0)
        raise AssertionError(f"unexpected request url {url}")

    def count(self, base_url):
        return sum(1 for url in self.calls if url.startswith(base_url))


@pytest.fixture
def http(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(llm_mod.httpx, "post", fake.post)
    return fake


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


def make_registry(tmp_path, **over):
    settings = make_settings(tmp_path, **over)
    registry = ProviderRegistry(settings, store=ProviderStore(settings), health=HealthTracker())
    return settings, registry


def add(registry, preset, base_url, api_key="", model="", name=""):
    """Add a provider the way the Settings UI does."""
    row = registry.create(preset=preset, base_url=base_url, api_key=api_key, model=model, name=name)
    return row["id"]


def test_router_is_empty_until_a_provider_is_configured(tmp_path):
    settings, registry = make_registry(tmp_path)
    router = build_llm_router(settings, registry=registry)
    assert router.configured is False
    with pytest.raises(LLMError, match="No LLM provider is configured"):
        router.chat(MESSAGES)


def test_settings_change_applies_to_the_live_router(tmp_path, http):
    settings, registry = make_registry(tmp_path)
    router = build_llm_router(settings, registry=registry)
    assert router.provider_names == []

    add(registry, "groq", GROQ_BASE_URL, api_key="gq-1", model="qwen/qwen3.8-27b")
    assert router.provider_names == ["groq"]

    add(registry, "nvidia", NVIDIA_BASE_URL, api_key="nv-1")
    # priority is insertion order until the user reorders it
    assert router.provider_names == ["groq", "nvidia"]

    registry.update("nvidia", enabled=False)
    assert router.provider_names == ["groq"]
    registry.update("nvidia", enabled=True)
    assert router.provider_names == ["groq", "nvidia"]

    registry.set_order(["nvidia", "groq"])
    assert router.provider_names == ["nvidia", "groq"]

    http.queue(GROQ_BASE_URL, Resp(200, "from groq"))
    http.queue(NVIDIA_BASE_URL, Resp(200, "from nvidia"))
    assert router.chat(MESSAGES).provider == "nvidia"


def test_rate_limited_first_provider_retries_then_falls_back(tmp_path, http):
    settings, registry = make_registry(tmp_path)
    add(registry, "nvidia", NVIDIA_BASE_URL, api_key="nv-1")
    add(registry, "groq", GROQ_BASE_URL, api_key="gq-1")
    router = build_llm_router(settings, registry=registry)

    http.queue(NVIDIA_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))

    result = router.chat(MESSAGES)
    assert result.provider == "groq"
    assert http.count(NVIDIA_BASE_URL) == 3
    assert router.stats["fallbacks"] == 1
    assert registry.health.get("nvidia").rate_limited() is True


def test_invalid_credentials_are_not_retried(tmp_path, http):
    settings, registry = make_registry(tmp_path)
    add(registry, "nvidia", NVIDIA_BASE_URL, api_key="nv-1")
    add(registry, "groq", GROQ_BASE_URL, api_key="gq-1")
    router = build_llm_router(settings, registry=registry)

    http.queue(NVIDIA_BASE_URL, Resp(401, text='{"error":"invalid key"}'))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))

    assert router.chat(MESSAGES).provider == "groq"
    assert http.count(NVIDIA_BASE_URL) == 1, "401 must not be retried"


def test_all_providers_unavailable_raises_useful_error(tmp_path, http):
    settings, registry = make_registry(tmp_path)
    add(registry, "nvidia", NVIDIA_BASE_URL, api_key="nv-secret")
    add(registry, "groq", GROQ_BASE_URL, api_key="gq-secret")
    router = build_llm_router(settings, registry=registry)

    for base in (NVIDIA_BASE_URL, GROQ_BASE_URL):
        http.queue(base, Resp(503), Resp(503), Resp(503))

    with pytest.raises(LLMError) as excinfo:
        router.chat(MESSAGES)
    message = str(excinfo.value)
    assert "All configured LLM providers failed" in message
    assert "nv-secret" not in message and "gq-secret" not in message


def test_exhausted_provider_is_not_hammered_on_the_next_request(tmp_path, http):
    settings, registry = make_registry(tmp_path)
    add(registry, "nvidia", NVIDIA_BASE_URL, api_key="nv-1")
    add(registry, "groq", GROQ_BASE_URL, api_key="gq-1")
    router = build_llm_router(settings, registry=registry)

    http.queue(NVIDIA_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))
    assert router.chat(MESSAGES).provider == "groq"
    assert http.count(NVIDIA_BASE_URL) == 3

    http.queue(GROQ_BASE_URL, Resp(200, "groq again"))
    assert router.chat(MESSAGES).content == "groq again"
    assert http.count(NVIDIA_BASE_URL) == 3, "exhausted provider must not be hammered again"


def test_preferred_provider_and_model_are_tried_first(tmp_path, http):
    settings, registry = make_registry(tmp_path)
    add(registry, "nvidia", NVIDIA_BASE_URL, api_key="nv-1", model="nvidia/default")
    add(registry, "groq", GROQ_BASE_URL, api_key="gq-1", model="qwen/default")
    router = build_llm_router(settings, registry=registry)

    http.queue(GROQ_BASE_URL, Resp(200, "ok"))
    result = router.chat(MESSAGES, prefer="groq", model="qwen/picked")

    assert result.provider == "groq"
    assert result.fallback is False
    assert http.count(NVIDIA_BASE_URL) == 0
    assert http.payloads[-1]["model"] == "qwen/picked"
    # The result must report the model actually sent, not the configured default.
    assert result.model == "qwen/picked"


def test_preference_still_falls_back_with_its_own_model(tmp_path, http):
    settings, registry = make_registry(tmp_path)
    add(registry, "nvidia", NVIDIA_BASE_URL, api_key="nv-1", model="nvidia/default")
    add(registry, "groq", GROQ_BASE_URL, api_key="gq-1", model="qwen/default")
    router = build_llm_router(settings, registry=registry)

    http.queue(GROQ_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(NVIDIA_BASE_URL, Resp(200, "fallback"))
    result = router.chat(MESSAGES, prefer="groq", model="qwen/picked")

    assert result.provider == "nvidia"
    assert http.payloads[-1]["model"] == "nvidia/default"
    assert result.fallback is True
    assert result.requested_provider == "groq"


def test_removing_a_provider_removes_it_from_the_live_router(tmp_path):
    settings, registry = make_registry(tmp_path)
    pid = add(registry, "openrouter", "https://openrouter.ai/api/v1", api_key="or-1")
    router = build_llm_router(settings, registry=registry)
    assert pid in router.provider_names

    registry.remove(pid)
    assert pid not in router.provider_names


def test_router_reads_the_persistent_store_at_request_time(tmp_path, http):
    """An edit to providers.json is honoured without a restart or API call."""
    settings, registry = make_registry(tmp_path)
    pid = add(registry, "groq", GROQ_BASE_URL, api_key="gq-1", model="qwen/original")
    router = build_llm_router(settings, registry=registry)
    assert router.describe()[0]["model"] == "qwen/original"

    ProviderStore(settings).set_provider(pid, model="qwen/rewritten")

    assert router.describe()[0]["model"] == "qwen/rewritten"
    http.queue(GROQ_BASE_URL, Resp(200, "ok"))
    router.chat(MESSAGES)
    assert http.payloads[-1]["model"] == "qwen/rewritten"


def test_external_add_of_an_unknown_provider_is_picked_up(tmp_path):
    """A provider written straight into providers.json (never in source) works."""
    settings, registry = make_registry(tmp_path)
    router = build_llm_router(settings, registry=registry)
    assert router.provider_names == []

    store = ProviderStore(settings)
    store.create_provider(name="Externally Added", base_url="https://srv.example/v1",
                          api_key="k", model="m")
    assert router.provider_names == ["externally-added"]


def test_external_enable_disable_is_read_at_request_time(tmp_path):
    settings, registry = make_registry(tmp_path)
    nvidia = add(registry, "nvidia", NVIDIA_BASE_URL, api_key="nv-1")
    add(registry, "groq", GROQ_BASE_URL, api_key="gq-1")
    router = build_llm_router(settings, registry=registry)
    assert router.provider_names == ["nvidia", "groq"]

    ProviderStore(settings).set_provider(nvidia, enabled=False)
    assert router.provider_names == ["groq"]
