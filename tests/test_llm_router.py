"""Unit tests for the multi-provider LLM router.

All HTTP is mocked — these tests never touch NVIDIA, Groq or OpenRouter, so they
consume no quota and are fully deterministic.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

import backend.providers.llm as llm_mod
from backend.config import Settings
from backend.providers.llm import LLMError
from backend.providers.providers import (
    GROQ_BASE_URL,
    NVIDIA_BASE_URL,
    OPENROUTER_BASE_URL,
    ProviderSpec,
    build_providers,
)
from backend.providers.router import LLMRouter, build_llm_router

NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
GROQ_MODEL = "qwen/qwen3.8-27b"
OPENROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"

NVIDIA_KEY, GROQ_KEY, OPENROUTER_KEY = "nv-secret-key", "gq-secret-key", "or-secret-key"

MESSAGES = [{"role": "user", "content": "What is the primary validation code for Project Lumen?"}]

_DEFAULTS = {
    "nvidia": (NVIDIA_BASE_URL, NVIDIA_MODEL, NVIDIA_KEY, {}),
    "groq": (GROQ_BASE_URL, GROQ_MODEL, GROQ_KEY, {"reasoning_format": "hidden"}),
    "openrouter": (OPENROUTER_BASE_URL, OPENROUTER_MODEL, OPENROUTER_KEY, {}),
}
_BASES = {
    "nvidia": NVIDIA_BASE_URL,
    "groq": GROQ_BASE_URL,
    "openrouter": OPENROUTER_BASE_URL,
}


# ----------------------------------------------------------------------
# fakes
# ----------------------------------------------------------------------
class Resp:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code, content="", reasoning=None, text=None):
        self.status_code = status_code
        if status_code < 400:
            message = {"content": content}
            if reasoning is not None:
                message["reasoning"] = reasoning
            self._payload = {"choices": [{"message": message}]}
        else:
            self._payload = {"error": {"message": "upstream error"}}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeHTTP:
    """Fake httpx.post that routes by base URL and records every call."""

    def __init__(self):
        self.queues = {}
        self.calls = []

    def queue(self, base_url, *responses):
        self.queues[base_url] = list(responses)

    def post(self, url, **kwargs):
        self.calls.append({"url": url, "json": kwargs.get("json"), "headers": kwargs.get("headers")})
        for base, queued in self.queues.items():
            if url.startswith(base):
                if not queued:
                    raise AssertionError(f"unexpected extra request to {url}")
                item = queued.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
        raise AssertionError(f"unexpected request url {url}")

    def count(self, base_url):
        return sum(1 for c in self.calls if c["url"].startswith(base_url))

    def payloads(self, base_url):
        return [c["json"] for c in self.calls if c["url"].startswith(base_url)]

    def headers(self, base_url):
        return [c["headers"] for c in self.calls if c["url"].startswith(base_url)]


@pytest.fixture
def http(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(llm_mod.httpx, "post", fake.post)
    return fake


def spec(name, api_key=None, model=None):
    base, default_model, default_key, extra = _DEFAULTS[name]
    return ProviderSpec(
        name=name,
        label=f"{name} label",
        base_url=base,
        api_key=default_key if api_key is None else api_key,
        model=model or default_model,
        extra_payload=dict(extra),
    )


def make_router(*names, **over):
    names = names or ("nvidia", "groq", "openrouter")
    return LLMRouter([spec(n) for n in names], retry_base_delay=0.0, **over)


# ----------------------------------------------------------------------
# 1. NVIDIA succeeds — nobody else is called
# ----------------------------------------------------------------------
def test_nvidia_success_skips_other_providers(http):
    http.queue(NVIDIA_BASE_URL, Resp(200, "LUMEN-5831-ORBIT"))
    result = make_router().chat(MESSAGES)

    assert result.content == "LUMEN-5831-ORBIT"
    assert result.provider == "nvidia"
    assert result.model == NVIDIA_MODEL
    assert http.count(NVIDIA_BASE_URL) == 1
    assert http.count(GROQ_BASE_URL) == 0
    assert http.count(OPENROUTER_BASE_URL) == 0


# ----------------------------------------------------------------------
# 2/3. retries then fallback
# ----------------------------------------------------------------------
def test_nvidia_429_retries_exactly_max_attempts_then_falls_back(http):
    http.queue(NVIDIA_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))
    result = make_router().chat(MESSAGES)

    assert http.count(NVIDIA_BASE_URL) == 3, "1 initial attempt + 2 retries"
    assert result.provider == "groq"
    assert result.content == "groq answer"


def test_nvidia_503_exhausts_and_groq_answers(http):
    http.queue(NVIDIA_BASE_URL, Resp(503), Resp(503), Resp(503))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))
    result = make_router().chat(MESSAGES)

    assert http.count(NVIDIA_BASE_URL) == 3
    assert result.provider == "groq"
    assert http.count(OPENROUTER_BASE_URL) == 0


# ----------------------------------------------------------------------
# 4. missing Groq key — skipped, OpenRouter used
# ----------------------------------------------------------------------
def test_missing_groq_key_is_skipped(http):
    http.queue(NVIDIA_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(OPENROUTER_BASE_URL, Resp(200, "openrouter answer"))

    router = LLMRouter(
        [spec("nvidia"), spec("groq", api_key=""), spec("openrouter")],
        retry_base_delay=0.0,
    )
    result = router.chat(MESSAGES)

    assert result.provider == "openrouter"
    assert http.count(GROQ_BASE_URL) == 0, "Groq must not be attempted without a key"
    assert router.skipped_names == ["groq"]


# ----------------------------------------------------------------------
# 5/6. deeper fallbacks
# ----------------------------------------------------------------------
def test_groq_success_means_openrouter_never_called(http):
    http.queue(NVIDIA_BASE_URL, Resp(503), Resp(503), Resp(503))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))
    result = make_router().chat(MESSAGES)

    assert result.provider == "groq"
    assert http.count(OPENROUTER_BASE_URL) == 0


def test_openrouter_answers_when_nvidia_and_groq_fail(http):
    http.queue(NVIDIA_BASE_URL, Resp(503), Resp(503), Resp(503))
    http.queue(GROQ_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(OPENROUTER_BASE_URL, Resp(200, "openrouter answer"))
    result = make_router().chat(MESSAGES)

    assert result.provider == "openrouter"
    assert http.count(NVIDIA_BASE_URL) == 3
    assert http.count(GROQ_BASE_URL) == 3
    assert http.count(OPENROUTER_BASE_URL) == 1


# ----------------------------------------------------------------------
# 7. everything fails
# ----------------------------------------------------------------------
def test_all_providers_fail_raises_clean_llm_error(http):
    for base in (NVIDIA_BASE_URL, GROQ_BASE_URL, OPENROUTER_BASE_URL):
        http.queue(base, Resp(503), Resp(503), Resp(503))

    with pytest.raises(LLMError) as excinfo:
        make_router().chat(MESSAGES)

    message = str(excinfo.value)
    assert "All configured LLM providers failed" in message
    for key in (NVIDIA_KEY, GROQ_KEY, OPENROUTER_KEY):
        assert key not in message


def test_no_provider_configured_raises_clear_error():
    router = LLMRouter([spec("nvidia", api_key=""), spec("groq", api_key="")])
    with pytest.raises(LLMError) as excinfo:
        router.chat(MESSAGES)
    assert "No LLM provider is configured" in str(excinfo.value)


# ----------------------------------------------------------------------
# 8/9. permanent errors are not retried
# ----------------------------------------------------------------------
def test_401_is_not_retried_and_falls_back(http):
    http.queue(NVIDIA_BASE_URL, Resp(401, text='{"error":"invalid api key"}'))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))
    result = make_router().chat(MESSAGES)

    assert http.count(NVIDIA_BASE_URL) == 1, "401 must not be retried"
    assert result.provider == "groq"


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_permanent_statuses_are_not_retried(http, status):
    http.queue(NVIDIA_BASE_URL, Resp(status))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))
    result = make_router().chat(MESSAGES)

    assert http.count(NVIDIA_BASE_URL) == 1
    assert result.provider == "groq"


# ----------------------------------------------------------------------
# 10. transient statuses are retried
# ----------------------------------------------------------------------
@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_transient_statuses_are_retried(http, status):
    http.queue(NVIDIA_BASE_URL, Resp(status), Resp(200, "recovered"))
    result = make_router().chat(MESSAGES)

    assert http.count(NVIDIA_BASE_URL) == 2
    assert result.provider == "nvidia"
    assert result.content == "recovered"


# ----------------------------------------------------------------------
# 11/12. transport errors are retried
# ----------------------------------------------------------------------
def test_timeout_is_retried(http):
    http.queue(NVIDIA_BASE_URL, httpx.ReadTimeout("timed out"), Resp(200, "recovered"))
    result = make_router().chat(MESSAGES)

    assert http.count(NVIDIA_BASE_URL) == 2
    assert result.provider == "nvidia"


def test_connection_error_is_retried(http):
    http.queue(NVIDIA_BASE_URL, httpx.ConnectError("connection refused"), Resp(200, "recovered"))
    result = make_router().chat(MESSAGES)

    assert http.count(NVIDIA_BASE_URL) == 2
    assert result.provider == "nvidia"


def test_transport_error_exhausts_then_falls_back(http):
    http.queue(NVIDIA_BASE_URL, *[httpx.ReadTimeout("nope") for _ in range(3)])
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))
    result = make_router().chat(MESSAGES)

    assert http.count(NVIDIA_BASE_URL) == 3
    assert result.provider == "groq"


# ----------------------------------------------------------------------
# 13/14. credentials never leak
# ----------------------------------------------------------------------
def test_api_keys_never_appear_in_logs(http, caplog):
    http.queue(NVIDIA_BASE_URL, Resp(429), Resp(200, "ok"))
    http.queue(GROQ_BASE_URL, Resp(503), Resp(200, "ok"))
    with caplog.at_level(logging.DEBUG):
        make_router().chat(MESSAGES)

    log = caplog.text
    for key in (NVIDIA_KEY, GROQ_KEY, OPENROUTER_KEY):
        assert key not in log
    assert "Bearer" not in log
    # observability: the provider and model must be visible
    assert "provider=nvidia" in log
    assert f"model={NVIDIA_MODEL}" in log


def test_api_keys_never_appear_in_exceptions(http):
    # Simulate upstreams that echo the credential back in their error bodies.
    for base, key in (
        (NVIDIA_BASE_URL, NVIDIA_KEY),
        (GROQ_BASE_URL, GROQ_KEY),
        (OPENROUTER_BASE_URL, OPENROUTER_KEY),
    ):
        echoed = Resp(500, text=json.dumps({"error": f"rejected {key}"}))
        http.queue(base, echoed, echoed, echoed)

    with pytest.raises(LLMError) as excinfo:
        make_router().chat(MESSAGES)

    message = str(excinfo.value)
    assert "All configured LLM providers failed" in message
    for key in (NVIDIA_KEY, GROQ_KEY, OPENROUTER_KEY):
        assert key not in message


def test_upstream_echoed_key_is_scrubbed_from_error(http):
    echoed = Resp(500, text=json.dumps({"error": f"rejected {NVIDIA_KEY}"}))
    http.queue(NVIDIA_BASE_URL, echoed, echoed, echoed)

    with pytest.raises(LLMError) as excinfo:
        LLMRouter([spec("nvidia")], retry_base_delay=0.0).chat(MESSAGES)

    assert NVIDIA_KEY not in str(excinfo.value)
    assert "***" in str(excinfo.value)


# ----------------------------------------------------------------------
# 15/16/17. exact model identifiers and endpoints
# ----------------------------------------------------------------------
def test_nvidia_uses_its_own_endpoint_and_model(http):
    http.queue(NVIDIA_BASE_URL, Resp(200, "ok"))
    make_router().chat(MESSAGES)

    call = http.calls[0]
    assert call["url"] == f"{NVIDIA_BASE_URL}/chat/completions"
    assert call["json"]["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_groq_uses_its_own_endpoint_and_model(http):
    http.queue(NVIDIA_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(GROQ_BASE_URL, Resp(200, "ok"))
    make_router().chat(MESSAGES)

    assert http.payloads(GROQ_BASE_URL)[0]["model"] == "qwen/qwen3.8-27b"
    assert http.calls[3]["url"] == f"{GROQ_BASE_URL}/chat/completions"


def test_openrouter_uses_its_own_endpoint_and_model(http):
    http.queue(NVIDIA_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(GROQ_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(OPENROUTER_BASE_URL, Resp(200, "ok"))
    make_router().chat(MESSAGES)

    assert http.payloads(OPENROUTER_BASE_URL)[0]["model"] == "nvidia/nemotron-3.5-lightning:free"
    assert http.calls[6]["url"] == f"{OPENROUTER_BASE_URL}/chat/completions"


# ----------------------------------------------------------------------
# 18. provider-specific model configuration
# ----------------------------------------------------------------------
def test_provider_specific_models_are_configurable(http):
    http.queue(NVIDIA_BASE_URL, Resp(200, "ok"))
    router = LLMRouter([spec("nvidia", model="nvidia/custom-model")], retry_base_delay=0.0)
    router.chat(MESSAGES)

    assert http.payloads(NVIDIA_BASE_URL)[0]["model"] == "nvidia/custom-model"


def test_build_providers_reads_models_and_keys_from_settings():
    settings = Settings(
        _env_file=None,
        athena_llm_providers="nvidia,groq,openrouter",
        nvidia_api_key="nv",
        athena_nvidia_model="nvidia/custom-nv",
        groq_api_key="gq",
        athena_groq_model="qwen/custom-qwen",
        openrouter_api_key="or",
        athena_openrouter_model="vendor/custom-or",
    )
    specs = build_providers(settings)
    assert [s.name for s in specs] == ["nvidia", "groq", "openrouter"]
    assert [s.model for s in specs] == ["nvidia/custom-nv", "qwen/custom-qwen", "vendor/custom-or"]


def test_base_urls_are_the_documented_ones():
    settings = Settings(_env_file=None)
    specs = {s.name: s for s in build_providers(settings)}
    assert specs["nvidia"].base_url == "https://integrate.api.nvidia.com/v1"
    assert specs["groq"].base_url == "https://api.groq.com/openai/v1"
    assert specs["openrouter"].base_url == "https://openrouter.ai/api/v1"
    assert specs["nvidia"].model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert specs["groq"].model == "qwen/qwen3.8-27b"
    assert specs["openrouter"].model == "nvidia/nemotron-3.5-lightning:free"


# ----------------------------------------------------------------------
# 19. ATHENA_LLM_PROVIDERS controls the order
# ----------------------------------------------------------------------
def test_default_provider_order_is_groq_first():
    """Groq leads by default: it answers in ~1s vs ~150s for the Nemotron models."""
    specs = build_providers(Settings(_env_file=None))
    assert [s.name for s in specs] == ["groq", "nvidia", "openrouter"]


def test_blank_provider_list_falls_back_to_default_order():
    specs = build_providers(Settings(_env_file=None, athena_llm_providers=""))
    assert [s.name for s in specs] == ["groq", "nvidia", "openrouter"]


def test_default_order_groq_answers_without_touching_nvidia(http):
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))
    router = build_llm_router(
        Settings(
            _env_file=None,
            nvidia_api_key="nv", groq_api_key="gq", openrouter_api_key="or",
        )
    )
    assert router.provider_names == ["groq", "nvidia", "openrouter"]

    result = router.chat(MESSAGES)
    assert result.provider == "groq"
    assert http.count(NVIDIA_BASE_URL) == 0
    assert http.count(OPENROUTER_BASE_URL) == 0


def test_provider_order_from_env_is_respected():
    settings = Settings(
        _env_file=None,
        athena_llm_providers="openrouter,groq,nvidia",
        nvidia_api_key="nv", groq_api_key="gq", openrouter_api_key="or",
    )
    assert [s.name for s in build_providers(settings)] == ["openrouter", "groq", "nvidia"]


def test_provider_order_change_changes_which_provider_answers(http):
    http.queue(OPENROUTER_BASE_URL, Resp(200, "openrouter first"))
    router = LLMRouter([spec("openrouter"), spec("nvidia")], retry_base_delay=0.0)
    result = router.chat(MESSAGES)

    assert result.provider == "openrouter"
    assert http.count(NVIDIA_BASE_URL) == 0


def test_unknown_provider_names_are_ignored():
    settings = Settings(_env_file=None, athena_llm_providers="nvidia,bogus,groq")
    assert [s.name for s in build_providers(settings)] == ["nvidia", "groq"]


def test_duplicate_provider_names_are_deduplicated():
    settings = Settings(_env_file=None, athena_llm_providers="groq,groq,nvidia")
    assert [s.name for s in build_providers(settings)] == ["groq", "nvidia"]


# ----------------------------------------------------------------------
# 20. a missing optional key must not crash the app
# ----------------------------------------------------------------------
def test_app_starts_and_answers_with_only_one_key(http):
    settings = Settings(
        _env_file=None,
        athena_llm_providers="nvidia,groq,openrouter",
        nvidia_api_key="",
        groq_api_key="",
        openrouter_api_key="or-only",
    )
    router = build_llm_router(settings)
    assert router.provider_names == ["openrouter"]
    assert router.skipped_names == ["nvidia", "groq"]

    http.queue(OPENROUTER_BASE_URL, Resp(200, "answer"))
    assert router.chat(MESSAGES).content == "answer"


def test_legacy_llm_api_key_still_reaches_openrouter(http):
    settings = Settings(_env_file=None, llm_api_key="legacy-or-key")
    router = build_llm_router(settings)
    assert router.provider_names == ["openrouter"]

    http.queue(OPENROUTER_BASE_URL, Resp(200, "answer"))
    assert router.chat(MESSAGES).content == "answer"


# ----------------------------------------------------------------------
# 21. reasoning is never exposed as the answer
# ----------------------------------------------------------------------
def test_reasoning_is_not_returned_as_content(http):
    http.queue(
        NVIDIA_BASE_URL,
        Resp(200, content="LUMEN-5831-ORBIT", reasoning="Here's a thinking process: the user asks..."),
    )
    result = make_router().chat(MESSAGES)

    assert result.content == "LUMEN-5831-ORBIT"
    assert "thinking process" not in result.content


def test_empty_content_with_reasoning_only_falls_back(http):
    http.queue(NVIDIA_BASE_URL, Resp(200, content="", reasoning="only reasoning here"))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))
    result = make_router().chat(MESSAGES)

    assert result.provider == "groq"
    assert "only reasoning here" not in result.content


def test_groq_hides_reasoning_via_reasoning_format(http):
    http.queue(NVIDIA_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(GROQ_BASE_URL, Resp(200, "ok"))
    make_router().chat(MESSAGES)

    assert http.payloads(GROQ_BASE_URL)[0]["reasoning_format"] == "hidden"


def test_content_parts_are_joined(http):
    class Parts(Resp):
        def __init__(self):
            self.status_code = 200
            self._payload = {
                "choices": [{"message": {"content": [{"text": "LUMEN-"}, {"text": "5831-ORBIT"}]}}]
            }
            self.text = ""

    http.queue(NVIDIA_BASE_URL, Parts())
    assert make_router().chat(MESSAGES).content == "LUMEN-5831-ORBIT"


# ----------------------------------------------------------------------
# observability
# ----------------------------------------------------------------------
def test_fallback_and_success_are_logged(http, caplog):
    http.queue(NVIDIA_BASE_URL, Resp(429), Resp(429), Resp(429))
    http.queue(GROQ_BASE_URL, Resp(200, "groq answer"))

    with caplog.at_level(logging.INFO):
        make_router().chat(MESSAGES)

    log = caplog.text
    assert "provider=nvidia failed status=429; falling back to groq" in log
    assert "LLM success provider=groq" in log


def test_describe_reports_active_chain():
    router = LLMRouter([spec("nvidia"), spec("groq", api_key="")], retry_base_delay=0.0)
    assert router.provider_names == ["nvidia"]
    assert router.skipped_names == ["groq"]
    assert router.describe() == [
        {"name": "nvidia", "model": NVIDIA_MODEL, "label": "nvidia label"}
    ]
    assert router.configured is True
