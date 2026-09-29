"""Unit tests for LLMClient retry behavior.

All HTTP is mocked — these tests never touch OpenRouter.
"""

from __future__ import annotations

import json
import logging

import pytest

import backend.providers.llm as llm_mod
from backend.providers.llm import LLMClient, LLMError

MESSAGES = [{"role": "user", "content": "hi"}]


class FakeResponse:
    def __init__(self, status_code: int, content: str = "", text: str | None = None) -> None:
        self.status_code = status_code
        if status_code < 400:
            self._payload = {"choices": [{"message": {"content": content}}]}
        else:
            self._payload = {"error": {"message": text or "error"}}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


def _client() -> LLMClient:
    return LLMClient(
        base_url="https://example.test/v1",
        api_key="super-secret-key",
        model="nvidia/nemotron-3.5-lightning:free",
        provider="openrouter",
        retry_base_delay=0.0,  # no real sleeping in tests
    )


def _patch(monkeypatch, responses):
    """Return a call counter; the fake post walks through `responses`."""
    state = {"calls": 0}

    def fake_post(url, **kwargs):
        i = state["calls"]
        state["calls"] += 1
        return responses[min(i, len(responses) - 1)]

    monkeypatch.setattr(llm_mod.httpx, "post", fake_post)
    return state


def test_200_success_single_request(monkeypatch):
    state = _patch(monkeypatch, [FakeResponse(200, content="hello")])
    assert _client().chat(MESSAGES) == "hello"
    assert state["calls"] == 1


def test_429_then_200_retries_once(monkeypatch):
    state = _patch(monkeypatch, [FakeResponse(429), FakeResponse(200, content="ok")])
    assert _client().chat(MESSAGES) == "ok"
    assert state["calls"] == 2


def test_two_429_then_200_retries_twice(monkeypatch):
    state = _patch(
        monkeypatch, [FakeResponse(429), FakeResponse(429), FakeResponse(200, content="ok")]
    )
    assert _client().chat(MESSAGES) == "ok"
    assert state["calls"] == 3


def test_three_429_raises_after_max_attempts(monkeypatch):
    state = _patch(monkeypatch, [FakeResponse(429) for _ in range(4)])
    with pytest.raises(LLMError) as excinfo:
        _client().chat(MESSAGES)
    assert "429" in str(excinfo.value)
    assert state["calls"] == 3, "must stop after 3 attempts"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_4xx_is_not_retried(monkeypatch, status):
    state = _patch(monkeypatch, [FakeResponse(status) for _ in range(3)])
    with pytest.raises(LLMError):
        _client().chat(MESSAGES)
    assert state["calls"] == 1, f"HTTP {status} must not be retried"


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_transient_5xx_is_retried_then_succeeds(monkeypatch, status):
    state = _patch(monkeypatch, [FakeResponse(status), FakeResponse(200, content="ok")])
    assert _client().chat(MESSAGES) == "ok"
    assert state["calls"] == 2


def test_retry_logs_are_clear_and_do_not_leak_key(monkeypatch, caplog):
    _patch(monkeypatch, [FakeResponse(429), FakeResponse(200, content="ok")])
    with caplog.at_level(logging.WARNING):
        _client().chat(MESSAGES)
    log = caplog.text
    assert "rate limited" in log
    assert "attempt=1/3" in log
    assert "retrying_in=" in log
    assert "super-secret-key" not in log


def test_final_failure_log_reports_attempts(monkeypatch, caplog):
    _patch(monkeypatch, [FakeResponse(429) for _ in range(3)])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(LLMError):
            _client().chat(MESSAGES)
    assert "attempts=3" in caplog.text
    assert "status=429" in caplog.text


def test_malformed_body_still_raises_llmerror(monkeypatch):
    class BadJSON(FakeResponse):
        def json(self):
            raise ValueError("not json")

    _patch(monkeypatch, [BadJSON(200)])
    with pytest.raises(LLMError):
        _client().chat(MESSAGES)


def test_inline_reasoning_is_stripped_from_the_answer(monkeypatch):
    """Gemma/Gemini inline <thought>…</thought>; it must never reach the user."""
    raw = (
        "<thought>* Question: which model are you\n"
        "    * I am a large language model trained by Google.</thought>"
        "I am a large language model, trained by Google."
    )
    _patch(monkeypatch, [FakeResponse(200, content=raw)])
    assert _client().chat(MESSAGES) == "I am a large language model, trained by Google."


def test_reasoning_only_response_fails_over(monkeypatch):
    """If stripping leaves nothing, that is not an answer."""
    _patch(monkeypatch, [FakeResponse(200, content="<thought>only reasoning</thought>")])
    with pytest.raises(LLMError):
        _client().chat(MESSAGES)
