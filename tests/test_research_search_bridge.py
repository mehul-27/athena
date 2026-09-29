from unittest.mock import Mock

import pytest

import search.web as web
from backend.config import Settings


def test_call_provider_uses_exact_named_provider(monkeypatch):
    settings = Settings(_env_file=None, search_provider="disabled")
    caller = Mock(return_value=[{"title": "A", "url": "https://a.test", "snippet": "s"}])
    monkeypatch.setitem(web._CALLERS, "searxng", caller)

    result = web.call_provider("searxng", "query", 10, settings=settings)

    assert result[0]["url"] == "https://a.test"
    caller.assert_called_once_with("query", 10, settings, None)


def test_call_provider_disabled_does_not_call_provider(monkeypatch):
    caller = Mock(return_value=[])
    monkeypatch.setitem(web._CALLERS, "duckduckgo", caller)
    assert web.call_provider("disabled", "query") == []
    caller.assert_not_called()


def test_call_provider_unknown_name_is_explicit():
    with pytest.raises(ValueError, match="Unknown search provider"):
        web.call_provider("not-a-provider", "query", settings=Settings(_env_file=None))


def test_research_provider_chain_reuses_athena_fallback_order():
    assert web.provider_chain("searxng") == ["searxng", "duckduckgo"]
    assert web.provider_chain("disabled") == []
    assert web.provider_chain("") == []
