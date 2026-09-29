"""Chat capabilities: one conversation, optional grounding.

RAG and Web Search are capabilities of Chat, not separate pages. These tests
prove the dispatch reaches the existing pipelines (no duplication) and that
doing nothing still means "no retrieval at all".
"""

from __future__ import annotations

import pytest

from backend.chat import search_chat
from backend.chat.prompts import NO_CONTEXT_MESSAGE
from backend.config import Settings
from backend.main import create_app
from fastapi.testclient import TestClient

QUERY = "What is the primary validation code for Project Lumen?"
OFF_TOPIC = "What is the capital of France?"


def _fake_results():
    return [
        {"title": "NVIDIA Nemotron news", "url": "https://news.example/nemotron",
         "snippet": "Nemotron 3.5 landed this week."},
        {"title": "Follow-up", "url": "https://news.example/follow", "snippet": "More detail."},
    ]


# ----------------------------------------------------------------------
# capability metadata
# ----------------------------------------------------------------------
def test_capabilities_endpoint_lists_only_real_capabilities(client):
    body = client.get("/api/chat/capabilities").json()
    ids = {c["id"] for c in body["capabilities"]}
    assert ids == {"web_search", "rag"}
    combos = [sorted(c) for c in body["supported_combinations"]]
    assert [] in combos
    assert ["rag"] in combos
    assert ["web_search"] in combos


def test_combined_capabilities_are_explicitly_unsupported(client):
    resp = client.post("/api/chat", json={"message": "hi", "capabilities": ["rag", "web_search"]})
    assert resp.status_code == 400
    assert "cannot be enabled together" in resp.json()["detail"]


def test_unknown_capability_is_ignored(client, fake_llm, monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("search must not run for an unknown capability")

    monkeypatch.setattr(search_chat, "web_search", boom)
    resp = client.post("/api/chat", json={"message": "hi", "capabilities": ["mcp", "bogus"]})
    assert resp.status_code == 200
    assert resp.json()["capabilities"] == []


# ----------------------------------------------------------------------
# capabilities OFF -> exactly today's behaviour
# ----------------------------------------------------------------------
def test_normal_chat_makes_no_chroma_query(app, client, monkeypatch):
    from tests.conftest import SpyEngine

    spy = SpyEngine()
    app.state.rag_engine = spy
    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 200
    assert spy.queried is False, "normal Chat must not touch Chroma"


def test_normal_chat_never_calls_search(client, fake_llm, monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("search must not run without the web_search capability")

    monkeypatch.setattr(search_chat, "web_search", boom)
    monkeypatch.setattr(search_chat, "news_search", boom)
    body = client.post("/api/chat", json={"message": "hi"}).json()
    assert body["capabilities"] == []
    assert body["sources"] == []
    assert body["grounded"] is False


def test_normal_chat_response_shape_is_unchanged(client):
    body = client.post("/api/chat", json={"message": "hi"}).json()
    # The pre-existing contract still holds (older clients keep working).
    for key in ("mode", "answer", "sources", "grounded", "provider", "model"):
        assert key in body


# ----------------------------------------------------------------------
# RAG capability
# ----------------------------------------------------------------------
def test_rag_capability_uses_the_existing_rag_pipeline(rag_client, fake_llm):
    body = rag_client.post("/api/chat", json={"message": QUERY, "capabilities": ["rag"]}).json()

    assert body["capabilities"] == ["rag"]
    assert body["mode"] == "rag"
    assert body["grounded"] is True
    assert body["context_injected"] is True
    # The chunk really reached the model, i.e. this is the RAG pipeline.
    assert "LUMEN-5831-ORBIT" in fake_llm.last_text
    # Sources carry the document metadata.
    assert any(s["filename"] == "Project Lumen.pdf" for s in body["sources"])
    assert body["retrieval"]["performed"] is True


def test_rag_capability_refuses_below_threshold(rag_client, fake_llm):
    body = rag_client.post("/api/chat", json={"message": OFF_TOPIC, "capabilities": ["rag"]}).json()
    assert body["capabilities"] == ["rag"]
    assert body["answer"] == NO_CONTEXT_MESSAGE
    assert body["grounded"] is False
    assert body["context_injected"] is False
    assert body["sources"] == []
    assert fake_llm.calls == [], "below-threshold RAG must not call the LLM"


def test_rag_capability_without_documents_still_refuses(client, fake_llm):
    body = client.post("/api/chat", json={"message": QUERY, "capabilities": ["rag"]}).json()
    assert body["mode"] == "rag"
    assert body["grounded"] is False
    assert body["answer"] == NO_CONTEXT_MESSAGE
    assert fake_llm.calls == []


# ----------------------------------------------------------------------
# Web Search capability
# ----------------------------------------------------------------------
def test_web_search_capability_calls_the_search_backend(client, fake_llm, monkeypatch):
    calls = []

    def fake_search(query, **kwargs):
        calls.append(query)
        return _fake_results()

    monkeypatch.setattr(search_chat, "web_search", fake_search)
    body = client.post(
        "/api/chat",
        json={"message": "What happened with NVIDIA Nemotron this week?", "capabilities": ["web_search"]},
    ).json()

    assert calls == ["What happened with NVIDIA Nemotron this week?"]
    assert body["capabilities"] == ["web_search"]
    assert body["mode"] == "search"
    assert body["answer"] == "STUB_ANSWER"
    assert [s["url"] for s in body["sources"]] == [
        "https://news.example/nemotron", "https://news.example/follow",
    ]
    # the search results were injected into the prompt
    assert "Nemotron 3.5 landed this week" in fake_llm.last_text


def test_web_search_capability_does_not_touch_chroma(app, client, monkeypatch):
    from tests.conftest import SpyEngine

    monkeypatch.setattr(search_chat, "web_search", lambda query, **kw: _fake_results())
    spy = SpyEngine()
    app.state.rag_engine = spy
    client.post("/api/chat", json={"message": "news?", "capabilities": ["web_search"]})
    assert spy.queried is False


def test_web_search_with_no_results_reports_no_answer(client, monkeypatch):
    monkeypatch.setattr(search_chat, "web_search", lambda query, **kw: [])
    body = client.post("/api/chat", json={"message": "obscure", "capabilities": ["web_search"]}).json()
    assert body["answer"] is None
    assert body["sources"] == []


# ----------------------------------------------------------------------
# the standalone endpoints still exist and still work
# ----------------------------------------------------------------------
def test_standalone_rag_and_search_endpoints_still_exist(rag_client):
    assert rag_client.post("/api/rag/chat", json={"message": QUERY}).status_code == 200
    assert rag_client.get("/api/rag/documents").status_code == 200
    assert rag_client.post("/api/search", json={"query": "x", "answer": False}).status_code in (200,)


# ----------------------------------------------------------------------
# conversation history (needed for edit → resend)
# ----------------------------------------------------------------------
def test_history_is_forwarded_to_normal_chat(client, fake_llm):
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    client.post("/api/chat", json={"message": "second question", "history": history})
    messages = fake_llm.calls[-1]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "first question"
    assert messages[2]["content"] == "first answer"
    assert messages[3]["content"] == "second question"


def test_history_is_forwarded_into_the_rag_prompt(rag_client, fake_llm):
    history = [{"role": "user", "content": "earlier turn"},
               {"role": "assistant", "content": "earlier reply"}]
    body = rag_client.post(
        "/api/chat",
        json={"message": QUERY, "capabilities": ["rag"], "history": history},
    ).json()
    messages = fake_llm.calls[-1]
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "earlier turn"
    assert body["grounded"] is True


def test_history_is_forwarded_into_the_search_prompt(client, fake_llm, monkeypatch):
    monkeypatch.setattr(search_chat, "web_search", lambda query, **kw: _fake_results())
    client.post("/api/chat", json={
        "message": "follow up", "capabilities": ["web_search"],
        "history": [{"role": "user", "content": "earlier"}],
    })
    messages = fake_llm.calls[-1]
    assert messages[1]["content"] == "earlier"
    assert "Search results" in messages[-1]["content"]


def test_history_is_sanitised_and_capped(client, fake_llm):
    history = [
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": 42},
        {"role": "user", "content": "keep me"},
        "not-a-dict",
    ]
    client.post("/api/chat", json={"message": "q", "history": history})
    messages = fake_llm.calls[-1]
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert messages[1]["content"] == "keep me"

    # A very long thread is trimmed to the last N turns.
    long_history = [{"role": "user", "content": f"turn {i}"} for i in range(60)]
    client.post("/api/chat", json={"message": "q", "history": long_history})
    trimmed = fake_llm.calls[-1]
    assert len(trimmed) == 1 + 24 + 1  # system + capped history + the new message


# ----------------------------------------------------------------------
# UI affordances: stop button + edit
# ----------------------------------------------------------------------
def test_composer_has_a_stop_button(client):
    html = client.get("/").text
    assert 'id="chat-stop"' in html
    assert "Stop" in html


def test_no_separate_rag_or_search_pages(client):
    html = client.get("/").text
    assert 'data-mode="rag"' not in html, "RAG must not be a top-level page"
    assert 'data-mode="search"' not in html, "Search must not be a top-level page"
    assert 'id="panel-rag"' not in html
    assert 'id="panel-search"' not in html
    assert 'data-mode="chat"' in html
    assert 'data-mode="research"' in html, "Research stays a top-level mode"


def test_index_has_capability_bar_and_documents_panel(client):
    html = client.get("/").text
    assert 'class="cap-bar"' in html
    assert 'data-cap="rag"' in html
    assert 'data-cap="web_search"' in html
    assert 'aria-pressed="false"' in html          # toggles start off
    assert 'id="docs-panel"' in html and "hidden" in html  # panel exists, hidden until RAG is on
    assert 'id="model-chat"' in html               # model selector still in Chat


def test_index_does_not_duplicate_the_document_panel(client):
    html = client.get("/").text
    assert html.count('id="docs-list"') == 1
    assert html.count('id="upload-input"') == 1


def test_research_is_a_modal_launched_from_chat(client):
    """Deep Research is a Chat capability, not a top-level page."""
    html = client.get("/").text
    # No top-level Research tab (the per-mode model selector keeps its data-mode).
    assert 'class="tab" role="tab" data-mode="research"' not in html
    assert "Research\n      </button>" not in html
    assert 'id="panel-research"' not in html
    assert 'id="research-modal"' in html           # still the same controls…
    assert 'id="research-form"' in html
    assert 'id="research-jobs"' in html
    assert 'id="research-query"' in html
    assert 'id="open-research"' in html            # …launched from the chat composer
    assert 'class="cap-bar"' in html
    assert 'id="model-research"' in html           # research model selection kept


def test_chat_header_and_menu_exist(client):
    html = client.get("/").text
    assert 'id="chat-header"' in html
    assert 'id="chat-title"' in html
    assert 'id="chat-count"' in html
    assert 'id="chat-cost"' in html
    for action in ("rename", "compact", "copy", "pdf", "save", "delete"):
        assert f'data-action="{action}"' in html
    assert "/static/chatHeader.js" in html


def test_library_pane_is_available(client):
    html = client.get("/").text
    assert 'data-section="library"' in html
