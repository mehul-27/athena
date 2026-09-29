"""Normal Chat must bypass RAG entirely: no Chroma retrieval, no sources."""

from backend.chat.prompts import resolve_identity
from tests.conftest import SpyEngine

QUERY = "What is the capital of France?"


def test_chat_answers_normally(client):
    resp = client.post("/api/chat", json={"message": QUERY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "chat"
    assert body["answer"] == "STUB_ANSWER"
    assert body["sources"] == []
    assert body["retrieval"]["performed"] is False


def test_chat_does_not_use_retrieval(client):
    spy = SpyEngine()
    client.app.state.rag_engine = spy
    resp = client.post("/api/chat", json={"message": QUERY})
    assert resp.status_code == 200
    assert spy.queried is False, "normal Chat must not perform Chroma retrieval"


def test_chat_prompt_has_no_document_context(client, fake_llm):
    client.post("/api/chat", json={"message": QUERY})
    prompt = fake_llm.last_text
    assert "Document context" not in prompt
    assert len(fake_llm.calls[-1]) == 2  # system + user only
    assert fake_llm.calls[-1][0]["role"] == "system"
    assert fake_llm.calls[-1][1]["role"] == "user"


def test_chat_system_prompt_names_the_model_when_it_is_the_only_one(client, fake_llm):
    """With a single provider there is no failover, so naming it is truthful."""
    client.post("/api/chat", json={"message": "which model are you"})
    system = fake_llm.calls[-1][0]["content"]
    assert "running on" in system
    assert "stub-model" in system
    assert "which model or AI you are" in system


def test_resolve_identity_is_blank_when_failover_is_possible():
    """Two usable providers -> the request could fall over, so claim nothing."""
    chain = [
        {"name": "groq", "model": "qwen/qwen3.8-27b", "label": "Groq"},
        {"name": "google", "model": "gemini-3.6-flash", "label": "Google Gemini"},
    ]

    class Router:
        def describe(self):
            return chain

    assert resolve_identity(Router()) == ""
    assert resolve_identity(Router(), "google") == ""
    assert resolve_identity(Router(), "ghost") == ""


def test_resolve_identity_names_a_single_provider():
    class Router:
        def describe(self):
            return [{"name": "google", "model": "gemini-3.6-flash", "label": "Google Gemini"}]

    assert resolve_identity(Router()) == "Google Gemini (gemini-3.6-flash)"
    assert resolve_identity(Router(), "google") == "Google Gemini (gemini-3.6-flash)"


def test_no_vendor_claim_reaches_the_model_when_failover_is_possible(tmp_path, monkeypatch):
    """End-to-end: the prompt must not name a provider that might not answer."""
    import backend.providers.llm as llm_mod
    from backend.config import Settings
    from backend.main import create_app
    from fastapi.testclient import TestClient

    settings = Settings(
        _env_file=None, embedding_provider="hashing", chroma_mode="embedded",
        data_dir=str(tmp_path / "data"), documents_dir=str(tmp_path / "docs"),
        chroma_path=str(tmp_path / "chroma"), rag_collection="identity-test",
        nvidia_api_key="", groq_api_key="", openrouter_api_key="", google_api_key="",
    )
    app = create_app(settings)
    registry = app.state.provider_registry
    registry.create(name="Groq", preset="groq",
                    base_url="https://api.groq.com/openai/v1",
                    api_key="gq", model="qwen/qwen3.8-27b")
    registry.create(name="Google Gemini", preset="google",
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                    api_key="gk", model="gemini-3.6-flash")

    seen = {}

    def fake_chat_with_usage(self, messages, **kwargs):
        from backend.providers.llm import CallOutcome

        seen["messages"] = messages
        return CallOutcome(content="hello", input_tokens=3, output_tokens=1, latency_ms=5)

    monkeypatch.setattr(llm_mod.LLMClient, "chat_with_usage", fake_chat_with_usage)

    with TestClient(app) as c:
        body = c.post("/api/chat", json={"message": "which model are you"}).json()

    system = seen["messages"][0]["content"]
    assert body["provider"] == "groq"
    # Neither vendor may be asserted: Google could still serve this request.
    assert "Groq" not in system
    assert "Google" not in system
    assert "gemini" not in system.lower()


def test_resolve_identity_is_blank_without_a_chain():
    class Empty:
        def describe(self):
            return []

    assert resolve_identity(Empty()) == ""
    assert resolve_identity(object()) == ""  # no describe() at all
