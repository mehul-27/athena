from backend.main import create_app


def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # This app was built with an injected stub router.
    assert body["llm"]["providers"] == ["stub"]
    assert body["llm"]["primary"] == "stub"
    assert body["llm"]["primary_model"] == "stub-model"
    assert body["rag"]["count"] == 0
    assert body["documents"]["documents"] == 0


def test_health_reports_provider_chain(settings):
    """With the real router, health must show the active chain and skips."""
    from fastapi.testclient import TestClient

    from backend.providers.router import build_llm_router

    app = create_app(settings, llm=build_llm_router(settings))
    with TestClient(app) as c:
        body = c.get("/api/health").json()

    assert body["llm"]["providers"] == ["openrouter"]
    assert body["llm"]["skipped"] == ["groq", "nvidia"]
    assert body["llm"]["primary_model"] == "stub-model"


def test_health_reports_chroma_and_embeddings(client):
    body = client.get("/api/health").json()
    assert body["chroma"]["mode"] == "embedded"
    assert body["embedding"]["provider"] == "hashing"
    assert body["rag"]["dimension"] > 0
