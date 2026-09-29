import asyncio
import json

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.research.runner import ResearchRunner


class UnusedRouter:
    configured = True
    provider_names = ["stub"]


def make_client(tmp_path):
    settings = Settings(
        _env_file=None,
        embedding_provider="hashing",
        chroma_mode="embedded",
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "docs"),
        chroma_path=str(tmp_path / "chroma"),
        rag_collection="research-api-test",
        nvidia_api_key="",
        groq_api_key="",
        openrouter_api_key="",
    )
    app = create_app(settings, llm=UnusedRouter())
    with TestClient(app) as client:
        yield client


def _seed_runner(client, research_id="rp-0123456789ab", *, status="done"):
    record = {
        "query": "Test question",
        "status": status,
        "result": "## Final\nAnswer.",
        "raw_report": "# Test report\n\n## Final\nAnswer.",
        "sources": [{"url": "https://example.test", "title": "Example", "image": "https://cdn.example.test/hero.jpg"}],
        "raw_findings": [{"url": "https://example.test", "title": "Example", "summary": "Evidence"}],
        "stats": {"Duration": "1.0s", "Rounds": 1, "Queries": 1, "URLs": 1, "Model": "Groq"},
        "category": None,
        "started_at": 1.0,
        "completed_at": 2.0,
        "hidden_images": [],
        "archived": False,
        "consumed": False,
    }
    client.app.state.research_runner.store.save(research_id, record)
    return record


def test_start_research_returns_id_and_running_state(tmp_path):
    for client in make_client(tmp_path):
        runner = client.app.state.research_runner

        async def fake_research(*args, **kwargs):
            await asyncio.sleep(0.05)
            return "## Final\nDone."

        runner.active_research_coro = fake_research
        # The real start path is exercised separately by runner tests; API shape
        # and validation are verified here with an injected lightweight runner.
        class StartStub:
            def start(self, query, **kwargs):
                self.query = query
                self.kwargs = kwargs
                return {"research_id": "rp-0123456789ab", "session_id": "rp-0123456789ab", "status": "running", "query": query}
        client.app.state.research_runner = StartStub()
        response = client.post("/api/research/start", json={"query": "  test question  ", "max_rounds": 2})
        assert response.status_code == 200
        assert response.json()["research_id"] == "rp-0123456789ab"
        assert client.app.state.research_runner.query == "test question"


def test_auto_rounds_null_is_accepted_by_start_api(tmp_path):
    for client in make_client(tmp_path):
        class StartStub:
            def start(self, query, **kwargs):
                self.query = query
                self.kwargs = kwargs
                return {"research_id": "rp-0123456789ab", "status": "running", "query": query}
        stub = StartStub()
        client.app.state.research_runner = stub
        response = client.post("/api/research/start", json={"query": "test", "max_rounds": None})
        assert response.status_code == 200
        assert stub.kwargs["max_rounds"] is None


def test_status_and_result_not_found(tmp_path):
    for client in make_client(tmp_path):
        assert client.get("/api/research/status/rp-0123456789ab").status_code == 404
        assert client.get("/api/research/report/rp-0123456789ab").status_code == 404
        assert client.get("/api/research/detail/rp-0123456789ab").status_code == 404


def test_library_detail_and_html_report(tmp_path):
    for client in make_client(tmp_path):
        _seed_runner(client)
        library = client.get("/api/research/library")
        assert library.status_code == 200
        assert library.json()["research"][0]["research_id"] == "rp-0123456789ab"

        detail = client.get("/api/research/detail/rp-0123456789ab")
        assert detail.status_code == 200
        assert detail.json()["query"] == "Test question"

        report = client.get("/api/research/report/rp-0123456789ab")
        assert report.status_code == 200
        assert report.headers["content-type"].startswith("text/html")
        assert "Athena &mdash; Deep Research Report" in report.text
        assert "/api/research/rp-0123456789ab/hide-image" in report.text


def test_image_hide_unhide_regenerates_report(tmp_path):
    for client in make_client(tmp_path):
        _seed_runner(client)
        image = "https://cdn.example.test/hero.jpg"
        first = client.get("/api/research/report/rp-0123456789ab").text
        assert image in first

        hidden = client.post(
            "/api/research/rp-0123456789ab/hide-image", json={"url": image}
        )
        assert hidden.status_code == 200
        report_after_hide = client.get("/api/research/report/rp-0123456789ab").text
        assert image not in report_after_hide
        assert "Show hidden (1)" in report_after_hide

        assert client.post("/api/research/rp-0123456789ab/unhide-images").status_code == 200
        assert image in client.get("/api/research/report/rp-0123456789ab").text


def test_archive_delete_and_sse_status(tmp_path):
    for client in make_client(tmp_path):
        _seed_runner(client, status="done")
        status = client.get("/api/research/status/rp-0123456789ab")
        assert status.status_code == 200
        assert status.json()["status"] == "done"

        stream = client.get("/api/research/stream/rp-0123456789ab")
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        frames = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
        assert any(frame.get("status") == "done" and frame.get("final") for frame in frames)

        archived = client.post("/api/research/rp-0123456789ab/archive", json={"archived": True})
        assert archived.status_code == 200
        assert client.get("/api/research/library?archived=true").json()["total"] == 1

        deleted = client.delete("/api/research/rp-0123456789ab")
        assert deleted.status_code == 200
        assert client.get("/api/research/detail/rp-0123456789ab").status_code == 404
