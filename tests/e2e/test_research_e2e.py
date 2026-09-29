"""Opt-in real-provider Deep Research E2E.

The LLM calls are real; search and fetch are mocked to keep source content
controlled, deterministic and independent of external web availability.
Enable with ATHENA_E2E=1.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.main import create_app

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(
    __import__("os").getenv("ATHENA_E2E") != "1",
    reason="set ATHENA_E2E=1 to call a real provider",
)]


@pytest.fixture
def real_research_client(tmp_path, monkeypatch):
    from backend.research import engine as engine_module

    settings = get_settings()
    if not settings.llm_configured:
        pytest.skip("Configure at least one LLM provider key in .env")
    settings = settings.model_copy(update={
        "data_dir": str(tmp_path / "data"),
        "documents_dir": str(tmp_path / "documents"),
        "chroma_path": str(tmp_path / "chroma"),
        "research_data_dir": str(tmp_path / "research"),
        "research_max_rounds": 1,
        "research_min_rounds": 1,
        "research_max_time": 180,
        "research_run_timeout_seconds": 300,
        "research_search_provider": "mock",
    })

    monkeypatch.setattr(engine_module, "provider_chain", lambda _provider: ["mock"])
    monkeypatch.setattr(
        engine_module,
        "call_provider",
        lambda provider, query, count, **kwargs: [{
            "title": "Project Lumen Technical Briefing",
            "url": "https://research-fixture.example/lumen",
            "snippet": "Project Lumen primary validation code.",
        }],
    )
    monkeypatch.setattr(
        engine_module.DeepResearcher,
        "_fetch_page",
        lambda self, url: {
            "url": url,
            "title": "Project Lumen Technical Briefing",
            "content": (
                "Project Lumen validation. The primary validation code for Project Lumen "
                "is LUMEN-5831-ORBIT. Every validation gateway request must include this code."
            ),
            "og_image": "https://research-fixture.example/lumen-cover.jpg",
            "success": True,
            "error": "",
        },
    )

    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def test_real_llm_research_round_synthesis_sources_and_visual_report(real_research_client):
    client = real_research_client
    started = client.post(
        "/api/research/start",
        json={
            "query": "What is the primary validation code for Project Lumen?",
            "max_rounds": 1,
            "search_provider": "mock",
        },
    )
    assert started.status_code == 200, started.text
    research_id = started.json()["research_id"]

    # The SSE endpoint runs until the background job emits its terminal frame.
    stream = client.get(f"/api/research/stream/{research_id}")
    assert stream.status_code == 200
    frames = [
        __import__("json").loads(line[6:])
        for line in stream.text.splitlines()
        if line.startswith("data: ")
    ]
    assert any(frame.get("phase") == "planning" for frame in frames)
    # Progress is coalesced by the 1.5s SSE poll; short intermediate phases may
    # be skipped, but the stream must still deliver its terminal frame.
    assert any(frame.get("final") for frame in frames)

    detail = client.get(f"/api/research/detail/{research_id}")
    assert detail.status_code == 200
    record = detail.json()
    assert record["status"] == "done"
    assert record["round_count"] == 1
    assert record["raw_findings"]
    assert record["sources"]
    assert record["stats"]["Provider"] in {"nvidia", "groq", "openrouter"}
    assert record["stats"]["Model"]
    assert "LUMEN-5831-ORBIT" in record["result"]

    report = client.get(f"/api/research/report/{research_id}")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/html")
    assert "LUMEN-5831-ORBIT" in report.text
    assert "Sources (1)" in report.text
    assert "Project Lumen Technical Briefing" in report.text
    assert "research-fixture.example/lumen-cover.jpg" in report.text

    print(
        f"\nDeep Research answered by provider={record['stats']['Provider']} "
        f"model={record['stats']['Model']}"
    )
