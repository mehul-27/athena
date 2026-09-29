import json

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend.providers.router import LLMResult
from backend.research.runner import ResearchRunner


class ScriptedRouter:
    configured = True
    provider_names = ["groq"]
    skipped_names = []

    def __init__(self):
        self.calls = []
        self.sequence = [
            # plan
            json.dumps({"sub_questions": ["code?"], "key_topics": ["validation"], "success_criteria": "identify exact code"}),
            # category
            "general",
            # round 1: broad query, extraction, synthesis, then continue
            '["Project Lumen validation code source"]',
            json.dumps({"rational": "The page covers validation.", "evidence": "Primary validation code is LUMEN-5831-ORBIT.", "summary": "Primary validation code is LUMEN-5831-ORBIT."}),
            "## Validation\nThe primary validation code is LUMEN-5831-ORBIT. [Source](https://source.example.test/lumen)",
            "NO — Gather a second source confirming the gateway requirement.",
            # round 2: targeted follow-up, another extraction and synthesis, then stop
            '["Project Lumen gateway validation requirement"]',
            json.dumps({"rational": "The gateway documentation confirms the same identifier.", "evidence": "Every request must include LUMEN-5831-ORBIT.", "summary": "Gateway requests require LUMEN-5831-ORBIT."}),
            "## Validation\nTwo independent sources confirm LUMEN-5831-ORBIT. [Source](https://source.example.test/lumen) [Gateway](https://source.example.test/gateway)",
            "YES — Two sources agree and answer the question.",
            # final report plus the source's short-report expansion pass
            "# Project Lumen Validation\n\n## Executive Summary\nThe primary validation code is LUMEN-5831-ORBIT.\n\n## Evidence\nThe sources confirm the code. [Source](https://source.example.test/lumen) [Gateway](https://source.example.test/gateway)\n\n## Conclusion\nLUMEN-5831-ORBIT is the code.",
            "# Project Lumen Validation\n\n## Executive Summary\nThe primary validation code is LUMEN-5831-ORBIT.\n\n## Evidence\nTwo independent sources confirm the code. [Source](https://source.example.test/lumen) [Gateway](https://source.example.test/gateway)\n\n## Conclusion\nLUMEN-5831-ORBIT is the code required by the gateway.",
        ]

    @property
    def configured(self):
        return True

    def describe(self):
        return [{"name": "groq", "model": "qwen/qwen3.8-27b", "label": "Groq · Qwen 3.8 27B"}]

    def chat(self, messages, *, temperature=None, max_tokens=None, timeout=None, **kwargs):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, "timeout": timeout})
        content = self.sequence.pop(0)
        return LLMResult(content, "groq", "qwen/qwen3.8-27b", "Groq · Qwen 3.8 27B")


def test_research_golden_path_api_multi_round_synthesis_and_visual_report(tmp_path, monkeypatch):
    import backend.research.engine as engine_module

    search_calls = []
    fetch_calls = []

    def fake_provider(provider, query, count, *, settings, time_filter=None):
        search_calls.append((provider, query, count))
        if "gateway" in query.lower():
            return [{"title": "Lumen Gateway Requirements", "url": "https://source.example.test/gateway", "snippet": "gateway requirement"}]
        return [{"title": "Project Lumen Validation", "url": "https://source.example.test/lumen", "snippet": "validation code"}]

    def fake_fetch(url, *, settings, timeout, max_bytes):
        fetch_calls.append(url)
        gateway = url.endswith("/gateway")
        return {
            "url": url,
            "title": "Lumen Gateway Requirements" if gateway else "Project Lumen Validation",
            "content": "Every gateway request must include LUMEN-5831-ORBIT." if gateway else "The primary validation code for Project Lumen is LUMEN-5831-ORBIT.",
            "og_image": "https://source.example.test/gateway-cover.jpg" if gateway else "https://source.example.test/lumen-cover.jpg",
            "success": True,
            "error": "",
        }

    monkeypatch.setattr(engine_module, "call_provider", fake_provider)
    monkeypatch.setattr(engine_module.DeepResearcher, "_fetch_page", lambda self, url: fake_fetch(
        url, settings=settings, timeout=10, max_bytes=self._settings.web_fetch_max_bytes
    ))

    settings = Settings(
        _env_file=None,
        embedding_provider="hashing",
        chroma_mode="embedded",
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        chroma_path=str(tmp_path / "chroma"),
        rag_collection="research-golden-test",
        nvidia_api_key="",
        groq_api_key="",
        openrouter_api_key="",
        research_data_dir=str(tmp_path / "research"),
        research_max_rounds=2,
        research_min_rounds=1,
        research_max_time=30,
        research_run_timeout_seconds=60,
        research_search_provider="duckduckgo",
    )
    llm = ScriptedRouter()
    app = create_app(settings, llm=llm)
    with TestClient(app) as client:
        started = client.post(
            "/api/research/start",
            json={"query": "What is the primary validation code for Project Lumen?", "max_rounds": 2},
        )
        assert started.status_code == 200
        research_id = started.json()["research_id"]

        runner = app.state.research_runner
        task = runner.active[research_id]["task"]
        async def wait_for_task():
            await task
        client.portal.call(wait_for_task)

        status = client.get(f"/api/research/status/{research_id}").json()
        assert status["status"] == "done"
        result = client.get(f"/api/research/detail/{research_id}").json()
        assert "LUMEN-5831-ORBIT" in result["result"]
        assert result["round_count"] == 2
        assert result["sources"] == [
            {"url": "https://source.example.test/lumen", "title": "Project Lumen Validation", "image": "https://source.example.test/lumen-cover.jpg"},
            {"url": "https://source.example.test/gateway", "title": "Lumen Gateway Requirements", "image": "https://source.example.test/gateway-cover.jpg"},
        ]
        assert len(search_calls) == 2
        assert fetch_calls == ["https://source.example.test/lumen", "https://source.example.test/gateway"]
        assert len(llm.calls) == 12  # plan, classify, 2×(query/extract/synth/stop), final + expansion

        report = client.get(f"/api/research/report/{research_id}")
        assert report.status_code == 200
        assert report.headers["content-type"].startswith("text/html")
        assert "LUMEN-5831-ORBIT" in report.text
        assert "Project Lumen Validation" in report.text
        assert "Sources (2)" in report.text
        assert "https://source.example.test/lumen-cover.jpg" in report.text
