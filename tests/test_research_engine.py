import asyncio

from backend.config import Settings
from backend.research.engine import DeepResearcher
from backend.research.prompts import CATEGORY_PROMPTS


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def call(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_round_one_requests_four_broad_queries_and_deduplicates():
    llm = FakeLLM(['["one", "two", "three", "four"]', '["one", "five", "six"]'])
    engine = DeepResearcher(llm, max_rounds=2, max_time=30, min_rounds=3)

    first = asyncio.run(engine._generate_queries("question", "", 1))
    second = asyncio.run(engine._generate_queries("question", "partial", 2))

    assert first == ["one", "two", "three", "four"]
    assert second == ["five", "six"]
    assert "4 focused search queries" in llm.calls[0][0][0]["content"]
    assert "3 focused search queries" in llm.calls[1][0][0]["content"]
    assert "fill gaps" in llm.calls[1][0][0]["content"]


def test_handler_min_round_override_can_be_preserved():
    engine = DeepResearcher(FakeLLM([]), max_rounds=20, min_rounds=18)
    assert engine.min_rounds == 18


def test_engine_constructor_keeps_audited_limits():
    engine = DeepResearcher(FakeLLM([]))
    assert engine.max_rounds == 8
    assert engine.max_time == 300
    assert engine.max_urls_per_round == 3
    assert engine.max_content_chars == 15000
    assert engine.max_report_tokens == 8192
    assert engine.extraction_concurrency == 3
    assert engine.max_empty_rounds == 2
    assert engine.synthesis_window == 10


def test_synthesis_uses_only_last_findings_window():
    llm = FakeLLM(["updated report"])
    engine = DeepResearcher(llm, synthesis_window=2)
    findings = [
        {"url": f"https://example.test/{i}", "title": str(i), "summary": f"finding-{i}"}
        for i in range(4)
    ]
    report = asyncio.run(engine._synthesize("q", findings, "old report"))
    prompt = llm.calls[0][0][0]["content"]
    assert report == "updated report"
    assert "finding-0" not in prompt
    assert "finding-1" not in prompt
    assert "finding-2" in prompt and "finding-3" in prompt


def test_synthesis_failure_keeps_previous_report():
    llm = FakeLLM([RuntimeError("provider failed")])
    engine = DeepResearcher(llm)
    assert asyncio.run(engine._synthesize("q", [{"summary": "f"}], "previous")) == "previous"


def test_search_provider_override_precedes_settings(monkeypatch):
    import backend.research.engine as module

    engine = DeepResearcher(FakeLLM([]), search_provider="brave")
    engine._settings = Settings(_env_file=None, search_provider="duckduckgo")
    calls = []

    async def fake_to_thread(fn, name, query, count, *, settings):
        calls.append((name, query, count))
        return [{"url": "https://source.test", "title": "Source"}]

    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(module, "provider_chain", lambda provider: [provider])
    assert asyncio.run(engine._search("topic"))[0]["url"] == "https://source.test"
    assert calls == [("brave", "topic", 10)]
    assert engine.providers_used == ["brave"]


def test_search_disabled_returns_empty():
    engine = DeepResearcher(FakeLLM([]), search_provider="disabled")
    engine._settings = Settings(_env_file=None)
    assert asyncio.run(engine._search("topic")) == []


def test_empty_round_with_no_findings_reports_search_unavailable():
    llm = FakeLLM(['{"sub_questions":[],"key_topics":[],"success_criteria":"complete"}', "general", '["one query"]'])
    engine = DeepResearcher(llm, max_rounds=2, max_empty_rounds=1)
    engine._search_and_extract = lambda queries, question: asyncio.sleep(0, result=[])
    result = asyncio.run(engine.research("question"))
    assert "Search unavailable" in result


def test_category_constants_match_audited_prompt_categories():
    assert set(CATEGORY_PROMPTS) == {"product", "comparison", "howto", "factcheck"}


def test_cancellation_is_cooperative_at_round_boundary():
    llm = FakeLLM(['{"sub_questions":[],"key_topics":[],"success_criteria":"done"}', "product"])
    events = []
    engine = DeepResearcher(llm, max_rounds=4, progress_callback=events.append)
    engine.cancel()
    result = asyncio.run(engine.research("question"))
    assert result == "No information could be gathered for this question."
    assert engine.round_count == 1
