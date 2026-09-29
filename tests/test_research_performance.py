"""Deep Research performance: concurrency, call reduction, adaptive stops,
role models and provider fallback. All offline and deterministic.
"""

from __future__ import annotations

import asyncio
import json
import time

import backend.providers.llm as llm_mod
from backend.config import Settings
from backend.providers.presets import GROQ_BASE_URL, NVIDIA_BASE_URL
from backend.providers.providers import ProviderSpec
from backend.providers.router import LLMResult, LLMRouter
from backend.research.engine import DeepResearcher
from backend.research.json_utils import parse_json_array_items
from backend.research.llm import ResearchLLM


class FakeLLM:
    """Deterministic engine LLM: pops replies in order, records every call."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = []

    async def call(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not self.replies:
            raise RuntimeError("FakeLLM ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def finding(url="https://example.test/a", title="A", summary="Evidence"):
    return {"url": url, "title": title, "summary": summary}


# ----------------------------------------------------------------------
# concurrency
# ----------------------------------------------------------------------
def test_independent_searches_run_concurrently():
    engine = DeepResearcher(FakeLLM(), max_rounds=1)
    engine._start_time = time.time()
    state = {"inflight": 0, "max": 0}

    async def fake_search(query):
        state["inflight"] += 1
        state["max"] = max(state["max"], state["inflight"])
        await asyncio.sleep(0.02)
        state["inflight"] -= 1
        return [{"url": f"https://example.test/{query}", "title": query}]

    async def no_extract(url, question, title):
        return None

    engine._search = fake_search
    engine._fetch_and_extract = no_extract
    engine.batch_extraction = False

    asyncio.run(engine._search_and_extract(["a", "b", "c"], "goal"))
    assert state["max"] == 3, "queries must be searched concurrently"


def test_independent_fetches_run_concurrently():
    engine = DeepResearcher(FakeLLM(), max_rounds=1, extraction_concurrency=3)
    engine._start_time = time.time()
    state = {"inflight": 0, "max": 0}

    async def fake_fetch_and_extract(url, question, title):
        state["inflight"] += 1
        state["max"] = max(state["max"], state["inflight"])
        await asyncio.sleep(0.02)
        state["inflight"] -= 1
        return finding(url)

    engine._fetch_and_extract = fake_fetch_and_extract
    rows = [{"url": f"https://example.test/{i}", "title": str(i)} for i in range(3)]
    findings = asyncio.run(engine._extract_individually(rows, "goal"))

    assert len(findings) == 3
    assert state["max"] == 3, "pages must be fetched concurrently"


# ----------------------------------------------------------------------
# batching
# ----------------------------------------------------------------------
def _batch_engine(replies, **over):
    engine = DeepResearcher(FakeLLM(replies), max_rounds=1, batch_extraction=True, extraction_batch_size=3, **over)
    engine._settings = Settings(_env_file=None, search_provider="duckduckgo")
    engine._start_time = time.time()

    async def fake_search(query):
        return [{"url": f"https://example.test/{query}", "title": query}]

    engine._search = fake_search
    engine._fetch_page = lambda url: {
        "url": url,
        "title": url,
        "content": f"content for {url}",
        "og_image": "",
        "success": True,
        "error": "",
    }
    return engine


def test_batched_extraction_uses_one_call_for_many_pages():
    payload = json.dumps([
        {"rational": "r", "evidence": "e1", "summary": "s1"},
        {"rational": "r", "evidence": "e2", "summary": "s2"},
        {"rational": "r", "evidence": "e3", "summary": "s3"},
    ])
    engine = _batch_engine([payload])
    findings = asyncio.run(engine._search_and_extract(["a", "b", "c"], "goal"))

    assert len(engine.llm.calls) == 1, "3 pages must be extracted in a single call"
    assert len(findings) == 3
    assert engine.pages_fetched == 3


def test_batched_extraction_falls_back_to_per_page_on_bad_json():
    good = json.dumps({"rational": "r", "evidence": "e", "summary": "s"})
    # First reply (the batch) is unusable; then one reply per page.
    engine = _batch_engine(["not a json array at all", good, good, good])
    findings = asyncio.run(engine._search_and_extract(["a", "b", "c"], "goal"))

    assert len(engine.llm.calls) == 4, "1 failed batch + 3 per-page fallbacks"
    assert len(findings) == 3


def test_parse_json_array_items_is_tolerant():
    assert parse_json_array_items('[{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]
    assert parse_json_array_items('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json_array_items("prose around {\"a\": 1} here") == [{"a": 1}]
    assert parse_json_array_items("nothing") == []


# ----------------------------------------------------------------------
# avoiding wasted work
# ----------------------------------------------------------------------
def test_synthesis_is_skipped_when_a_round_adds_no_new_findings():
    engine = DeepResearcher(FakeLLM([json.dumps({"sub_questions": []}), "general", "word " * 450]),
                            max_rounds=2, min_rounds=99, max_empty_rounds=5)
    synthesis_calls = {"n": 0}

    async def fake_queries(question, report, round_num):
        return ["q"]

    async def fake_search_switch(queries, question):
        if not called["any"]:
            called["any"] = True
            return [finding()]
        return []

    called = {"any": False}

    async def fake_synth(question, findings, report):
        synthesis_calls["n"] += 1
        return "report"

    engine._generate_queries = fake_queries
    engine._search_and_extract = fake_search_switch
    engine._synthesize = fake_synth

    asyncio.run(engine.research("question"))
    assert synthesis_calls["n"] == 1, "synthesis must not repeat when nothing changed"


def test_adaptive_stop_ends_research_after_min_rounds():
    replies = [
        json.dumps({"sub_questions": []}),   # plan
        "general",                            # category
        '["q"]',                              # round 1 queries
        "report one",                         # round 1 synthesis
        '["q"]',                              # round 2 queries
        "report two",                         # round 2 synthesis
        "YES — sufficient evidence gathered", # stop decision
        "word " * 450,                        # final report
    ]
    engine = DeepResearcher(FakeLLM(replies), max_rounds=5, min_rounds=2, max_empty_rounds=5)

    async def fake_search_and_extract(queries, question):
        return [finding()]

    engine._search_and_extract = fake_search_and_extract
    asyncio.run(engine.research("question"))

    assert engine.round_count == 2, "must stop as soon as the LLM says it has enough"


def test_stagnation_stops_research_early_without_extra_llm_calls():
    engine = DeepResearcher(
        FakeLLM([json.dumps({"sub_questions": []}), "general", "report one", "word " * 450]),
        max_rounds=8,
        min_rounds=99,          # never ask the stop LLM
        max_empty_rounds=99,    # never bail as "search unavailable"
        max_stagnant_rounds=2,
    )
    calls = {"round": 0}

    async def fake_queries(question, report, round_num):
        return ["q"]

    async def fake_search_and_extract(queries, question):
        calls["round"] += 1
        return [finding()] if calls["round"] == 1 else []

    engine._generate_queries = fake_queries
    engine._search_and_extract = fake_search_and_extract

    asyncio.run(engine.research("question"))
    # round 1 adds a finding; rounds 2 and 3 add nothing -> stop at round 3.
    assert engine.round_count == 3


# ----------------------------------------------------------------------
# role models
# ----------------------------------------------------------------------
class CapturingRouter:
    def __init__(self):
        self.models = []

    def chat(self, messages, *, model=None, **kwargs):
        self.models.append(model)
        return LLMResult("ok", "provider-x", model or "default-model", "provider-x")


def test_research_llm_maps_roles_to_models():
    router = CapturingRouter()
    llm = ResearchLLM(router, fast_model="fast-model", strong_model="strong-model")

    async def run():
        await llm.call([{"role": "user", "content": "a"}], role="fast")
        await llm.call([{"role": "user", "content": "b"}], role="strong")
        await llm.call([{"role": "user", "content": "c"}])  # defaults to fast

    asyncio.run(run())
    assert router.models == ["fast-model", "strong-model", "fast-model"]
    assert llm.llm_calls == 3
    assert llm.providers == ["provider-x", "provider-x", "provider-x"]


def test_research_llm_uses_provider_default_when_roles_blank():
    router = CapturingRouter()
    llm = ResearchLLM(router)
    asyncio.run(llm.call([{"role": "user", "content": "a"}], role="strong"))
    assert router.models == [None], "blank roles must not force a model override"


# ----------------------------------------------------------------------
# provider fallback inside research
# ----------------------------------------------------------------------
class Resp:
    def __init__(self, status_code, content=""):
        self.status_code = status_code
        if status_code < 400:
            self._payload = {"choices": [{"message": {"content": content}}]}
        else:
            self._payload = {"error": {"message": "rate limited"}}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def _spec(name, base, key, model):
    return ProviderSpec(name=name, label=name, base_url=base, api_key=key, model=model)


def test_research_llm_falls_back_across_providers(monkeypatch):
    queues = {NVIDIA_BASE_URL: [Resp(429), Resp(429), Resp(429)], GROQ_BASE_URL: [Resp(200, "groq answer")]}

    def fake_post(url, **kwargs):
        for base, queue in queues.items():
            if url.startswith(base):
                return queue.pop(0)
        raise AssertionError(url)

    monkeypatch.setattr(llm_mod.httpx, "post", fake_post)

    router = LLMRouter(
        [
            _spec("nvidia", NVIDIA_BASE_URL, "nv-1", "nvidia/model"),
            _spec("groq", GROQ_BASE_URL, "gq-1", "qwen/qwen3.8-27b"),
        ],
        retry_base_delay=0.0,
    )
    llm = ResearchLLM(router)
    content = asyncio.run(llm.call([{"role": "user", "content": "hi"}], role="fast"))

    assert content == "groq answer"
    assert llm.providers == ["groq"]
    assert router.stats["fallbacks"] == 1


def test_engine_diagnostics_report_usage():
    engine = DeepResearcher(FakeLLM(), max_rounds=1)
    engine.searches = 3
    engine.pages_fetched = 2
    diag = engine.get_diagnostics()
    assert diag["searches"] == 3
    assert diag["pages_fetched"] == 2
    assert diag["rounds"] == 0
