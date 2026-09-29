"""Usage + cost tracking.

The rules under test are the ones this task is strict about: cost comes from the
provider's own token counts and a real price, a genuinely free model is `$0.00`,
and an unknown price is reported as unknown (`$—`) rather than invented.
"""

from __future__ import annotations

from backend.providers.llm import LLMClient
from backend.providers.pricing import estimate_cost, format_cost, is_free_model
from backend.providers.router import LLMResult


# ────────────────────────── pricing rules ──────────────────────────
def test_free_model_is_zero_not_unknown():
    cost, known = estimate_cost("meta-llama/llama-3.3-70b-instruct:free", 10_000, 2_000)
    assert known is True
    assert cost == 0.0
    assert format_cost(cost) == "$0.0000"


def test_local_endpoint_is_never_billed():
    cost, known = estimate_cost("gpt-4o", 1_000_000, 1_000_000, is_local=True)
    assert (cost, known) == (0.0, True)


def test_unknown_model_reports_no_cost():
    cost, known = estimate_cost("qwen/qwen3.8-27b", 5_000, 1_000)
    assert cost is None
    assert known is False
    assert format_cost(cost) == "$—"


def test_known_model_matches_longest_key():
    """`gpt-4o-mini` must not be billed at `gpt-4o` rates."""
    mini, _ = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    full, _ = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert mini is not None and full is not None
    assert mini < full
    assert round(mini, 6) == 0.75   # 0.15 + 0.60


def test_cost_scales_with_tokens():
    cost, known = estimate_cost("gpt-4o", 1_000_000, 0)
    assert known and round(cost, 7) == 2.50


def test_no_tokens_is_zero_cost_for_a_priced_model():
    cost, known = estimate_cost("gpt-4o", 0, 0)
    assert (cost, known) == (0.0, True)


def test_is_free_model_detection():
    assert is_free_model("x/y:free")
    assert not is_free_model("x/y")
    assert not is_free_model("")


# ────────────────────────── provider usage capture ──────────────────────────
class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


def test_usage_is_read_from_the_provider_response(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        return _FakeResponse({
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        })

    monkeypatch.setattr("backend.providers.llm.httpx.post", fake_post)

    client = LLMClient(
        base_url="https://example.invalid/v1", api_key="k", model="gpt-4o",
        provider="test", auth_type="bearer",
    )
    outcome = client.chat_with_usage([{"role": "user", "content": "hi"}])
    assert outcome.content == "hi"
    assert outcome.input_tokens == 120
    assert outcome.output_tokens == 30
    assert outcome.total_tokens == 150
    assert outcome.has_usage


def test_missing_usage_is_not_invented(monkeypatch):
    monkeypatch.setattr(
        "backend.providers.llm.httpx.post",
        lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": "hi"}}]}),
    )
    client = LLMClient(
        base_url="https://example.invalid/v1", api_key="k", model="gpt-4o",
        provider="test", auth_type="bearer",
    )
    outcome = client.chat_with_usage([{"role": "user", "content": "hi"}])
    assert outcome.input_tokens == 0 and outcome.output_tokens == 0
    assert not outcome.has_usage


def test_malformed_usage_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "backend.providers.llm.httpx.post",
        lambda *a, **k: _FakeResponse({
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": "lots", "completion_tokens": None},
        }),
    )
    client = LLMClient(
        base_url="https://example.invalid/v1", api_key="k", model="gpt-4o",
        provider="test", auth_type="bearer",
    )
    outcome = client.chat_with_usage([{"role": "user", "content": "hi"}])
    assert outcome.input_tokens == 0 and outcome.output_tokens == 0


def test_usage_summary_reports_provider_and_model():
    result = LLMResult(
        content="x", provider="groq", model="qwen/qwen3.8-27b", label="Groq",
        input_tokens=10, output_tokens=5, cost_usd=None, cost_known=False, latency_ms=42,
    )
    summary = result.usage_summary()
    assert summary["provider"] == "groq"
    assert summary["model"] == "qwen/qwen3.8-27b"
    assert summary["total_tokens"] == 15
    assert summary["cost_usd"] is None and summary["cost_known"] is False


# ────────────────────────── conversation aggregation ──────────────────────────
def test_usage_is_aggregated_per_conversation(settings):
    from backend.conversations.store import ConversationStore

    store = ConversationStore(settings)
    conv = store.create()
    store.record_usage(conv["id"], provider="groq", model="qwen/qwen3.8-27b",
                       input_tokens=100, output_tokens=20, cost_usd=None,
                       cost_known=False, source="chat")
    store.record_usage(conv["id"], provider="groq", model="qwen/qwen3.8-27b",
                       input_tokens=50, output_tokens=10, cost_usd=None,
                       cost_known=False, source="research")

    usage = store.usage(conv["id"])
    assert usage["totals"]["requests"] == 2
    assert usage["totals"]["input_tokens"] == 150
    assert usage["totals"]["output_tokens"] == 30
    assert usage["totals"]["total_tokens"] == 180
    # Pricing for this model is unknown -> the total is unknown, not $0.
    assert usage["totals"]["cost_known"] is False
    assert usage["totals"]["cost_usd"] is None
    assert usage["by_model"][0]["provider"] == "groq"
    assert usage["by_model"][0]["requests"] == 2


def test_free_model_usage_aggregates_to_zero_cost(settings):
    from backend.conversations.store import ConversationStore

    store = ConversationStore(settings)
    conv = store.create()
    store.record_usage(conv["id"], provider="openrouter", model="meta-llama/llama-3.3-70b:free",
                       input_tokens=1000, output_tokens=200, cost_usd=0.0, cost_known=True)
    totals = store.usage(conv["id"])["totals"]
    assert totals["cost_known"] is True
    assert totals["cost_usd"] == 0.0
    assert totals["total_tokens"] == 1200      # free != untracked


def test_one_unpriced_request_makes_the_total_unknown(settings):
    from backend.conversations.store import ConversationStore

    store = ConversationStore(settings)
    conv = store.create()
    store.record_usage(conv["id"], provider="a", model="gpt-4o",
                       input_tokens=10, output_tokens=1, cost_usd=0.000025, cost_known=True)
    store.record_usage(conv["id"], provider="b", model="mystery-model",
                       input_tokens=10, output_tokens=1, cost_usd=None, cost_known=False)
    totals = store.usage(conv["id"])["totals"]
    assert totals["cost_known"] is False
    assert totals["cost_usd"] is None
    assert totals["unpriced_requests"] == 1
