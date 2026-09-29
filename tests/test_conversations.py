"""Conversation persistence and the chat header actions.

Covers the behaviours the Chat UI depends on: a durable conversation, a title
that survives a reload, message counts, usage aggregated onto the conversation,
the research ↔ conversation link, and the destructive actions behaving safely.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.conversations.store import DEFAULT_TITLE


# ────────────────────────── creation + title ──────────────────────────
def test_conversation_starts_as_new_chat(client):
    conv = client.post("/api/conversations", json={}).json()["conversation"]
    assert conv["title"] == DEFAULT_TITLE == "New Chat"
    assert conv["message_count"] == 0
    assert conv["cost_display"] == "$—"      # nothing used yet, nothing claimed


def test_chat_creates_a_conversation_and_titles_it(client):
    res = client.post("/api/chat", json={"message": "Explain the Lumen roadmap for next quarter"}).json()
    conv = res["conversation"]
    assert conv["id"]
    # Title derives from the first message — the user never has to name it.
    assert conv["title"] == "Explain the Lumen roadmap for next…"
    assert conv["message_count"] == 2        # user + assistant
    assert conv["request_count"] == 1


def test_title_is_not_rewritten_on_later_messages(client):
    first = client.post("/api/chat", json={"message": "first question about widgets"}).json()
    cid = first["conversation"]["id"]
    second = client.post("/api/chat", json={"message": "second unrelated question", "conversation_id": cid}).json()
    assert second["conversation"]["title"] == first["conversation"]["title"]
    assert second["conversation"]["message_count"] == 4


def test_rename_persists_across_requests(client):
    cid = client.post("/api/conversations", json={"title": "New Chat"}).json()["conversation"]["id"]
    renamed = client.patch(f"/api/conversations/{cid}", json={"title": "Project Lumen Validation Inquiry"})
    assert renamed.status_code == 200
    assert renamed.json()["conversation"]["title"] == "Project Lumen Validation Inquiry"

    # A fresh read (not just the response body) must show it.
    again = client.get(f"/api/conversations/{cid}").json()["conversation"]
    assert again["title"] == "Project Lumen Validation Inquiry"
    listed = client.get("/api/conversations").json()["conversations"]
    assert any(c["id"] == cid and c["title"] == "Project Lumen Validation Inquiry" for c in listed)


def test_rename_rejects_blank_title(client):
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    # An empty string fails validation; whitespace-only is rejected by the store.
    assert client.patch(f"/api/conversations/{cid}", json={"title": ""}).status_code == 422
    assert client.patch(f"/api/conversations/{cid}", json={"title": "   "}).status_code == 400
    # And the title is unchanged after both rejections.
    assert client.get(f"/api/conversations/{cid}").json()["conversation"]["title"] == "New Chat"


def test_unknown_conversation_is_404(client):
    assert client.get("/api/conversations/does-not-exist").status_code == 404


# ────────────────────────── messages + usage ──────────────────────────
def test_messages_and_usage_come_back_with_the_conversation(settings, fake_llm):
    from backend.main import create_app

    fake_llm.input_tokens = 120
    fake_llm.output_tokens = 30
    fake_llm.cost_usd = 0.000123
    fake_llm.cost_known = True
    app = create_app(settings, llm=fake_llm)
    with TestClient(app) as c:
        res = c.post("/api/chat", json={"message": "hello there"}).json()
        cid = res["conversation"]["id"]
        data = c.get(f"/api/conversations/{cid}").json()

    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
    assert data["messages"][1]["content"] == "STUB_ANSWER"
    assert data["messages"][1]["provider"] == "stub"
    assert data["messages"][1]["model"] == "stub-model"
    conv = data["conversation"]
    assert conv["total_tokens"] == 150
    assert conv["cost_known"] is True
    assert conv["cost_display"].startswith("$0.0001")


def test_usage_keeps_provider_and_model(client):
    client.post("/api/chat", json={"message": "which model are you"})
    cid = client.get("/api/conversations").json()["conversations"][0]["id"]
    conv = client.get(f"/api/conversations/{cid}").json()["conversation"]
    assert conv["by_model"][0]["provider"] == "stub"
    assert conv["by_model"][0]["model"] == "stub-model"
    assert conv["by_model"][0]["requests"] == 1


def test_header_reports_unknown_cost_honestly(client):
    """The stub reports no price -> the header shows $—, not a made-up number."""
    res = client.post("/api/chat", json={"message": "hi"}).json()
    conv = res["conversation"]
    assert conv["cost_known"] is False
    assert conv["cost_display"] == "$—"


def test_history_is_sent_from_the_server_not_the_client(client, fake_llm):
    first = client.post("/api/chat", json={"message": "my name is Ada"}).json()
    cid = first["conversation"]["id"]
    client.post("/api/chat", json={"message": "what is my name?", "conversation_id": cid,
                                   "history": [{"role": "user", "content": "IGNORED CLIENT HISTORY"}]})
    joined = "\n".join(m["content"] for m in fake_llm.calls[-1])
    assert "my name is Ada" in joined
    assert "IGNORED CLIENT HISTORY" not in joined


# ────────────────────────── transcript (Copy Chat / PDF) ──────────────────────────
def test_transcript_contains_both_turns(client):
    res = client.post("/api/chat", json={"message": "summarise the Lumen brief"}).json()
    cid = res["conversation"]["id"]
    text = client.get(f"/api/conversations/{cid}/transcript").text
    assert "summarise the Lumen brief" in text
    assert "STUB_ANSWER" in text
    assert "**User**" in text and "**Athena**" in text


def test_transcript_is_empty_safe(client):
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    text = client.get(f"/api/conversations/{cid}/transcript").text
    assert "New Chat" in text


# ────────────────────────── compact ──────────────────────────
def test_compact_needs_enough_messages(client):
    res = client.post("/api/chat", json={"message": "one"}).json()
    cid = res["conversation"]["id"]
    resp = client.post(f"/api/conversations/{cid}/compact")
    assert resp.status_code == 400
    assert "Not enough messages" in resp.json()["detail"]


def test_compact_summarises_and_archives_without_losing_messages(client):
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    for i in range(5):
        client.post("/api/chat", json={"message": f"question {i}", "conversation_id": cid})
    before = len(client.get(f"/api/conversations/{cid}").json()["messages"])
    assert before == 10

    res = client.post(f"/api/conversations/{cid}/compact")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["summarized"] > 0
    assert body["archived"] == body["summarized"]

    after = client.get(f"/api/conversations/{cid}").json()
    # The active thread is now summary + recent tail…
    assert len(after["messages"]) < before
    assert after["messages"][0]["role"] == "system"
    assert "[Conversation summary]" in after["messages"][0]["content"]

    # …but nothing was destroyed: the archived originals are still stored.
    store = client.app.state.conversation_store
    everything = store.messages(cid, include_compacted=True)
    assert len(everything) == before + 1


def test_compaction_summary_reaches_later_prompts(client, fake_llm):
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    for i in range(5):
        client.post("/api/chat", json={"message": f"question {i}", "conversation_id": cid})
    client.post(f"/api/conversations/{cid}/compact")
    client.post("/api/chat", json={"message": "follow-up", "conversation_id": cid})
    joined = "\n".join(m["content"] for m in fake_llm.calls[-1])
    assert "[Conversation summary]" in joined


def test_compact_records_its_own_usage(client):
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    for i in range(5):
        client.post("/api/chat", json={"message": f"q{i}", "conversation_id": cid})
    before = client.get(f"/api/conversations/{cid}").json()["conversation"]["request_count"]
    client.post(f"/api/conversations/{cid}/compact")
    after = client.get(f"/api/conversations/{cid}").json()["conversation"]["request_count"]
    assert after == before + 1     # the summarisation call is usage too


# ────────────────────────── save to documents ──────────────────────────
def test_save_to_documents_persists_and_is_retrievable(client):
    res = client.post("/api/chat", json={"message": "draft the Lumen summary"}).json()
    cid = res["conversation"]["id"]

    saved = client.post(f"/api/conversations/{cid}/save", json={})
    assert saved.status_code == 200, saved.text
    doc = saved.json()["document"]
    assert doc["id"]

    listed = client.get("/api/saved-documents").json()["documents"]
    assert any(d["id"] == doc["id"] for d in listed)

    content = client.get(f"/api/saved-documents/{doc['id']}").text
    assert "draft the Lumen summary" in content
    assert "STUB_ANSWER" in content


def test_saved_document_can_be_deleted(client):
    cid = client.post("/api/chat", json={"message": "save me"}).json()["conversation"]["id"]
    doc = client.post(f"/api/conversations/{cid}/save", json={}).json()["document"]
    assert client.delete(f"/api/saved-documents/{doc['id']}").status_code == 200
    assert client.get(f"/api/saved-documents/{doc['id']}").status_code == 404


# ────────────────────────── delete ──────────────────────────
def test_delete_removes_conversation_messages_and_usage(client):
    cid = client.post("/api/chat", json={"message": "delete me"}).json()["conversation"]["id"]
    assert client.delete(f"/api/conversations/{cid}").status_code == 200
    assert client.get(f"/api/conversations/{cid}").status_code == 404
    store = client.app.state.conversation_store
    assert store.messages(cid, include_compacted=True) == []
    assert store.usage(cid)["totals"]["requests"] == 0


def test_delete_keeps_saved_documents(client):
    """Deleting a chat must not destroy the document saved out of it."""
    cid = client.post("/api/chat", json={"message": "keep my document"}).json()["conversation"]["id"]
    doc = client.post(f"/api/conversations/{cid}/save", json={}).json()["document"]

    client.delete(f"/api/conversations/{cid}")
    listed = client.get("/api/saved-documents").json()["documents"]
    assert any(d["id"] == doc["id"] for d in listed)
    assert d_id_in(listed, doc["id"])


def d_id_in(listed, doc_id):
    """The orphaned document keeps its content but drops the conversation link."""
    row = next(d for d in listed if d["id"] == doc_id)
    return row["conversation_id"] is None


def test_delete_never_touches_rag_documents(rag_client):
    before = rag_client.get("/api/rag/documents").json()["documents"]
    assert before
    cid = rag_client.post("/api/conversations", json={}).json()["conversation"]["id"]
    rag_client.delete(f"/api/conversations/{cid}")
    after = rag_client.get("/api/rag/documents").json()["documents"]
    assert len(after) == len(before)


def test_delete_removes_the_conversation_from_the_list(client):
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    client.delete(f"/api/conversations/{cid}")
    assert all(c["id"] != cid for c in client.get("/api/conversations").json()["conversations"])


# ────────────────────────── Deep Research ↔ conversation ──────────────────────────
def test_research_start_accepts_a_conversation_id(client):
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    res = client.post("/api/research/start", json={"query": "Lumen architecture", "conversation_id": cid})
    assert res.status_code == 200, res.text
    body = res.json()
    rid = body["research_id"]
    assert rid
    assert body["conversation_id"] == cid
    # The association is on the live job (it is persisted when the run finishes).
    assert client.app.state.research_runner.active[rid]["conversation_id"] == cid


def test_research_without_a_conversation_still_starts(client):
    res = client.post("/api/research/start", json={"query": "standalone question"})
    assert res.status_code == 200
    assert res.json()["conversation_id"] is None
    assert client.app.state.research_runner.active[res.json()["research_id"]]["conversation_id"] is None


def test_research_result_is_posted_into_the_conversation(client):
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    runner = client.app.state.research_runner

    # Simulate a finished job (the runner posts the card on completion).
    runner._record_in_conversation("rp-test0001", {
        "conversation_id": cid,
        "query": "NVIDIA Blackwell architecture",
        "status": "done",
        "sources": [{"url": "https://example.invalid/a"}, {"url": "https://example.invalid/b"}],
        "round_count": 8,
        "stats": {"Rounds": 8, "Provider": "groq", "Model": "qwen/qwen3.8-27b", "Duration": "3m"},
    })

    messages = client.get(f"/api/conversations/{cid}").json()["messages"]
    card = messages[-1]
    assert card["role"] == "assistant"
    assert "Deep Research completed" in card["content"]
    assert card["meta"]["kind"] == "research"
    assert card["meta"]["rounds"] == 8
    assert card["meta"]["sources"] == 2
    assert card["meta"]["report_url"] == "/api/research/report/rp-test0001"


def test_research_card_uses_real_metadata_not_invented_counts(client):
    """A job that reported no rounds must not get a fabricated round count."""
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    client.app.state.research_runner._record_in_conversation("rp-test0002", {
        "conversation_id": cid, "query": "sparse job", "status": "done",
        "sources": [], "stats": {},
    })
    card = client.get(f"/api/conversations/{cid}").json()["messages"][-1]
    assert card["meta"]["rounds"] == 0
    assert card["meta"]["sources"] == 0


def test_research_is_not_posted_when_it_has_no_conversation(client):
    before = len(client.get("/api/conversations").json()["conversations"])
    client.app.state.research_runner._record_in_conversation("rp-orphan", {
        "conversation_id": None, "query": "standalone", "status": "done", "sources": [], "stats": {},
    })
    assert len(client.get("/api/conversations").json()["conversations"]) == before


def test_deleting_a_conversation_detaches_its_research_instead_of_deleting_it(client):
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    runner = client.app.state.research_runner
    # A finished record, as a completed run would have written it.
    rid = "rp-aaaaaaaaaaaa"
    runner.store.save(rid, {"query": "Lumen", "status": "done", "conversation_id": cid,
                            "sources": [], "result": "the report body"})

    res = client.delete(f"/api/conversations/{cid}").json()
    assert res["detached_research"] >= 1

    record = runner.store.load(rid)
    assert record is not None                   # the report survives
    assert record["conversation_id"] is None    # only the link is dropped
    assert record["result"] == "the report body"


def test_research_llm_calls_contribute_to_conversation_usage(client):
    """Every research LLM call is attributed to the conversation (source=research)."""
    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    sink = client.app.state.research_runner._usage_sink(cid)
    assert sink is not None

    from backend.providers.router import LLMResult

    sink(LLMResult(content="plan", provider="groq", model="qwen/qwen3.8-27b",
                   label="Groq", input_tokens=800, output_tokens=200))
    sink(LLMResult(content="report", provider="groq", model="qwen/qwen3.8-27b",
                   label="Groq", input_tokens=3000, output_tokens=1200))

    conv = client.get(f"/api/conversations/{cid}").json()["conversation"]
    assert conv["request_count"] == 2
    assert conv["total_tokens"] == 5200
    assert conv["by_model"][0]["provider"] == "groq"
    assert conv["by_model"][0]["model"] == "qwen/qwen3.8-27b"

    # The usage rows are tagged as research, so they are distinguishable.
    store = client.app.state.conversation_store
    rows = store.usage(cid)["by_model"]
    assert rows[0]["requests"] == 2


def test_standalone_research_has_no_usage_sink(client):
    """Research with no conversation must not create usage somewhere random."""
    assert client.app.state.research_runner._usage_sink(None) is None


def test_research_llm_records_usage_through_the_adapter(client):
    """The adapter actually invokes the sink on each call."""
    import asyncio

    from backend.research.llm import ResearchLLM

    cid = client.post("/api/conversations", json={}).json()["conversation"]["id"]
    recorded = []
    adapter = ResearchLLM(
        client.app.state.llm,
        usage_sink=lambda result: recorded.append(result),
    )
    asyncio.run(adapter.call([{"role": "user", "content": "plan"}], role="fast"))
    assert len(recorded) == 1
    assert adapter.llm_calls == 1


def test_usage_accumulates_across_chat_and_research(settings, fake_llm):
    """Research LLM calls contribute to the conversation's usage."""
    from backend.main import create_app

    fake_llm.input_tokens = 100
    fake_llm.output_tokens = 20
    fake_llm.cost_usd = 0.0
    fake_llm.cost_known = True
    app = create_app(settings, llm=fake_llm)
    with TestClient(app) as c:
        cid = c.post("/api/chat", json={"message": "hi"}).json()["conversation"]["id"]
        store = app.state.conversation_store
        store.record_usage(cid, provider="groq", model="qwen/qwen3.8-27b",
                           input_tokens=900, output_tokens=400, cost_usd=None,
                           cost_known=False, source="research")
        conv = c.get(f"/api/conversations/{cid}").json()["conversation"]

    assert conv["total_tokens"] == 1420            # 120 chat + 1300 research
    assert conv["request_count"] == 2
    assert {m["model"] for m in conv["by_model"]} == {"stub-model", "qwen/qwen3.8-27b"}
    # The research model has no known price, so the conversation total is unknown.
    assert conv["cost_known"] is False
