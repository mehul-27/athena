"""RAG Chat grounding: context is retrieved and injected into the LLM prompt."""

from backend.chat.prompts import GROUNDING_SYSTEM_PROMPT

QUERY = "What is the primary validation code for Project Lumen?"


def test_rag_chat_returns_answer_with_sources(rag_client):
    resp = rag_client.post("/api/rag/chat", json={"message": QUERY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["context_injected"] is True
    assert body["answer"] == "STUB_ANSWER"
    assert body["retrieval"]["performed"] is True
    names = [s["filename"] for s in body["sources"]]
    assert "Project Lumen.pdf" in names


def test_rag_chat_injects_document_context_into_prompt(rag_client, fake_llm):
    rag_client.post("/api/rag/chat", json={"message": QUERY})
    assert fake_llm.calls, "the LLM must have been called"
    prompt = fake_llm.last_text
    # The validation code from the PDF must actually reach the model.
    assert "LUMEN-5831-ORBIT" in prompt
    # And the grounding instruction must be the system message.
    assert fake_llm.calls[-1][0]["role"] == "system"
    assert fake_llm.calls[-1][0]["content"] == GROUNDING_SYSTEM_PROMPT


def test_rag_identity_note_does_not_weaken_the_grounding_contract(rag_client, fake_llm):
    """The grounding system message is unchanged; identity rides in the user turn."""
    rag_client.post("/api/rag/chat", json={"message": QUERY})
    messages = fake_llm.calls[-1]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == GROUNDING_SYSTEM_PROMPT
    assert "running on" in messages[1]["content"]
    assert "LUMEN-5831-ORBIT" in messages[1]["content"]


def test_rag_chat_never_answers_without_context(settings, client, fake_llm):
    # No documents uploaded -> retrieval finds nothing -> LLM must not be called.
    resp = client.post("/api/rag/chat", json={"message": QUERY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is False
    assert body["context_injected"] is False
    assert body["sources"] == []
    assert fake_llm.calls == []
