"""Negative RAG test: an out-of-document question must NOT be answered from
general knowledge."""

from backend.chat.prompts import NO_CONTEXT_MESSAGE

QUERY = "What is the capital of France?"


def test_negative_query_refuses_to_answer_from_knowledge(rag_client, fake_llm):
    resp = rag_client.post("/api/rag/chat", json={"message": QUERY})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == NO_CONTEXT_MESSAGE
    assert body["grounded"] is False
    assert body["context_injected"] is False
    assert body["sources"] == []
    assert body["retrieval"]["performed"] is True
    assert body["retrieval"]["kept"] == 0


def test_negative_query_never_calls_the_llm(rag_client, fake_llm):
    rag_client.post("/api/rag/chat", json={"message": QUERY})
    assert fake_llm.calls == [], "the LLM must not be consulted without grounded context"


def test_negative_answer_does_not_contain_paris(rag_client):
    body = rag_client.post("/api/rag/chat", json={"message": QUERY}).json()
    assert "paris" not in body["answer"].lower()
