"""Opt-in end-to-end suite that calls the REAL configured LLM provider.

Disabled by default so CI stays deterministic and offline. Enable with:
    set ATHENA_E2E=1
    set LLM_API_KEY=sk-or-...
    pytest tests/e2e -m e2e

It exercises the complete RAG pipeline with a real model: upload -> retrieval ->
context injection -> generated answer, plus the negative case.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.main import create_app
from tests.conftest import LUMEN_PDF

pytestmark = pytest.mark.e2e

_ENABLED = os.getenv("ATHENA_E2E") == "1"


@pytest.fixture(scope="module")
def real_client():
    settings = get_settings()
    if not settings.llm_configured:
        pytest.skip("LLM_API_KEY / ATHENA_LLM_MODEL not configured")
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


@pytest.mark.skipif(not _ENABLED, reason="set ATHENA_E2E=1 to run real-LLM tests")
def test_lumen_end_to_end(real_client):
    data = LUMEN_PDF.read_bytes()
    up = real_client.post(
        "/api/rag/documents",
        files={"file": ("Project Lumen.pdf", data, "application/pdf")},
    )
    assert up.status_code == 200, up.text

    resp = real_client.post(
        "/api/rag/chat",
        json={"message": "What is the primary validation code for Project Lumen?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["context_injected"] is True
    assert "LUMEN-5831-ORBIT" in body["answer"], body["answer"]
    assert any(s["filename"] == "Project Lumen.pdf" for s in body["sources"])
    # The answer must come from a real provider in the configured chain.
    assert body["provider"] in {"nvidia", "groq", "openrouter"}, body["provider"]
    print(
        f"\nE2E answered by provider={body['provider']} "
        f"model={body['model']} ({body['provider_label']})"
    )


@pytest.mark.skipif(not _ENABLED, reason="set ATHENA_E2E=1 to run real-LLM tests")
def test_negative_end_to_end(real_client):
    resp = real_client.post("/api/rag/chat", json={"message": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is False
    assert "paris" not in body["answer"].lower()
    # Retrieval below threshold must short-circuit before any provider call.
    assert body["provider"] is None, "negative RAG must not call an LLM provider"
