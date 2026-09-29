"""Shared test fixtures.

Default tests run fully offline and deterministically:
  * embeddings use the `hashing` provider (no model download),
  * ChromaDB is embedded in a per-test temp dir,
  * the LLM is a stub that records the prompts it receives.

The real-LLM suite lives in `tests/e2e/` and is opt-in (see pytest.ini).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import Settings  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.providers.router import LLMResult  # noqa: E402

LUMEN_PDF = ROOT.parent / "docs" / "Project Lumen.pdf"


class FakeLLM:
    """Deterministic stub standing in for the LLM router."""

    def __init__(
        self,
        reply: str = "STUB_ANSWER",
        provider: str = "stub",
        model: str = "stub-model",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float | None = None,
        cost_known: bool = False,
    ) -> None:
        self.reply = reply
        self.provider = provider
        self.model = model
        self.label = f"{provider} · {model}"
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.cost_known = cost_known
        self.calls: list[list[dict]] = []

    @property
    def configured(self) -> bool:
        return True

    @property
    def provider_names(self) -> list[str]:
        return [self.provider]

    @property
    def skipped_names(self) -> list[str]:
        return []

    def describe(self) -> list[dict]:
        return [{"name": self.provider, "model": self.model, "label": self.label}]

    def chat(self, messages, **kwargs) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(
            content=self.reply, provider=self.provider, model=self.model, label=self.label,
            input_tokens=self.input_tokens, output_tokens=self.output_tokens,
            cost_usd=self.cost_usd, cost_known=self.cost_known, latency_ms=12,
        )

    @property
    def last_text(self) -> str:
        """All content from the most recent call, concatenated."""
        if not self.calls:
            return ""
        return "\n".join(m.get("content", "") for m in self.calls[-1])


class SpyEngine:
    """Engine stand-in that fails loudly if retrieval is attempted."""

    def __init__(self) -> None:
        self.queried = False

    def query(self, *args, **kwargs):
        self.queried = True
        raise AssertionError("RAG retrieval must not run on the normal Chat path")

    def count(self) -> int:
        return 0


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        embedding_provider="hashing",
        chroma_mode="embedded",
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        chroma_path=str(tmp_path / "chroma"),
        rag_collection="athena_test",
        rag_relevance_threshold=0.25,
        # Provider chain: only OpenRouter has a key, so it is the active provider.
        llm_api_key="",
        nvidia_api_key="",
        groq_api_key="",
        openrouter_api_key="test-key",
        athena_openrouter_model="stub-model",
        athena_e2e=0,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def app(settings, fake_llm):
    return create_app(settings, llm=fake_llm)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def lumen_bytes() -> bytes:
    assert LUMEN_PDF.exists(), f"missing test fixture: {LUMEN_PDF}"
    return LUMEN_PDF.read_bytes()


@pytest.fixture
def rag_client(app, client, lumen_bytes):
    """Client with Project Lumen already uploaded and indexed."""
    resp = client.post(
        "/api/rag/documents",
        files={"file": ("Project Lumen.pdf", lumen_bytes, "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    return client
