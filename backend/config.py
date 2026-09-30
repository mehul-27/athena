"""Athena configuration.

A single flat pydantic-settings model loaded from `.env`. This borrows the
*pattern* from Odysseus's `src/config.py` (env-driven, typed) but intentionally
drops that app's runtime settings blob, per-user prefs and DB-backed endpoints.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/config.py -> athena/
ATHENA_ROOT = Path(__file__).resolve().parents[1]

# Free OpenRouter model (NVIDIA Nemotron 3.5 Lightning). Override with
# ATHENA_LLM_MODEL (or the older LLM_MODEL).
DEFAULT_LLM_MODEL = "nvidia/nemotron-3.5-lightning:free"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ATHENA_ROOT / ".env"),
        env_file_encoding="utf-8-sig",
        extra="ignore",
        case_sensitive=False,
        validate_by_name=True,
    )

    # --- Application ---
    app_name: str = "Athena"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_env: str = "development"
    log_level: str = "INFO"

    # --- Legacy single-provider config (compatibility only) ---
    # The provider block below owns everything LLM-related. LLM_API_KEY survives
    # because pre-provider-split .env files hold an OpenRouter key in it; the
    # legacy ATHENA_LLM_MODEL / LLM_MODEL names are read by
    # `athena_openrouter_model` below.
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout: float = 60.0

    # --- LLM providers ---
    # Priority order: the first provider that answers wins; each is a fallback
    # for the one before it. Nothing outside the provider layer needs to know.
    # Groq leads because the Nemotron Lightning models on NVIDIA/OpenRouter are
    # slow reasoning models (~150s / ~180s per answer) while Groq answers in ~1s.
    athena_llm_providers: str = "groq,nvidia,openrouter"

    nvidia_api_key: str = ""
    athena_nvidia_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"

    groq_api_key: str = ""
    athena_groq_model: str = "qwen/qwen3.8-27b"

    # OPENROUTER_API_KEY wins; ATHENA_LLM_MODEL / LLM_MODEL are kept as
    # compatibility aliases for installs that predate the provider split.
    openrouter_api_key: str = ""
    athena_openrouter_model: str = Field(
        default=DEFAULT_LLM_MODEL,
        validation_alias=AliasChoices(
            "ATHENA_OPENROUTER_MODEL", "ATHENA_LLM_MODEL", "LLM_MODEL"
        ),
    )

    # Google Gemini (OpenAI-compatible endpoint). GEMINI_API_KEY is accepted as
    # an alias for installs that use Google's own preferred variable name.
    # Google retires pinned models for new users, so keep this on a current one.
    google_api_key: str = Field(default="", validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    athena_google_model: str = "gemini-3.6-flash"

    # --- ChromaDB ---
    chroma_mode: str = "embedded"  # embedded | http
    chroma_path: str = ""
    chroma_host: str = "localhost"
    chroma_port: int = 8100
    rag_collection: str = "athena_documents"

    # --- RAG ---
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    rag_relevance_threshold: float = 0.25

    # --- Embeddings ---
    embedding_provider: str = "fastembed"  # fastembed | http | hashing
    embedding_url: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_batch_size: int = 16
    embedding_max_chars: int = 900
    fastembed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fastembed_cache_path: str = ""

    # --- Storage ---
    data_dir: str = ""
    documents_dir: str = ""
    upload_max_bytes: int = 25 * 1024 * 1024

    # --- Search ---
    search_provider: str = "duckduckgo"
    searxng_instance: str = "http://localhost:8080"
    search_result_count: int = 5
    brave_api_key: str = ""
    tavily_api_key: str = ""
    serper_api_key: str = ""
    google_pse_key: str = ""
    google_pse_cx: str = ""
    web_fetch_user_agent: str = DEFAULT_USER_AGENT
    web_fetch_max_bytes: int = 2_000_000
    web_fetch_timeout: float = 10.0
    search_fetch_pages: int = 3
    search_answer: bool = True

    # --- Deep Research ---
    research_data_dir: str = ""
    research_max_rounds: int = 20
    research_max_time: int = 300
    research_max_urls_per_round: int = 3
    research_max_content_chars: int = 15000
    research_max_report_tokens: int = 16384
    research_extraction_timeout_seconds: int = 90
    research_planning_timeout_seconds: int = 90
    research_query_timeout_seconds: int = 120
    research_extraction_concurrency: int = 3
    # Minimum rounds before the LLM is allowed to decide the research is done.
    # Kept deliberately small: with a large floor (the old value was 18) even a
    # quick question burned ~18 rounds of search + extraction + synthesis before
    # the stop check could fire. The engine still won't stop on a single
    # convincing source — the LLM stop decision sees the whole report — this just
    # lets it stop *early* once the evidence is genuinely sufficient.
    research_min_rounds: int = 2
    research_max_empty_rounds: int = 2
    # Rounds that fetch pages but add no new findings before stopping early.
    research_max_stagnant_rounds: int = 2
    # Batch independent page extractions into one LLM call (N pages -> 1 call)
    # instead of one call per page. Single-page rounds are unaffected.
    research_batch_extraction: bool = True
    research_extraction_batch_size: int = 3
    research_synthesis_window: int = 10
    research_run_timeout_seconds: int = 1800
    research_search_provider: str = ""
    # Role models for the research engine. Blank = whatever model the active
    # provider is configured with (no change from today's behaviour). Set these
    # (in .env or Settings) to run cheap control calls on a small/fast model and
    # the final report on a stronger one.
    research_fast_model: str = ""
    research_strong_model: str = ""

    # --- MCP (Model Context Protocol) ---
    # Athena ships no bundled MCP server: the user adds their own (stdio, SSE or
    # streamable HTTP) in Settings, and this is the master switch. `ATHENA_MCP=0`
    # turns the whole subsystem off — no connections, no schemas, no tools.
    mcp_enabled: bool = True
    # Odysseus uses 20s for a per-server connect; kept identical so a slow stdio
    # server behaves the same way here.
    mcp_connect_timeout: float = 20.0
    # Tool-use budget for one chat message. Odysseus allows 50 rounds; a chat
    # message that needs more than a handful is a runaway, and every round is a
    # real (often metered) LLM call.
    tools_max_rounds: int = 6
    tools_max_calls: int = 12
    # A tool result is fed back to the model, so it is capped the way Odysseus
    # caps formatted tool output.
    tools_result_max_chars: int = 8000

    # --- Tests ---
    athena_e2e: int = 0

    def resolved_research_data_dir(self) -> Path:
        return Path(self.research_data_dir) if self.research_data_dir else self.resolved_data_dir() / "research"

    # ------------------------------------------------------------------
    # Resolved paths (kept as methods, not fields, so env stays optional)
    # ------------------------------------------------------------------
    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir) if self.data_dir else ATHENA_ROOT / "data"

    def resolved_documents_dir(self) -> Path:
        return Path(self.documents_dir) if self.documents_dir else ATHENA_ROOT / "documents"

    def resolved_chroma_path(self) -> Path:
        return Path(self.chroma_path) if self.chroma_path else self.resolved_data_dir() / "chroma"

    def resolved_fastembed_cache(self) -> Path:
        return (
            Path(self.fastembed_cache_path)
            if self.fastembed_cache_path
            else self.resolved_data_dir() / "fastembed_cache"
        )

    def registry_path(self) -> Path:
        return self.resolved_data_dir() / "documents.json"

    def ensure_dirs(self) -> None:
        self.resolved_data_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_documents_dir().mkdir(parents=True, exist_ok=True)
        self.resolved_research_data_dir().mkdir(parents=True, exist_ok=True)
        if self.chroma_mode == "embedded":
            self.resolved_chroma_path().mkdir(parents=True, exist_ok=True)

    @property
    def llm_configured(self) -> bool:
        """True when at least one provider is usable (Settings store or `.env`)."""
        try:
            from backend.providers.store import resolve_specs

            if any(spec.api_key or not spec.api_key_required for spec in resolve_specs(self)):
                return True
        except Exception:  # pragma: no cover - defensive, keeps boot resilient
            pass
        # `.env` bootstrap values still count, before the store has been seeded.
        from backend.providers.providers import build_providers

        return any(spec.api_key for spec in build_providers(self))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
