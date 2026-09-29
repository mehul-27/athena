# Athena — Odysseus Migration Audit (Phase 0)

Read-only audit of `M:\Athena\odysseus` performed **before** any Athena code was written.
Nothing under `M:\Athena\odysseus` was modified.

- **Reference implementation:** `M:\Athena\odysseus` (Odysseus, v1.0.3)
- **Target:** `M:\Athena\athena` (Athena) — independent, runnable without Odysseus
- **Environment observed:** Python 3.13.3; Odysseus venv already contains `chromadb 1.5.9`,
  `fastembed`, `pypdf`, `numpy`, `httpx`, `fastapi`.
- **Test fixture:** `M:\Athena\docs\Project Lumen.pdf` — verified to contain `LUMEN-5831-ORBIT`
  (1 page, plain extractable text via `pypdf`).

Legend: **REUSE** = copy/adapt nearly as-is · **ADAPT** = reuse logic, rewrite the plumbing ·
**REWRITE** = re-implement cleanly · **SKIP** = do not migrate.

---

## 0. Executive summary

Odysseus's RAG stack is the strongest reuse candidate and is only lightly coupled to the rest of
the app. Its search stack is also largely self-contained. Its **LLM layer (`src/llm_core.py`,
~4000 LOC)** and **request-routing / session architecture** are heavily coupled and should *not*
be copied — Athena gets a small, purpose-built OpenAI-compatible LLM client instead.

The central Athena design rule — **explicit modes, no automatic agent routing** — is exactly what
Odysseus violates: `chat route → RAG → agent-intent → agent_loop → LLM`. Athena's RAG path stops at
`RAG API → retriever → context → LLM`.

---

## 1. RAG (vector store + hybrid retrieval)

**Odysseus source files (inspected directly):**
- `src/rag_vector.py` — `VectorRAG`: the real engine. Chunking, add/delete, hybrid search, stats.
- `src/rag_manager.py` — thin back-compat wrapper (`RAGManager`) over `VectorRAG`.
- `src/rag_singleton.py` — process-wide lazy singleton + retry throttle.
- `src/chroma_client.py` — singleton **HTTP** ChromaDB client (standalone service on `:8100`).
- `src/embeddings.py` — `EmbeddingClient` (OpenAI-compatible HTTP `/v1/embeddings`) + `FastEmbedClient`
  (local ONNX), with HTTP→local fallback and a "HTTP endpoint is down" process latch.
- `src/embedding_lanes.py` — dual-lane scheme (`custom` + `fastembed`) so different embedding
  dimensions never share one Chroma collection; includes lane migration/reset/backfill.
- `src/personal_docs.py` — `extract_pdf_text`, `split_chunks`, keyword fallback, dir indexing.
- `src/constants.py` — paths, `COLLECTION_NAME`, chunk defaults live in `personal_docs.PersonalDocsConfig`.

**Dependencies traced:**
- Third-party: `chromadb` (`chromadb-client`, HTTP), `fastembed`, `numpy`, `pypdf`, `httpx`.
- Internal: `src.constants`, `src.index_walk` (`prune_index_dirs`, `is_indexable_file`),
  `src.runtime_paths`, `src.secret_storage` (only for decrypting a saved embedding API key).

**What to reuse:**
- The **chunking algorithm** in `VectorRAG._split_into_chunks` (sentence-boundary aware, size 1000 /
  overlap 200) and `personal_docs.split_chunks` (fixed overlap, tail-dedup fix).
- The **hybrid scoring** formula: `0.7 * (1 - distance) + 0.3 * keyword_overlap` (`VECTOR_WEIGHT` /
  `KEYWORD_WEIGHT`), plus cosine `hnsw:space`.
- The **content-addressed id** scheme `doc_<sha256(owner\x00text)[:16]>` — makes re-indexing
  idempotent and de-duplicates identical chunks (required by the "avoid duplicate indexing" rule).
- The **delete-by-source** path-boundary logic (exact `source` match, never substring) — prevents
  orphan vectors and over-deletion.
- The **keyword fallback** `_keyword_search_fallback` when the vector query fails.

**What to rewrite / adapt:**
- Collapse the **multi-lane** design to a single lane for Athena (one collection, one embedding
  model). The lane machinery exists to keep a user-configured HTTP model separate from the
  FastEmbed fallback; Athena's `.env` picks *one* embedding backend at startup, so lanes add
  complexity without benefit. Keep the fingerprint-in-collection-metadata idea as a safety check.
- Replace the **HTTP-only ChromaDB client** with an embedded `chromadb.PersistentClient` (default)
  so Athena is runnable with no external service, while still allowing an HTTP endpoint via env.
- Drop `rag_manager.py` / `rag_singleton.py` wrappers; use the engine directly.
- Drop `migrate_legacy_collection` (exists to upgrade Odysseus's older unsuffixed collection).

**Do NOT migrate:**
- `owner`/`rename_owner`/`remove_directory` multi-tenant metadata sweeps (no multi-tenancy in
  Athena milestone 1). Keep `owner` as an *optional* metadata field only.
- `index_personal_documents`'s directory walker (Odysseus indexes arbitrary dirs; Athena ingests
  uploads). Keep the walker code in reserve for a later "add folder" feature.

**Config required:** `CHROMA_MODE` (`embedded`|`http`), `CHROMA_PATH`, `CHROMA_HOST`, `CHROMA_PORT`,
`RAG_COLLECTION`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_RELEVANCE_THRESHOLD`.

**Tests that exist (Odysseus):** `test_rag_vector_id_stability.py`, `test_rag_search_signature.py`,
`test_rag_keyword_fallback_owner.py`, `test_rag_index_hidden_dirs.py`,
`test_rag_remove_directory_scope.py`, `test_split_chunks_no_duplicate_tail.py`,
`test_personal_docs_pdf_index.py`, `test_personal_docs_office_index.py`,
`test_personal_index_hidden_dirs.py`, `test_chroma_client.py`, `test_embeddings.py`,
`test_embeddings_client.py`, `test_embedding_lanes*.py`.

**Risks / coupling:** Low-to-moderate. The engine itself imports almost nothing app-specific. The
main coupling is to the `src.*` import namespace and to the HTTP-Chroma assumption.

---

## 2. Document ingestion / upload

**Odysseus source files:**
- `src/document_processor.py` (inspected) — PDF extraction via `pypdf`, VL-model OCR for
  image-heavy pages, text/office handling, inline-context budgeting.
- `src/personal_docs.py::extract_pdf_text`, `extract_office_text`.
- `routes/upload_routes.py`, `routes/personal_routes.py`, `src/upload_handler.py`,
  `src/upload_limits.py` (routes listed; handler/limits referenced).
- `src/office_doc.py`, `src/markitdown_runtime.py`, `src/pdf_runtime.py`, `src/pdf_forms.py`.

**What to reuse:**
- `pypdf`-based plain text extraction (`extract_pdf_text`) — simple, BSD, already proven.
- Basic **type validation / size cap** idea (`ODYSSEUS_PERSONAL_UPLOAD_MAX_BYTES`, magic-byte
  content detection in `test_upload_content_detection_magic.py`).

**What to rewrite / adapt:**
- Athena's upload flow is upload → validate → store → extract → chunk → embed → index → respond.
  Write this as a small, explicit pipeline; do **not** port `upload_handler.py` (it is a
  general-purpose multi-file reservation system tied to sessions/owners).
- Keep the **content-addressed de-duplication** (same file → same chunk ids → no double indexing).

**Do NOT migrate:**
- `document_processor.py`'s VL/OCR image path (needs `_resolve_model` + vision settings).
- `pdf_forms.py` / `pdf_form_doc.py` (form editing — unrelated to RAG).
- Office/EPUB via `markitdown` — **defer**; milestone 1 is PDF only.
- The whole `Document` SQLAlchemy model / editor (`office_doc.py`) — unrelated.

**Config required:** `UPLOAD_MAX_BYTES`, `UPLOAD_DIR`.

**Tests that exist:** many `test_upload_*`, `test_personal_*`, `test_document_pdf_marker.py`.

**Risks / coupling:** Extraction logic is clean; the *upload/route* layer is session/owner-coupled
and should not be copied.

---

## 3. ChromaDB

**Odysseus source files:** `src/chroma_client.py` (inspected), consumed by `rag_vector.py`,
`embedding_lanes.py`, `src/memory_vector.py`, `src/mcp_servers/rag_server.py`, `scripts/*`.

**What to reuse:** the fast-fail connect probe pattern (`_port_open` + `heartbeat()` before caching
the singleton) — good for the HTTP mode.

**What to rewrite / adapt:** switch the default to embedded `PersistentClient(path=...)`; keep an
HTTP mode behind env. Collection metadata carries the embedding fingerprint + cosine space.

**Do NOT migrate:** the standalone-service assumption as a hard requirement (Docker `chromadb`
service is an Odysseus deployment choice, not a RAG necessity).

**Config required:** as in §1.

**Tests that exist:** `test_chroma_client.py`, `test_migrate_faiss_to_chroma.py`,
`test_service_health_chromadb.py`.

**Risks / coupling:** Low. Athena pins `chromadb` (embedded client) — note Odysseus pins
`chromadb-client` (HTTP-only). Choose deps to match the selected mode.

---

## 4. FastEmbed / embeddings

**Odysseus source files:** `src/embeddings.py` (inspected), `src/embedding_lanes.py` (inspected).

**What to reuse:**
- `FastEmbedClient` (local ONNX, `sentence-transformers/all-MiniLM-L6-v2`, ~50MB, **no service**) —
  ideal Athena default so the app works offline with zero config.
- `EmbeddingClient` (OpenAI-compatible `/v1/embeddings`) for a configured remote endpoint.
- The Windows hardening: `HF_HUB_DISABLE_SYMLINKS` set before importing `hf_hub`, and the
  broken-symlink cache self-heal — **keep this**, it prevents real Windows failures on a network
  drive (`M:\`). The repo lives on `M:\`, so this matters here.
- Normalization (`normalize_embeddings=True`) and the dimension probe.

**What to rewrite / adapt:** a single `Embeddings` factory that returns exactly one client chosen by
env (`EMBEDDING_PROVIDER=fastembed|http`), dropping the dual-lane/latch complexity. Keep the
batch-size/char-cap guards (`EMBEDDING_BATCH_SIZE`, `EMBEDDING_MAX_CHARS`).

**Do NOT migrate:** `_load_persisted_endpoint` (admin-panel JSON), `reset_http_embed_state` latch,
`secret_storage` decryption — env-only configuration for Athena.

**Config required:** `EMBEDDING_PROVIDER`, `EMBEDDING_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`,
`FASTEMBED_MODEL`, `FASTEMBED_CACHE_PATH`.

**Tests that exist:** `test_embeddings.py`, `test_embeddings_client.py`,
`test_fastembed_cache_path.py`, `test_embedding_lane*.py`.

**Risks / coupling:** Low. First run downloads ~50MB — needs network once.

---

## 5. Retrieval / hybrid search

Covered by §1 (`VectorRAG.search`, `_keyword_search_fallback`, `query_lanes`). Key Athena-specific
addition: a **mandatory relevance threshold** in the RAG Chat path (`RAG_RELEVANCE_THRESHOLD`,
default 0.0–1.0 on the hybrid `similarity` score) so RAG Chat can honestly refuse to answer.

**Risks:** hybrid `similarity` is a *relative* score (0.7*cosine + 0.3*keyword-overlap), not a
probability. The threshold is a **tunable gate**, not a calibrated confidence. The Project Lumen
doc (single page, exact code) makes the positive case easy; the threshold mainly matters for the
negative case. Must be validated empirically (see tests plan).

---

## 6. LLM provider abstraction

**Odysseus source files:** `src/llm_core.py` (~4000 LOC, partially inspected: `_detect_provider`,
Ollama native/compat, Anthropic payload building, Mistral handling, harmony/reasoning models,
response caching, local-model gate, Kimi UA), `src/endpoint_resolver.py`, `src/model_context.py`,
`src/ai_interaction.py`, `src/settings.py` (model config), DB `model_endpoints`.

**What to reuse:** only the *concept* — an OpenAI-compatible `POST /v1/chat/completions` with
`Authorization: Bearer`, plus provider quirks for `max_completion_tokens` vs `max_tokens`.
**Reuse requirement:** API-key providers only (no Ollama/vLLM/local-model gate) per project taste.

**What to rewrite / adapt:** a small `providers/llm.py` exposing
`chat(messages, model=None, temperature=None, max_tokens=None) -> str` and
`chat_stream(...)`, configured from env: `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`. OpenAI-compatible base URL covers OpenAI, **OpenRouter**, Together, Groq, etc.

**Do NOT migrate:** `llm_core.py` wholesale — it drags in FastAPI `HTTPException`, response
caching, host-dead tracking, Ollama/Kimi/ChatGPT-subscription branches, and model capability
tables. None of that is needed for a two-endpoint (chat + optional embeddings) app and would
recreate Odysseus's entanglement.

**Config required:** as above.

**Tests that exist:** dozens of `test_llm_core_*.py` — useful as *behavioral reference* for edge
cases (SSE parsing, temperature restrictions, reasoning fields) but not ported.

**Risks / coupling:** Keeping Athena's LLM client tiny is a deliberate anti-goal-match with
Odysseus. Main risk is re-introducing provider quirks later — acceptable.

---

## 7. Web search

**Odysseus source files:** `services/search/core.py` (inspected), `services/search/providers.py`
(inspected). (`src/search/*` is a re-export shim package pointing at `services/search`.)

**What to reuse:**
- The **provider chain** pattern: primary provider + ordered fallback (`searxng`, `brave`,
  `duckduckgo`, `google_pse`, `tavily`, `serper`, `disabled`).
- `duckduckgo_search` — **no API key required** (library with HTML fallback). Best Athena default.
- `searxng_search_api` — for a self-hosted instance.
- Result normalization shape `{title, url, snippet, age}` and `rank_search_results` (ranking module
  not read directly; referenced).

**What to rewrite / adapt:** a `search/web.py` with a small provider registry + fallback chain,
reading keys from env. Keep the SafeSearch normalization idea but simplify.

**Do NOT migrate:** the file-based `SEARCH_CACHE_DIR` cache/analytics/session machinery, admin
settings store, `comprehensive_web_search`'s giant prompt-string formatter (Athena builds its own
grounded context), `update_search_config`/`get_search_config` admin surface.

**Config required:** `SEARCH_PROVIDER`, `SEARXNG_INSTANCE`, `SEARCH_RESULT_COUNT`,
`BRAVE_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY`, `GOOGLE_PSE_KEY`/`GOOGLE_PSE_CX` (optional).

**Tests that exist:** `test_search_query*.py`, `test_search_ranking*.py`,
`test_search_provider_json.py`, `test_duckduckgo*`, `test_searxng_*`, `test_web_search_*`.

**Risks / coupling:** Moderate — `providers.py` imports `src.constants` and settings helpers;
trivial to re-point. Network egress required at runtime.

---

## 8. URL / page fetching

**Odysseus source files:** `services/search/content.py` (inspected), `src/outbound_fetch.py`
(SSRF guard, referenced), `src/url_safety.py` / `src/url_security.py`.

**What to reuse:**
- `fetch_webpage_content(url, timeout)` — HTML→text heuristics (prefer `main/article/content`
  containers, strip nav/header/footer/script), plus `title`, meta description, lists, tables, code
  blocks, `og_image`.
- The **SSRF guard** (`_public_http_url` / pinned-IP transport) — **important to keep** since
  Athena fetches arbitrary user/LLM-supplied URLs.
- The download size caps (`WEB_FETCH_SOFT_MAX_BYTES=2MB`, `WEB_FETCH_HARD_MAX_BYTES=20MB`) and
  `WEB_FETCH_USER_AGENT`.

**What to rewrite / adapt:** a `search/fetch.py` with the extraction heuristics + a lean SSRF check
(block private/loopback/link-local IPs, resolve-then-pin). Drop the 2-hour disk content cache.

**Do NOT migrate:** the disk content cache (`cache.py`), PDF-via-pdfminer branch (Athena has
`pypdf`), analytics.

**Config required:** `WEB_FETCH_USER_AGENT`, `WEB_FETCH_MAX_BYTES`, `WEB_FETCH_TIMEOUT`.

**Tests that exist:** `test_web_fetch_plaintext.py`, `test_web_fetch_size_caps.py`,
`test_check_outbound_url_nonstring.py`, `test_url_safety.py`.

**Risks / coupling:** Moderate — SSRF guard is security-critical; must be ported carefully, not
simplified away.

---

## 9. News search

**Finding:** Odysseus has **no dedicated news module**. "News" is a *mode within web search* —
`searxng_search_api` switches to `categories=news` + a time range when the query looks fresh
(`_NEWS_HINTS`) or a `time_filter` is supplied (see §7).

**Decision for Athena:** implement `search/news.py` as a thin specialization of `search/web.py`
(news category + recency filter) rather than migrating a separate subsystem. Low risk.

---

## 10. Deep research (audit only — NOT implemented in milestone 1)

**Odysseus source files:**
- `src/deep_research.py` (929 LOC) — **the engine**: `DeepResearcher` IterResearch loop
  (plan → generate queries → search → fetch → extract → synthesize → decide → final report).
  **Self-contained**: needs one chat call, one search call, one fetch call, plus
  `research_utils` + two prompt helpers.
- `src/research_utils.py` (63 LOC) — `strip_thinking`, `is_low_quality` + markers. Trivial.
- `src/research_handler.py` (991 LOC) — **orchestrator**: in-memory task registry keyed by
  session, JSON persistence (`data/deep_research/<session_id>.json`), progress callbacks, cancel,
  continuation, report formatting. **Heavily coupled** to settings/event_bus/endpoint_resolver/auth/
  session manager/DB `ModelEndpoint`.
- `src/visual_report.py` (1933 LOC) — markdown→styled sanitized HTML report (`bs4`+`markdown`+
  `nh3`); client JS hard-references `/api/research/*`.
- `services/research/*` — an **older duplicate** handler+service (only tests import it) →
  **SKIP**.
- `routes/research/research_routes.py` — HTTP/SSE surface (start/status/stream/result/report/
  library/detail/archive/delete/spinoff).
- Legacy `research_engine.ResearchOrchestrator` fallback — **module does not exist** → dead code.

**Reuse later:** `deep_research.py` (swap `_llm`/`_search`/`_fetch_and_extract` to Athena's clients),
`research_utils.py` (copy verbatim), `visual_report.py` (optional renderer).
**Rewrite later:** the orchestrator + routes, cleanly, around Athena's stack.
**SKIP:** `services/research/*`, the `research_engine` fallback, the `routes/research_routes.py` shim.

**Config later:** endpoint/model for research, search provider, timeouts, concurrency, max rounds.
**Tests that exist:** ~20 (`test_deep_research_*.py`, `test_research_*.py`, `test_visual_report*.py`).

**Persistence contract worth preserving later:** `data/deep_research/<id>.json` with fields
`query, status, result, raw_report, sources, raw_findings, stats, category, started_at,
completed_at`.

---

## 11. MCP (audit only — NOT implemented in milestone 1)

**Odysseus source files:**
- `src/mcp_manager.py` (~720 LOC) — `McpManager`: transports (stdio/SSE/streamable-HTTP),
  `connect_server`, `call_tool`, `get_all_openai_schemas`, namespacing `mcp__{server}__{tool}`.
- `src/builtin_mcp.py` — registers built-in stdio servers (`image_gen, memory, rag, email`) + npx
  `builtin_browser`.
- `src/mcp_oauth.py` — generic OAuth bridge (RFC 9728 discovery, DCR, PKCE, refresh) with a
  pluggable token store. **Self-contained / highly reusable.**
- `mcp_servers/{memory,rag,image_gen,email}_server.py` — standalone stdio servers.
  `email_server.py` is ~2900 LOC and deeply coupled to Odysseus's email DB → **SKIP**.
- `routes/mcp/mcp_routes.py` — admin CRUD + OAuth callbacks (`require_admin`).
- SDK pinned **`mcp<2`** (v1 low-level `Server` API).

**Reuse later:** the **client core** (`mcp_manager.py` + `mcp_oauth.py` + transports) with a
pluggable config/token store.
**Rewrite later:** built-in servers as **native in-process tools** (mirroring Odysseus's own
fold-in of bash/python/filesystem/web_search), and the routes/auth layer.
**SKIP:** `email_server.py`, legacy Google paste-back OAuth, `scripts/odysseus-mcp`, the
`routes/mcp_routes.py` `sys.modules` shim, Odysseus-specific command allowlists.

**Tests that exist:** `test_mcp_manager.py`, `test_mcp_oauth.py`, `test_mcp_dependency_compatibility.py`,
`test_builtin_mcp_*.py`, `test_mcp_*` (~20).

**Note for Athena design:** the target architecture should allow a **Tool Registry**
(→ MCP tools / Search tools / Research tools) *without* forcing every tool into every chat.
Milestone 1 keeps Search separate and RAG doc-only.

---

## 12. Authentication / configuration

**Odysseus source files:**
- `src/config.py` — pydantic-settings `BaseSettings` with env prefixes. Clean pattern.
- `src/settings.py` — huge runtime `settings.json` blob + per-user prefs. **SKIP.**
- `src/secret_storage.py` — Fernet `EncryptedText` for secrets at rest. **Best standalone crypto
  file** (only depends on `safe_chmod`).
- `src/api_key_manager.py` — per-provider encrypted key store.
- `core/auth.py` (`AuthManager`) — bcrypt + opaque server-side session tokens + `auth.json`.
- `routes/auth_routes.py` — login/signup/TOTP/2FA/admin CRUD + owner-rename sweeps. **Coupled.**
- `app.py` — ~1300 LOC assembling ~50 routers, manager graph, many startup jobs.

**What to reuse:** the **pydantic-settings pattern** (`config.py`) and, if persistent secrets are
ever needed, `secret_storage.py` verbatim.
**What to rewrite / adapt:** a minimal `config.py` (flat `Settings` from `.env`) and — only if the
user wants it — a trivial single-user cookie/token gate. Odysseus's auth is optional; Athena
milestone 1 can run local-only.
**Do NOT migrate:** `settings.py`, the owner/privilege/TOTP/bearer-token/internal-loopback model,
the `model_endpoints` DB table, the ~50-router app graph, scheduled-task/owner-sweep startup jobs.

**Config required:** LLM keys/URLs, Chroma, embeddings, search keys, `APP_HOST`, `APP_PORT`,
`APP_ENV`, `LOG_LEVEL`, optional `AUTH_ENABLED`.

---

## 13. Dependencies: what Athena actually needs

Odysseus's `requirements.txt` includes many subsystems (email via `caldav`/`icalendar`,
`bcrypt`/`pyotp`/`qrcode`, `croniter`, `psycopg2-binary`, `mcp`, `youtube-transcript-api`,
`markdown`/`nh3`, `SQLAlchemy`, `beautifulsoup4`, `markdown`...). Athena needs a **subset**:

`fastapi`, `uvicorn[standard]`, `python-multipart`, `python-dotenv`, `httpx`, `pydantic`,
`pydantic-settings`, `pypdf`, `numpy`, `beautifulsoup4`, `chromadb`, `fastembed`, `pytest`.

Notes:
- `chromadb` (embedded client) vs `chromadb-client` (HTTP-only) — pick per `CHROMA_MODE`.
- No `SQLAlchemy`/`caldav`/`icalendar`/`mcp`/`bcrypt`/`pyotp`/`psycopg2`/`youtube-transcript-api`
  in milestone 1.
- `httpx2` is Odysseus *test-client* only — not needed.

---

## 14. Boundary / safety

- Nothing under `M:\Athena\odysseus` was written to during this audit.
- If a future step appears to require editing Odysseus, **stop and report** instead.

---

## 15. Recommended Athena decisions (for confirmation)

1. **ChromaDB:** default **embedded `PersistentClient`** (`M:\Athena\athena\data\chroma`), optional
   HTTP mode via env. (Odysseus requires a separate service; Athena should not.)
2. **Embeddings:** default **local FastEmbed** (no service, offline-capable), optional HTTP.
3. **LLM:** clean OpenAI-compatible client; **OpenRouter** is a first-class `LLM_BASE_URL`
   (`https://openrouter.ai/api/v1`).
4. **Scope now:** Chat + RAG Chat + Search (web/news/fetch). Deep Research & MCP: audit only.
5. **RAG gating:** mandatory retrieval + relevance threshold; grounded-only answers; sources shown.
