"""Background research task registry, lifecycle, and partial-result handling."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from backend.config import Settings
from backend.providers.router import LLMRouter
from backend.research.engine import DeepResearcher
from backend.research.llm import ResearchLLM
from backend.research.sources import extract_raw_findings, extract_sources
from backend.research.store import ResearchStore
from backend.research.text import strip_thinking

logger = logging.getLogger(__name__)


def format_research_report(question: str, report: str, stats: Dict[str, Any], elapsed: float) -> str:
    report = strip_thinking(report)
    summary_lines = [
        f"**Duration:** {elapsed:.1f}s",
        f"**Rounds:** {stats.get('Rounds', stats.get('Findings', '?'))}",
        f"**Queries:** {stats.get('Queries', stats.get('Searches', '?'))}",
        f"**URLs Analyzed:** {stats.get('URLs', '?')}",
    ]
    return "---\n\n## Research Summary\n\n" + " | ".join(summary_lines) + "\n\n---\n\n" + report + "\n"


class ResearchRunner:
    def __init__(
        self,
        settings: Settings,
        router: LLMRouter,
        store: Optional[ResearchStore] = None,
        registry: Optional[Any] = None,
        conversations: Optional[Any] = None,
    ) -> None:
        self.settings = settings
        self.router = router
        self.registry = registry
        self.conversations = conversations
        self.store = store or ResearchStore(settings)
        self.active: Dict[str, Dict[str, Any]] = {}

    def _conversations(self):
        """Lazily build the conversation store (kept optional so tests that
        construct a runner without one still work)."""
        if self.conversations is None:
            try:
                from backend.conversations.store import ConversationStore

                self.conversations = ConversationStore(self.settings)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Conversation store unavailable: %s", exc)
                return None
        return self.conversations

    def _role_models(self) -> Dict[str, str]:
        if self.registry is not None:
            return self.registry.research_models()
        return {
            "fast": self.settings.research_fast_model,
            "strong": self.settings.research_strong_model,
        }

    def _mode_model(self) -> Dict[str, str]:
        """The model selected for the Research tab (blank = fallback chain)."""
        if self.registry is not None:
            return self.registry.mode_model("research")
        return {"provider": "", "model": ""}

    def start(
        self,
        query: str,
        *,
        max_rounds: Optional[int] = None,
        max_time: Optional[int] = None,
        search_provider: Optional[str] = None,
        category: Optional[str] = None,
        prefer: Optional[str] = None,
        model: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        research_id = f"rp-{uuid.uuid4().hex[:12]}"
        rounds = self.settings.research_max_rounds if not max_rounds else max_rounds
        soft_time = self.settings.research_max_time if max_time is None else max_time
        hard_timeout = self.settings.research_run_timeout_seconds
        hard_timeout = None if hard_timeout <= 0 else max(60, min(86400, hard_timeout))

        for old_id, old in list(self.active.items()):
            if old.get("query") == query and old.get("status") == "running":
                self.cancel(old_id)

        entry: Dict[str, Any] = {
            "task": None,
            "researcher": None,
            "query": query,
            "status": "running",
            "progress": {},
            "result": None,
            "raw_report": "",
            "sources": [],
            "raw_findings": [],
            "stats": {},
            "category": category,
            "conversation_id": conversation_id,
            "started_at": time.time(),
            "completed_at": None,
            "hidden_images": [],
            "archived": False,
            "consumed": False,
        }
        self.active[research_id] = entry

        def on_progress(event: Dict[str, Any]) -> None:
            entry["progress"] = dict(event)

        async def run() -> None:
            roles = self._role_models()
            # Explicit per-request selection (the Research tab's model picker)
            # wins over the stored per-mode selection.
            selection = self._mode_model() if prefer is None and model is None else {"provider": prefer or "", "model": model or ""}
            llm = ResearchLLM(
                self.router,
                fast_model=roles.get("fast"),
                strong_model=roles.get("strong"),
                prefer_provider=selection.get("provider") or None,
                mode_model=selection.get("model") or None,
                # Deep Research makes many calls; every one of them counts
                # against the conversation the research was launched from.
                usage_sink=self._usage_sink(conversation_id),
            )
            researcher = DeepResearcher(
                llm,
                max_rounds=rounds,
                max_time=soft_time,
                max_urls_per_round=self.settings.research_max_urls_per_round,
                max_content_chars=self.settings.research_max_content_chars,
                max_report_tokens=self.settings.research_max_report_tokens,
                extraction_timeout=self.settings.research_extraction_timeout_seconds,
                planning_timeout=self.settings.research_planning_timeout_seconds,
                query_timeout=self.settings.research_query_timeout_seconds,
                extraction_concurrency=self.settings.research_extraction_concurrency,
                min_rounds=min(self.settings.research_min_rounds, max(2, rounds - 2)),
                max_empty_rounds=self.settings.research_max_empty_rounds,
                max_stagnant_rounds=self.settings.research_max_stagnant_rounds,
                batch_extraction=self.settings.research_batch_extraction,
                extraction_batch_size=self.settings.research_extraction_batch_size,
                synthesis_window=self.settings.research_synthesis_window,
                progress_callback=on_progress,
                search_provider=search_provider,
                category=category,
            )
            researcher._settings = self.settings
            entry["researcher"] = researcher
            # Snapshot the router's monotonic counters so fallbacks can be
            # attributed to *this* run.
            router_before = dict(getattr(self.router, "stats", {}) or {})
            try:
                operation = researcher.research(query)
                raw = await asyncio.wait_for(operation, timeout=hard_timeout)
                await self._finish(research_id, entry, researcher, raw, "done", router_before)
            except asyncio.TimeoutError:
                logger.error("Research hard timeout id=%s timeout=%s", research_id, hard_timeout)
                if researcher.evolving_report:
                    await self._finish(research_id, entry, researcher, researcher.evolving_report, "done", router_before)
                    on_progress({"phase": "warning", "message": f"Research timed out after {hard_timeout}s; partial results saved"})
                else:
                    entry["status"] = "error"
                    entry["result"] = f"Research timed out after {hard_timeout}s."
                    entry["completed_at"] = time.time()
                    self._persist(research_id, entry)
                    on_progress({"phase": "error", "message": entry["result"]})
            except asyncio.CancelledError:
                entry["status"] = "cancelled"
                entry["completed_at"] = time.time()
                self._persist(research_id, entry)
                on_progress({"phase": "error", "message": "Research cancelled"})
                raise
            except Exception as exc:
                logger.exception("Research task failed id=%s", research_id)
                if researcher.evolving_report:
                    await self._finish(research_id, entry, researcher, researcher.evolving_report, "done", router_before)
                    on_progress({"phase": "warning", "message": f"Research finished with errors; partial results saved ({exc})"})
                else:
                    entry["status"] = "error"
                    entry["result"] = str(exc)
                    entry["completed_at"] = time.time()
                    self._persist(research_id, entry)
                    on_progress({"phase": "error", "message": str(exc)})

        task = asyncio.create_task(run())
        entry["task"] = task
        return {
            "research_id": research_id,
            "session_id": research_id,
            "status": "running",
            "query": query,
            "conversation_id": conversation_id,
        }

    async def _finish(
        self,
        research_id: str,
        entry: Dict[str, Any],
        researcher: DeepResearcher,
        raw: str,
        status: str,
        router_before: Optional[Dict[str, int]] = None,
    ) -> None:
        raw = strip_thinking(raw)
        stats = researcher.get_stats()
        elapsed = time.time() - entry["started_at"]
        diagnostics = researcher.get_diagnostics()
        if router_before is not None:
            after = dict(getattr(self.router, "stats", {}) or {})
            diagnostics["fallbacks"] = max(0, after.get("fallbacks", 0) - router_before.get("fallbacks", 0))
            diagnostics["llm_requests"] = max(0, after.get("requests", 0) - router_before.get("requests", 0))
        logger.info(
            "Research diagnostics id=%s rounds=%s searches=%s pages=%s llm_calls=%s "
            "providers=%s fallbacks=%s duration=%.1fs",
            research_id,
            diagnostics.get("rounds"),
            diagnostics.get("searches"),
            diagnostics.get("pages_fetched"),
            diagnostics.get("llm_calls"),
            ",".join(diagnostics.get("providers") or []) or "none",
            diagnostics.get("fallbacks"),
            elapsed,
        )
        entry["raw_report"] = raw
        entry["result"] = format_research_report(entry["query"], raw, stats, elapsed)
        entry["stats"] = stats
        entry["findings"] = researcher.findings
        entry["sources"] = extract_sources(researcher.findings)
        entry["raw_findings"] = extract_raw_findings(researcher.findings)
        entry["analyzed_urls"] = list(researcher.analyzed_urls)
        entry["providers_used"] = list(researcher.providers_used)
        entry["round_count"] = researcher.round_count
        entry["category"] = researcher.category
        entry["diagnostics"] = diagnostics
        entry["status"] = status
        entry["completed_at"] = time.time()
        self._persist(research_id, entry)
        self._record_in_conversation(research_id, entry)

    def _usage_sink(self, conversation_id: Optional[str]):
        """A callback that records each research LLM call against the conversation.

        Returns `None` when there is no conversation to attribute to, so a
        standalone research job costs nothing extra.
        """
        if not conversation_id:
            return None
        store = self._conversations()
        if store is None:
            return None

        def record(result: Any) -> None:
            store.record_usage(
                conversation_id,
                provider=getattr(result, "provider", "") or "",
                model=getattr(result, "model", "") or "",
                input_tokens=getattr(result, "input_tokens", 0) or 0,
                output_tokens=getattr(result, "output_tokens", 0) or 0,
                cost_usd=getattr(result, "cost_usd", None),
                cost_known=bool(getattr(result, "cost_known", False)),
                latency_ms=getattr(result, "latency_ms", 0) or 0,
                source="research",
            )

        return record

    def _record_in_conversation(self, research_id: str, entry: Dict[str, Any]) -> None:
        """Post the finished research back into the conversation it came from.

        The chat gets a card (not the whole report): status, the real round and
        source counts, and a link to the visual report.
        """
        conversation_id = entry.get("conversation_id")
        if not conversation_id:
            return
        store = self._conversations()
        if store is None:
            return
        try:
            if store.get(conversation_id) is None:
                return
            stats = entry.get("stats") or {}
            sources = entry.get("sources") or []
            rounds = entry.get("round_count") or stats.get("Rounds") or 0
            ok = entry.get("status") == "done"
            summary = (
                f"Deep Research {'completed' if ok else entry.get('status', 'finished')}: "
                f"{entry.get('query', '')}"
            )
            store.add_message(
                conversation_id,
                "assistant",
                summary,
                provider=str(stats.get("Provider") or ""),
                model=str(stats.get("Model") or ""),
                sources=sources,
                meta={
                    "kind": "research",
                    "research_id": research_id,
                    "status": entry.get("status"),
                    "rounds": rounds,
                    "sources": len(sources),
                    "duration": stats.get("Duration"),
                    "report_url": f"/api/research/report/{research_id}",
                },
            )
            logger.info("Research %s linked to conversation %s", research_id, conversation_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not record research %s in its conversation: %s", research_id, exc)

    def _persist(self, research_id: str, entry: Dict[str, Any]) -> None:
        persistent_fields = (
            "query", "status", "result", "raw_report", "sources", "raw_findings",
            "stats", "category", "started_at", "completed_at", "hidden_images",
            "archived", "consumed", "analyzed_urls", "providers_used", "round_count",
            "diagnostics", "conversation_id",
        )
        record = {key: entry.get(key) for key in persistent_fields}
        self.store.save(research_id, record)

    def status(self, research_id: str) -> Optional[Dict[str, Any]]:
        entry = self.active.get(research_id)
        if entry:
            return {
                "status": entry["status"],
                "progress": entry.get("progress", {}),
                "query": entry["query"],
                "started_at": entry["started_at"],
            }
        record = self.store.load(research_id)
        if not record or record.get("consumed"):
            return None
        return {"status": record.get("status", "done"), "progress": {}, "query": record.get("query", ""), "started_at": record.get("started_at", 0)}

    def result(self, research_id: str, *, clear: bool = False) -> Optional[Dict[str, Any]]:
        entry = self.active.get(research_id)
        record = entry if entry and entry.get("status") != "running" else self.store.load(research_id)
        if not record or record.get("consumed"):
            return None
        result = {key: record.get(key) for key in ("result", "sources", "raw_findings", "category", "stats")}
        if clear:
            saved = self.store.load(research_id) or {}
            saved["consumed"] = True
            self.store.save(research_id, saved)
            self.active.pop(research_id, None)
        return result

    def get(self, research_id: str) -> Optional[Dict[str, Any]]:
        return self.store.load(research_id)

    def list(self, *, archived: Optional[bool] = None, limit: int = 50) -> List[Dict[str, Any]]:
        active = [
            {key: entry.get(key) for key in ("query", "status", "result", "raw_report", "sources", "raw_findings", "stats", "category", "started_at", "completed_at", "round_count", "providers_used")}
            | {"research_id": research_id}
            for research_id, entry in self.active.items()
            if entry.get("status") == "running"
        ]
        saved = self.store.list(archived=archived, limit=limit)
        rows = active + saved
        rows.sort(key=lambda row: row.get("completed_at") or row.get("started_at") or 0, reverse=True)
        return rows[:limit]

    def cancel(self, research_id: str) -> bool:
        entry = self.active.get(research_id)
        if not entry or entry.get("status") != "running":
            return False
        researcher = entry.get("researcher")
        if researcher:
            researcher.cancel()
        task = entry.get("task")
        if task and not task.done():
            task.cancel()
        entry["status"] = "cancelled"
        entry["completed_at"] = time.time()
        self._persist(research_id, entry)
        return True

    def update(self, research_id: str, **changes: Any) -> bool:
        record = self.store.load(research_id)
        if not record:
            return False
        record.update(changes)
        self.store.save(research_id, record)
        return True

    def report_html(self, research_id: str) -> Optional[str]:
        record = self.store.load(research_id)
        if not record:
            return None
        from backend.research.report.service import render_record

        return render_record(research_id, record)

    def delete(self, research_id: str) -> bool:
        self.cancel(research_id)
        self.active.pop(research_id, None)
        return self.store.delete(research_id)
