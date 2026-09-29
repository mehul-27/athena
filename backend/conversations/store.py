"""Conversation persistence.

Athena's Chat was previously client-only (the message list lived in the browser),
which is not enough for a chat header with usage/cost, persistent rename,
compaction, or associating a Deep Research job with the conversation it came
from. This module is the minimum durable model for that: conversations,
messages, per-request usage, and documents saved out of a chat.

Shaped after Odysseus's `sessions` / `chat_messages` tables (session name +
running `total_input_tokens`/`total_output_tokens`, metrics stored per message),
but with its own two things this task requires: a per-request `usage` row that
keeps provider/model/cost, and a `conversation_id` link other features can join
on. Standard-library `sqlite3`; no ORM.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from backend.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "New Chat"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL DEFAULT 'New Chat',
    created_at          TEXT,
    updated_at          TEXT,
    request_count       INTEGER DEFAULT 0,
    total_input_tokens  INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_cost_usd      REAL DEFAULT 0,
    unpriced_requests   INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    provider        TEXT,
    model           TEXT,
    capabilities    TEXT DEFAULT '',
    sources         TEXT DEFAULT '',
    meta            TEXT DEFAULT '',
    compacted       INTEGER DEFAULT 0,
    created_at      TEXT
);
CREATE TABLE IF NOT EXISTS usage (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT,
    message_id      TEXT,
    provider        TEXT,
    model           TEXT,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    cost_usd        REAL,
    cost_known      INTEGER DEFAULT 0,
    latency_ms      INTEGER DEFAULT 0,
    source          TEXT DEFAULT 'chat',
    created_at      TEXT
);
CREATE TABLE IF NOT EXISTS saved_documents (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT,
    title           TEXT NOT NULL DEFAULT 'Untitled',
    content         TEXT NOT NULL DEFAULT '',
    created_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_conversation ON usage(conversation_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, settings: Settings, *, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else settings.resolved_data_dir() / "conversations.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # conversations
    # ------------------------------------------------------------------
    def create(self, title: str = DEFAULT_TITLE) -> Dict[str, Any]:
        conv_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title.strip() or DEFAULT_TITLE, _now(), _now()),
            )
        return self.get(conv_id)

    def ensure(self, conversation_id: Optional[str]) -> Dict[str, Any]:
        """Return the conversation, creating it when the id is missing/unknown."""
        if conversation_id:
            existing = self.get(conversation_id)
            if existing:
                return existing
        if conversation_id:
            # Caller supplied an id (e.g. a resumed client) — keep it.
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    (conversation_id, DEFAULT_TITLE, _now(), _now()),
                )
            return self.get(conversation_id) or {}
        return self.create()

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not row:
                return None
            conv = dict(row)
            conv["message_count"] = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND compacted = 0",
                (conversation_id,),
            ).fetchone()[0]
            conv["cost_known"] = conv["unpriced_requests"] == 0
            conv["total_tokens"] = (conv["total_input_tokens"] or 0) + (conv["total_output_tokens"] or 0)
            return conv

    def list(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (max(1, limit),)
            ).fetchall()
        out = []
        for row in rows:
            conv = dict(row)
            conv["cost_known"] = conv["unpriced_requests"] == 0
            conv["total_tokens"] = (conv["total_input_tokens"] or 0) + (conv["total_output_tokens"] or 0)
            out.append(conv)
        return out

    def rename(self, conversation_id: str, title: str) -> bool:
        clean = (title or "").strip()[:200]
        if not clean:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (clean, _now(), conversation_id),
            )
            return cur.rowcount > 0

    def delete(self, conversation_id: str) -> Dict[str, Any]:
        """Delete the conversation and its messages/usage.

        Saved documents are detached (kept), and the caller is responsible for
        clearing research links — nothing here touches the RAG knowledge base.
        """
        with self._connect() as conn:
            existed = conn.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone() is not None
            if not existed:
                return {"deleted": False}
            messages = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()[0]
            conn.execute(
                "UPDATE saved_documents SET conversation_id = NULL WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM usage WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return {"deleted": True, "removed_messages": messages}

    def touch(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                         (_now(), conversation_id))

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        provider: str = "",
        model: str = "",
        capabilities: Optional[Iterable[str]] = None,
        sources: Optional[Iterable[Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        mid = message_id or str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, provider, model,"
                " capabilities, sources, meta, compacted, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    mid, conversation_id, role, content or "", provider or "", model or "",
                    ",".join(sorted(capabilities or [])),
                    json.dumps(list(sources or []), ensure_ascii=False),
                    json.dumps(meta or {}, ensure_ascii=False),
                    _now(),
                ),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                         (_now(), conversation_id))
        return {
            "id": mid,
            "conversation_id": conversation_id,
            "role": role,
            "content": content or "",
            "provider": provider or "",
            "model": model or "",
            "capabilities": sorted(capabilities or []),
            "sources": list(sources or []),
            "meta": meta or {},
            "created_at": _now(),
        }

    def messages(self, conversation_id: str, *, include_compacted: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE conversation_id = ?"
        if not include_compacted:
            sql += " AND compacted = 0"
        sql += " ORDER BY (role = 'system') DESC, rowid"
        with self._connect() as conn:
            rows = conn.execute(sql, (conversation_id,)).fetchall()
        out = []
        for row in rows:
            msg = dict(row)
            msg["capabilities"] = [c for c in (msg["capabilities"] or "").split(",") if c]
            try:
                msg["sources"] = json.loads(msg["sources"] or "[]")
            except ValueError:
                msg["sources"] = []
            try:
                msg["meta"] = json.loads(msg["meta"] or "{}")
            except ValueError:
                msg["meta"] = {}
            out.append(msg)
        return out

    def history(self, conversation_id: str, limit: int = 24) -> List[Dict[str, str]]:
        """Recent turns in the {role, content} shape the chat endpoints accept.

        Includes the compaction summary (`role="system"`) so a compacted thread
        still carries its earlier context into later turns.
        """
        turns = [
            m for m in self.messages(conversation_id)
            if m["role"] in ("user", "assistant") or m["role"] == "system"
        ]
        return [{"role": m["role"], "content": m["content"]} for m in turns[-limit:]]

    def mark_compacted(self, message_ids: Iterable[str]) -> int:
        """Hide messages from the active thread without destroying them."""
        ids = [mid for mid in message_ids if mid]
        if not ids:
            return 0
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in ids)
            cur = conn.execute(
                f"UPDATE messages SET compacted = 1 WHERE id IN ({placeholders})", tuple(ids)
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # usage
    # ------------------------------------------------------------------
    def record_usage(
        self,
        conversation_id: Optional[str],
        *,
        message_id: Optional[str] = None,
        provider: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: Optional[float] = None,
        cost_known: bool = False,
        latency_ms: int = 0,
        source: str = "chat",
    ) -> Dict[str, Any]:
        """Record one LLM call and roll it into the conversation totals."""
        usage_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO usage (id, conversation_id, message_id, provider, model,"
                " input_tokens, output_tokens, cost_usd, cost_known, latency_ms, source, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    usage_id, conversation_id, message_id, provider or "", model or "",
                    int(input_tokens or 0), int(output_tokens or 0),
                    cost_usd, 1 if cost_known else 0, int(latency_ms or 0), source, _now(),
                ),
            )
            if conversation_id:
                conn.execute(
                    "UPDATE conversations SET request_count = request_count + 1,"
                    " total_input_tokens = total_input_tokens + ?,"
                    " total_output_tokens = total_output_tokens + ?,"
                    " total_cost_usd = total_cost_usd + ?,"
                    " unpriced_requests = unpriced_requests + ?,"
                    " updated_at = ? WHERE id = ?",
                    (
                        int(input_tokens or 0), int(output_tokens or 0),
                        float(cost_usd or 0.0), 0 if cost_known else 1,
                        _now(), conversation_id,
                    ),
                )
        return {
            "id": usage_id, "provider": provider, "model": model,
            "input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0),
            "cost_usd": cost_usd, "cost_known": cost_known,
            "latency_ms": int(latency_ms or 0), "source": source,
        }

    def usage(self, conversation_id: str) -> Dict[str, Any]:
        """Per-provider/model breakdown plus the conversation totals."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT provider, model, COUNT(*) AS requests,"
                " SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,"
                " SUM(CASE WHEN cost_known = 1 THEN cost_usd ELSE 0 END) AS known_cost,"
                " SUM(CASE WHEN cost_known = 0 THEN 1 ELSE 0 END) AS unpriced"
                " FROM usage WHERE conversation_id = ?"
                " GROUP BY provider, model ORDER BY requests DESC",
                (conversation_id,),
            ).fetchall()
        breakdown = []
        for row in rows:
            item = dict(row)
            item["input_tokens"] = item["input_tokens"] or 0
            item["output_tokens"] = item["output_tokens"] or 0
            item["total_tokens"] = item["input_tokens"] + item["output_tokens"]
            item["cost_known"] = (item["unpriced"] or 0) == 0
            item["cost_usd"] = round(item["known_cost"] or 0.0, 6) if item["cost_known"] else None
            breakdown.append(item)
        totals = {
            "requests": sum(i["requests"] for i in breakdown),
            "input_tokens": sum(i["input_tokens"] for i in breakdown),
            "output_tokens": sum(i["output_tokens"] for i in breakdown),
            "total_tokens": sum(i["total_tokens"] for i in breakdown),
            "unpriced_requests": sum(i["unpriced"] or 0 for i in breakdown),
        }
        totals["cost_known"] = totals["unpriced_requests"] == 0 and totals["requests"] > 0
        totals["cost_usd"] = (
            round(sum(i["known_cost"] or 0.0 for i in rows), 6) if totals["cost_known"] else None
        )
        return {"totals": totals, "by_model": breakdown}

    # ------------------------------------------------------------------
    # saved documents ("Save to Documents")
    # ------------------------------------------------------------------
    def save_document(self, conversation_id: Optional[str], title: str, content: str) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO saved_documents (id, conversation_id, title, content, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (doc_id, conversation_id, (title or "Untitled")[:200], content or "", _now()),
            )
        return {"id": doc_id, "conversation_id": conversation_id,
                "title": (title or "Untitled")[:200], "created_at": _now()}

    def saved_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, conversation_id, title, created_at, LENGTH(content) AS chars"
                " FROM saved_documents ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def saved_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM saved_documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def delete_saved_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM saved_documents WHERE id = ?", (doc_id,))
            return cur.rowcount > 0
