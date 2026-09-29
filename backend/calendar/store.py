"""SQLite-backed calendar storage.

Mirrors the shape and semantics of Odysseus's `calendars` / `calendar_events`
tables (`core/database.py`) and its recurrence expansion
(`routes/calendar_routes.py::_expand_rrule`), but uses the standard library's
`sqlite3` instead of SQLAlchemy so Athena stays self-contained: no ORM, no
database server, one file under `data/`.

Local SQLite is the source of truth — the same guarantee Odysseus documents.
Odysseus's CalDAV tables, writeback markers and delete tombstones are not
carried over because CalDAV sync is a separate optional subsystem.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from backend.calendar.dates import parse_dt_pair
from backend.config import Settings

logger = logging.getLogger(__name__)

# Athena is single-user/local; kept as a column for parity with Odysseus's model.
DEFAULT_OWNER = "local"
DEFAULT_CALENDAR_COLOR = "#5b8abf"
RRULE_EXPANSION_LIMIT = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calendars (
    id          TEXT PRIMARY KEY,
    owner       TEXT,
    name        TEXT NOT NULL,
    color       TEXT DEFAULT '#5b8abf',
    source      TEXT DEFAULT 'local',
    created_at  TEXT,
    updated_at  TEXT
);
CREATE TABLE IF NOT EXISTS calendar_events (
    uid                TEXT PRIMARY KEY,
    calendar_id        TEXT NOT NULL,
    summary            TEXT NOT NULL DEFAULT '',
    description        TEXT DEFAULT '',
    location           TEXT DEFAULT '',
    dtstart            TEXT NOT NULL,
    dtend              TEXT NOT NULL,
    all_day            INTEGER DEFAULT 0,
    is_utc             INTEGER DEFAULT 0,
    rrule              TEXT DEFAULT '',
    recurrence_exdates TEXT DEFAULT '',
    color              TEXT,
    status             TEXT DEFAULT 'confirmed',
    importance         TEXT DEFAULT 'normal',
    event_type         TEXT,
    created_at         TEXT,
    updated_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_dtstart ON calendar_events(dtstart);
CREATE INDEX IF NOT EXISTS idx_events_calendar ON calendar_events(calendar_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


class CalendarStore:
    def __init__(self, settings: Settings, *, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else settings.resolved_data_dir() / "calendar.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # calendars
    # ------------------------------------------------------------------
    def ensure_default_calendar(self, owner: str = DEFAULT_OWNER) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM calendars WHERE owner = ? ORDER BY created_at LIMIT 1", (owner,)
            ).fetchone()
            if row:
                return dict(row)
            cal_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO calendars (id, owner, name, color, source, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, 'local', ?, ?)",
                (cal_id, owner, "Personal", DEFAULT_CALENDAR_COLOR, _now(), _now()),
            )
            row = conn.execute("SELECT * FROM calendars WHERE id = ?", (cal_id,)).fetchone()
            return dict(row)

    def list_calendars(self, owner: str = DEFAULT_OWNER) -> List[Dict[str, Any]]:
        self.ensure_default_calendar(owner)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM calendars WHERE owner = ? ORDER BY created_at", (owner,)
            ).fetchall()
        return [
            {"name": r["name"], "href": r["id"], "color": r["color"], "source": r["source"]}
            for r in rows
        ]

    def create_calendar(self, name: str, color: str = DEFAULT_CALENDAR_COLOR,
                        owner: str = DEFAULT_OWNER) -> Dict[str, Any]:
        cal_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO calendars (id, owner, name, color, source, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, 'local', ?, ?)",
                (cal_id, owner, name, color, _now(), _now()),
            )
        return {"id": cal_id, "name": name, "color": color}

    def update_calendar(self, cal_id: str, *, name: Optional[str] = None,
                        color: Optional[str] = None) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM calendars WHERE id = ?", (cal_id,)).fetchone()
            if not row:
                return False
            if name is not None:
                conn.execute("UPDATE calendars SET name = ?, updated_at = ? WHERE id = ?",
                             (name, _now(), cal_id))
            if color is not None:
                conn.execute("UPDATE calendars SET color = ?, updated_at = ? WHERE id = ?",
                             (color, _now(), cal_id))
        return True

    def delete_calendar(self, cal_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM calendars WHERE id = ?", (cal_id,)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM calendar_events WHERE calendar_id = ?", (cal_id,))
            conn.execute("DELETE FROM calendars WHERE id = ?", (cal_id,))
        return True

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def _calendar_for_event(self, conn, uid: str):
        return conn.execute(
            "SELECT e.*, c.name AS calendar_name, c.color AS calendar_color,"
            " c.id AS cal_id FROM calendar_events e"
            " LEFT JOIN calendars c ON c.id = e.calendar_id WHERE e.uid = ?",
            (uid,),
        ).fetchone()

    @staticmethod
    def _exdates(row) -> List[str]:
        raw = row["recurrence_exdates"] or ""
        if not raw:
            return []
        try:
            values = json.loads(raw)
        except Exception:
            return []
        if not isinstance(values, list):
            return []
        return [str(v) for v in values if isinstance(v, str) and v.strip()]

    def _event_to_dict(self, row) -> Dict[str, Any]:
        start = datetime.fromisoformat(row["dtstart"])
        end = datetime.fromisoformat(row["dtend"])
        all_day = bool(row["all_day"])
        is_utc = bool(row["is_utc"])
        if all_day:
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")
        else:
            suffix = "Z" if is_utc else ""
            start_str = _iso(start) + suffix
            end_str = _iso(end) + suffix
        return {
            "uid": row["uid"],
            "summary": row["summary"] or "",
            "dtstart": start_str,
            "dtend": end_str,
            "all_day": all_day,
            "is_utc": is_utc,
            "description": row["description"] or "",
            "location": row["location"] or "",
            "rrule": row["rrule"] or "",
            "recurrence_exdates": self._exdates(row),
            "calendar": row["calendar_name"] or "",
            "calendar_href": row["calendar_id"],
            "color": row["color"] or row["calendar_color"] or "",
            "event_type": row["event_type"],
            "importance": row["importance"] or "normal",
        }

    # -- recurrence ----------------------------------------------------
    def _expand_rrule(self, row, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Expand one recurring event into occurrence dicts (Odysseus semantics)."""
        base = self._event_to_dict(row)
        dtstart = datetime.fromisoformat(row["dtstart"])
        dtend = datetime.fromisoformat(row["dtend"])
        duration = dtend - dtstart

        if not (row["rrule"] or "").strip():
            base["is_recurrence"] = False
            base["series_uid"] = row["uid"]
            base["truncated"] = False
            return [base]

        rrule_str = row["rrule"]
        # dateutil rejects a tz-aware UNTIL against a naive DTSTART; the stored
        # DTSTART is naive UTC, so drop the trailing Z on UNTIL to match.
        import re as _re
        rrule_str = _re.sub(r"(UNTIL=\d{8}(?:T\d{6})?)Z", r"\1", rrule_str, flags=_re.IGNORECASE)

        try:
            from dateutil.rrule import rrulestr

            rule = rrulestr(rrule_str, dtstart=dtstart)
        except Exception as exc:
            logger.warning("Failed to parse rrule=%r for event %s: %s", row["rrule"], row["uid"], exc)
            base["is_recurrence"] = False
            base["series_uid"] = row["uid"]
            base["truncated"] = False
            return [base] if (dtstart < end and dtend > start) else []

        exdates = set(self._exdates(row))
        all_day = bool(row["all_day"])
        is_utc = bool(row["is_utc"])
        results: List[Dict[str, Any]] = []
        truncated = False

        for occ_start in rule.xafter(start - duration, inc=True):
            if occ_start >= end:
                break
            occ_end = occ_start + duration
            if occ_end <= start:
                continue
            if len(results) >= RRULE_EXPANSION_LIMIT:
                truncated = True
                break

            if all_day:
                key = occ_start.strftime("%Y-%m-%d")
                occ_uid = f"{row['uid']}::{key}"
            else:
                key = occ_start.strftime("%Y-%m-%dT%H:%M")
                occ_uid = f"{row['uid']}::{key}"
            if key in exdates:
                continue

            d = dict(base)
            d["uid"] = occ_uid
            d["series_uid"] = row["uid"]
            d["is_recurrence"] = True
            d["truncated"] = False
            if all_day:
                d["dtstart"] = occ_start.strftime("%Y-%m-%d")
                d["dtend"] = occ_end.strftime("%Y-%m-%d")
            else:
                suffix = "Z" if is_utc else ""
                d["dtstart"] = _iso(occ_start) + suffix
                d["dtend"] = _iso(occ_end) + suffix
                d["is_utc"] = is_utc
            results.append(d)

        if truncated:
            for d in results:
                d["truncated"] = True
        return results

    # -- CRUD ----------------------------------------------------------
    def list_events(self, start: datetime, end: datetime, *,
                    calendar: str = "", owner: str = DEFAULT_OWNER) -> Dict[str, Any]:
        with self._connect() as conn:
            sql = (
                "SELECT e.*, c.name AS calendar_name, c.color AS calendar_color,"
                " c.owner AS cal_owner FROM calendar_events e"
                " LEFT JOIN calendars c ON c.id = e.calendar_id"
                " WHERE e.status != 'cancelled' AND (c.owner = ? OR c.owner IS NULL)"
                " AND ("
                "   ((e.rrule IS NULL OR e.rrule = '') AND e.dtstart < ? AND e.dtend > ?)"
                "   OR (e.rrule IS NOT NULL AND e.rrule != '' AND e.dtstart < ?)"
                " ) ORDER BY e.dtstart"
            )
            start_iso = _iso(start)
            end_iso = _iso(end)
            rows = conn.execute(sql, (owner, end_iso, start_iso, end_iso)).fetchall()
            if calendar:
                rows = [r for r in rows if r["calendar_id"] == calendar or r["calendar_name"] == calendar]

            expanded: List[Dict[str, Any]] = []
            for row in rows:
                expanded.extend(self._expand_rrule(row, start, end))

        expanded.sort(key=lambda d: d["dtstart"])
        response: Dict[str, Any] = {"events": expanded}
        if any(e.get("truncated") for e in expanded):
            response["truncated"] = True
        return response

    def get_event(self, uid: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = self._calendar_for_event(conn, uid)
            return self._event_to_dict(row) if row else None

    def create_event(self, data: Dict[str, Any], owner: str = DEFAULT_OWNER) -> Dict[str, Any]:
        calendar_id = (data.get("calendar_href") or "").strip()
        with self._connect() as conn:
            if calendar_id:
                row = conn.execute("SELECT id FROM calendars WHERE id = ?", (calendar_id,)).fetchone()
                if not row:
                    calendar_id = ""
            if not calendar_id:
                calendar_id = self.ensure_default_calendar(owner)["id"]

            uid = str(uuid.uuid4())
            dtstart, start_utc = parse_dt_pair(data["dtstart"])
            all_day = bool(data.get("all_day"))
            if data.get("dtend"):
                dtend, end_utc = parse_dt_pair(data["dtend"])
                start_utc = start_utc or end_utc
            elif all_day:
                dtend = dtstart + timedelta(days=1)
            else:
                dtend = dtstart + timedelta(hours=1)

            conn.execute(
                "INSERT INTO calendar_events (uid, calendar_id, summary, description, location,"
                " dtstart, dtend, all_day, is_utc, rrule, recurrence_exdates, color, status,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, 'confirmed', ?, ?)",
                (
                    uid, calendar_id, data.get("summary") or "", data.get("description") or "",
                    data.get("location") or "", _iso(dtstart), _iso(dtend),
                    1 if all_day else 0, 1 if (start_utc and not all_day) else 0,
                    data.get("rrule") or "", data.get("color") or None, _now(), _now(),
                ),
            )
        return {"ok": True, "uid": uid}

    def update_event(self, uid: str, data: Dict[str, Any]) -> bool:
        with self._connect() as conn:
            row = self._calendar_for_event(conn, uid)
            if not row:
                return False
            fields: Dict[str, Any] = {}
            if data.get("summary") is not None:
                fields["summary"] = data["summary"]
            if data.get("description") is not None:
                fields["description"] = data["description"]
            if data.get("location") is not None:
                fields["location"] = data["location"]
            if data.get("rrule") is not None:
                fields["rrule"] = data["rrule"]
            if data.get("color") is not None:
                fields["color"] = data["color"] or None
            if data.get("dtstart") is not None:
                value, was_utc = parse_dt_pair(data["dtstart"])
                fields["dtstart"] = _iso(value)
                if was_utc:
                    fields["is_utc"] = 1
            if data.get("dtend") is not None:
                value, was_utc = parse_dt_pair(data["dtend"])
                fields["dtend"] = _iso(value)
                if was_utc:
                    fields["is_utc"] = 1
            if data.get("all_day") is not None:
                fields["all_day"] = 1 if data["all_day"] else 0
                if data["all_day"]:
                    fields["is_utc"] = 0
            if not fields:
                return True
            fields["updated_at"] = _now()
            assignments = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(f"UPDATE calendar_events SET {assignments} WHERE uid = ?",
                         (*fields.values(), uid))
        return True

    def delete_event(self, uid: str, *, scope: str = "series") -> Dict[str, Any]:
        base_uid = uid.split("::", 1)[0] if "::" in uid else uid
        with self._connect() as conn:
            row = self._calendar_for_event(conn, base_uid)
            if not row:
                return {"ok": False}
            is_occurrence = scope in {"occurrence", "instance"} and "::" in uid and bool(row["rrule"])
            if is_occurrence:
                suffix = uid.split("::", 1)[1]
                key = suffix[:10] if row["all_day"] else suffix[:16]
                exdates = self._exdates(row)
                if key not in exdates:
                    exdates.append(key)
                conn.execute(
                    "UPDATE calendar_events SET recurrence_exdates = ?, updated_at = ? WHERE uid = ?",
                    (json.dumps(sorted(exdates)), _now(), base_uid),
                )
                return {"ok": True, "scope": "occurrence", "exdate": key}
            conn.execute("DELETE FROM calendar_events WHERE uid = ?", (base_uid,))
        return {"ok": True, "scope": "series"}
