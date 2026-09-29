"""Calendar API — local SQLite-backed calendars and events.

Route surface and behaviour follow Odysseus's `routes/calendar_routes.py`: the
same paths, the same request/response shapes, the same overlap + recurrence
semantics, the same quick-parse contract, and the same ICS import/export. The
optional CalDAV sync/writeback endpoints are deliberately absent (separate
subsystem, not part of the local calendar).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.calendar.dates import parse_dt, strip_tz
from backend.calendar.store import CalendarStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calendar", tags=["calendar"])

ICS_MAX_BYTES = 10 * 1024 * 1024


class EventCreate(BaseModel):
    summary: str
    dtstart: str
    dtend: Optional[str] = None
    all_day: bool = False
    description: str = ""
    location: str = ""
    calendar_href: Optional[str] = None
    rrule: Optional[str] = None
    color: Optional[str] = None


class EventUpdate(BaseModel):
    summary: Optional[str] = None
    dtstart: Optional[str] = None
    dtend: Optional[str] = None
    all_day: Optional[bool] = None
    description: Optional[str] = None
    location: Optional[str] = None
    rrule: Optional[str] = None
    color: Optional[str] = None


class QuickParseRequest(BaseModel):
    text: str
    tz: Optional[str] = None


class CalendarCreate(BaseModel):
    name: str = "Imported"
    color: str = "#5b8abf"


def _store(request: Request) -> CalendarStore:
    store = getattr(request.app.state, "calendar_store", None)
    if store is None:
        store = CalendarStore(request.app.state.settings)
        request.app.state.calendar_store = store
    return store


# ── calendars ─────────────────────────────────────────────────────────

@router.get("/calendars")
def list_calendars(request: Request) -> dict:
    return {"calendars": _store(request).list_calendars()}


@router.post("/calendars")
def create_calendar(body: CalendarCreate, request: Request) -> dict:
    created = _store(request).create_calendar(body.name, body.color)
    return {"ok": True, **created}


@router.put("/calendars/{cal_id}")
def update_calendar(cal_id: str, request: Request, body: CalendarCreate) -> dict:
    if not _store(request).update_calendar(cal_id, name=body.name, color=body.color):
        raise HTTPException(404, "Calendar not found")
    return {"ok": True}


@router.delete("/calendars/{cal_id}")
def delete_calendar(cal_id: str, request: Request) -> dict:
    if not _store(request).delete_calendar(cal_id):
        raise HTTPException(404, "Calendar not found")
    return {"ok": True}


# ── events ────────────────────────────────────────────────────────────

@router.get("/events")
def list_events(request: Request, start: str, end: str, calendar: str = "") -> dict:
    try:
        start_dt = parse_dt(start)
        end_dt = parse_dt(end)
    except ValueError:
        # A malformed range shouldn't spam the user on every poll.
        logger.warning("list_events: unparseable range start=%r end=%r", start, end)
        return {"events": []}
    if end_dt <= start_dt:
        # Same-day / inverted ranges become a one-day window instead of
        # silently returning nothing.
        end_dt = start_dt + timedelta(days=1)
    return _store(request).list_events(start_dt, end_dt, calendar=calendar)


@router.post("/events")
def create_event(body: EventCreate, request: Request) -> dict:
    try:
        return _store(request).create_event(body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.put("/events/{uid}")
def update_event(uid: str, body: EventUpdate, request: Request) -> dict:
    base_uid = uid.split("::", 1)[0] if "::" in uid else uid
    try:
        ok = _store(request).update_event(base_uid, body.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not ok:
        raise HTTPException(404, "Event not found")
    return {"ok": True}


@router.delete("/events/{uid}")
def delete_event(uid: str, request: Request, scope: str = "series") -> dict:
    result = _store(request).delete_event(uid, scope=scope)
    if not result.get("ok"):
        raise HTTPException(404, "Event not found")
    return result


# ── quick add (LLM parse) ─────────────────────────────────────────────

_QUICK_SYSTEM = """You are a calendar event parser. Read the user's one-line \
description and emit STRICT JSON describing the event. The current local \
timestamp is {now}. Resolve relative dates ("tomorrow", "friday", "next monday", \
"in 30 minutes") against today. Default duration is 60 minutes when no end time \
is given. If the text mentions a date with no time, treat it as an all-day event.

Output ONLY this JSON shape, nothing else:
{{
  "summary": "<event title, capitalized>",
  "dtstart": "<YYYY-MM-DDTHH:MM:00>",
  "dtend":   "<YYYY-MM-DDTHH:MM:00>",
  "all_day": <true|false>,
  "location": "<place or empty>",
  "description": "",
  "confidence": <0.0-1.0>
}}
For all-day events use "YYYY-MM-DD" (no time) for both fields."""


@router.post("/quick-parse")
async def quick_parse(body: QuickParseRequest, request: Request) -> dict:
    """Parse "lunch with sara friday 1pm downtown" into structured event fields."""
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")

    llm = request.app.state.llm
    if not getattr(llm, "configured", False):
        return {"ok": False, "error": "No LLM provider is configured"}

    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    messages = [
        {"role": "system", "content": _QUICK_SYSTEM.format(now=now_iso)},
        {"role": "user", "content": text},
    ]
    selection = request.app.state.provider_registry.mode_model("chat")
    routing = {}
    if selection["provider"]:
        routing["prefer"] = selection["provider"]
    if selection["model"]:
        routing["model"] = selection["model"]

    try:
        result = await asyncio.to_thread(llm.chat, messages, max_tokens=512, **routing)
    except Exception as exc:
        return {"ok": False, "error": f"LLM call failed: {exc}"}

    raw = (getattr(result, "content", "") or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {"ok": False, "error": "Could not extract JSON", "raw": raw[:400]}
    try:
        parsed = json.loads(match.group())
    except Exception as exc:
        return {"ok": False, "error": f"Invalid JSON: {exc}", "raw": raw[:400]}

    summary = (parsed.get("summary") or text)[:200]
    # Strip time tokens the model (or the user) leaks into the summary — the
    # timing already lives in dtstart/dtend.
    summary = re.sub(r"\bin\s+\d+\s*(min|minute|hour|hr|day)s?\b", "", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\(\s*\d{1,2}:\d{2}\s*\)", "", summary)
    summary = re.sub(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", "", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\s+@\s+(?=\d)", " ", summary)
    summary = re.sub(r"\s+", " ", summary).strip(" -—,@")

    all_day = bool(parsed.get("all_day"))
    dtstart = strip_tz((parsed.get("dtstart") or "").strip())
    dtend = strip_tz((parsed.get("dtend") or "").strip())
    if not dtstart:
        return {"ok": False, "error": "Model did not produce a start time", "raw": raw[:400]}
    if not dtend:
        try:
            dtend = dtstart if all_day else (
                datetime.fromisoformat(dtstart) + timedelta(minutes=60)
            ).strftime("%Y-%m-%dT%H:%M:00")
        except Exception:
            dtend = dtstart

    return {
        "ok": True,
        "event": {
            "summary": summary,
            "dtstart": dtstart,
            "dtend": dtend,
            "all_day": all_day,
            "location": (parsed.get("location") or "").strip()[:200],
            "description": (parsed.get("description") or "").strip()[:2000],
        },
        "confidence": float(parsed.get("confidence", 0.7) or 0.7),
    }


# ── ICS import / export ───────────────────────────────────────────────

def _ics_escape(text: str) -> str:
    """Escape a value for an iCalendar TEXT field (RFC 5545 §3.3.11)."""
    return (
        (text or "")
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def _safe_ics_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", name or "").strip("._-") or "calendar"
    return f"{stem[:128]}.ics"


@router.post("/import")
async def import_ics(request: Request, file: UploadFile = File(...),
                     calendar_name: str = "") -> dict:
    """Import events from an .ics file into a local calendar."""
    try:
        from icalendar import Calendar as ICalendar
    except ImportError:  # pragma: no cover - dependency is declared
        raise HTTPException(400, "ICS import requires the 'icalendar' package") from None

    store = _store(request)
    raw = await file.read()
    if len(raw) > ICS_MAX_BYTES:
        raise HTTPException(400, "ICS file too large")
    try:
        parsed_calendar = ICalendar.from_ical(raw)
    except Exception as exc:
        raise HTTPException(400, f"Invalid ICS file: {exc}") from exc

    display = "".join(c for c in (calendar_name.strip() or (file.filename or "").replace(".ics", "").replace("_", " "))
                      if c.isprintable())[:120] or "Imported"
    target = None
    for cal in store.list_calendars():
        if cal["name"] == display:
            target = cal["href"]
            break
    if target is None:
        target = store.create_calendar(display, "#7c4dff")["id"]

    imported = skipped = 0
    from datetime import date as _date, timezone as _tz

    for comp in parsed_calendar.walk():
        if comp.name != "VEVENT":
            continue
        dtstart = comp.get("dtstart")
        if not dtstart:
            skipped += 1
            continue
        value = dtstart.dt
        all_day = isinstance(value, _date) and not isinstance(value, datetime)
        row_is_utc = False
        if all_day:
            start_dt = datetime(value.year, value.month, value.day)
            dtend = comp.get("dtend")
            end_dt = (datetime(dtend.dt.year, dtend.dt.month, dtend.dt.day)
                      if dtend else start_dt + timedelta(days=1))
        else:
            if getattr(value, "tzinfo", None) is not None:
                start_dt = value.astimezone(_tz.utc).replace(tzinfo=None)
                row_is_utc = True
            else:
                start_dt = value
            dtend = comp.get("dtend")
            if dtend:
                end_value = dtend.dt
                if getattr(end_value, "tzinfo", None) is not None:
                    end_dt = end_value.astimezone(_tz.utc).replace(tzinfo=None)
                else:
                    end_dt = end_value
            else:
                end_dt = start_dt + timedelta(hours=1)
        if end_dt <= start_dt:
            end_dt = start_dt + (timedelta(days=1) if all_day else timedelta(hours=1))

        store.create_event({
            "summary": str(comp.get("summary", "")),
            "description": str(comp.get("description", "")),
            "location": str(comp.get("location", "")),
            "dtstart": start_dt.isoformat(timespec="seconds") + ("Z" if row_is_utc else ""),
            "dtend": end_dt.isoformat(timespec="seconds") + ("Z" if row_is_utc else ""),
            "all_day": all_day,
            "rrule": (comp.get("rrule").to_ical().decode() if comp.get("rrule") else ""),
            "calendar_href": target,
        })
        imported += 1

    return {"ok": True, "imported": imported, "skipped": skipped,
            "calendar": display, "calendar_id": target}


@router.get("/export/{cal_id}")
def export_ics(cal_id: str, request: Request) -> Response:
    store = _store(request)
    calendars = store.list_calendars()
    match = next((c for c in calendars if c["href"] == cal_id), None)
    if not match:
        raise HTTPException(404, "Calendar not found")

    events = store.list_events(datetime(1970, 1, 1), datetime(2999, 12, 31), calendar=cal_id)
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Athena//Calendar//EN",
        f"X-WR-CALNAME:{_ics_escape(match['name'])}",
    ]
    for ev in events.get("events", []):
        if ev.get("is_recurrence"):
            continue  # occurrences are generated from the series RRULE
        lines.append("BEGIN:VEVENT")
        lines.append(f"UID:{ev['uid']}")
        lines.append(f"SUMMARY:{_ics_escape(ev.get('summary', ''))}")
        if ev.get("all_day"):
            lines.append(f"DTSTART;VALUE=DATE:{ev['dtstart'][:10].replace('-', '')}")
            lines.append(f"DTEND;VALUE=DATE:{ev['dtend'][:10].replace('-', '')}")
        else:
            suffix = "Z" if ev.get("is_utc") else ""
            start = ev["dtstart"].rstrip("Zz").replace("-", "").replace(":", "")[:15]
            end = ev["dtend"].rstrip("Zz").replace("-", "").replace(":", "")[:15]
            lines.append(f"DTSTART:{start}{suffix}")
            lines.append(f"DTEND:{end}{suffix}")
        if ev.get("description"):
            lines.append(f"DESCRIPTION:{_ics_escape(ev['description'])}")
        if ev.get("location"):
            lines.append(f"LOCATION:{_ics_escape(ev['location'])}")
        if ev.get("rrule"):
            lines.append(f"RRULE:{ev['rrule']}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")

    return Response(
        content="\r\n".join(lines),
        media_type="text/calendar",
        headers={
            "Content-Disposition": f'attachment; filename="{_safe_ics_filename(match["name"])}"',
            "X-Content-Type-Options": "nosniff",
        },
    )
