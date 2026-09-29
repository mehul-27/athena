"""Date/time parsing for the calendar.

Ported from `routes/calendar_routes.py` in Odysseus (`_parse_dt`,
`_parse_dt_pair`). The contract is the same: strict ISO first, then a small
natural-language parser covering the phrasings models and users actually emit
("tomorrow 3pm", "next friday", "in 30 minutes"), then dateutil as a last resort.

Events are stored naive; `is_utc` records whether the caller supplied an offset
(in which case the value was converted to UTC), which is what makes the
serializer emit a trailing `Z` so the browser renders it in local time.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Tuple

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _parse_time(text: str):
    """Return (hour, minute) from '1pm', '1:30 PM', '13:00', or None."""
    text = re.sub(r"\b([ap])\s*\.?\s*m\.?\b", r"\1m", text.strip(), flags=re.IGNORECASE)
    m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", text, re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    return hour, minute


def _day_offset(word: str) -> int:
    if word in ("tomorrow", "tmrw"):
        return 1
    if word == "yesterday":
        return -1
    return 0


def parse_dt(s: str) -> datetime:
    """Parse a date/datetime into a naive datetime (UTC if an offset was given)."""
    s = (s or "").strip()
    if not s:
        raise ValueError("empty datetime string")

    # Fast path: strict ISO.
    try:
        if len(s) == 10:
            return datetime.fromisoformat(s)
        candidate = s.replace("Z", "+00:00") if s.endswith("Z") else s
        parsed = datetime.fromisoformat(candidate)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        pass

    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    lower = s.lower().strip()

    # today/tonight/tomorrow/yesterday [at] TIME
    m = re.match(r"^(today|tonight|tomorrow|tmrw|yesterday)(?:\s+at)?\s*(.*)$", lower)
    if m:
        base = today + timedelta(days=_day_offset(m.group(1)))
        rest = m.group(2).strip()
        if not rest:
            return base
        t = _parse_time(rest)
        if t is not None:
            return base.replace(hour=t[0], minute=t[1])

    # time-first: "3pm today", "9am tomorrow"
    m = re.match(r"^(.+?)\s+(today|tonight|tomorrow|tmrw|yesterday)$", lower)
    if m:
        base = today + timedelta(days=_day_offset(m.group(2)))
        t = _parse_time(m.group(1).strip())
        if t is not None:
            return base.replace(hour=t[0], minute=t[1])

    # next <weekday> [at] TIME
    m = re.match(r"^next\s+(\w+)(?:\s+at)?\s*(.*)$", lower)
    if m and m.group(1) in _WEEKDAYS:
        target = _WEEKDAYS.index(m.group(1))
        days = (target - today.weekday()) % 7 or 7
        base = today + timedelta(days=days)
        rest = m.group(2).strip()
        if not rest:
            return base
        t = _parse_time(rest)
        if t is not None:
            return base.replace(hour=t[0], minute=t[1])

    # in N hours/minutes/days
    m = re.match(r"^in\s+(\d+)\s*(hour|hr|minute|min|day)s?\s*$", lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit in ("hour", "hr"):
            return now + timedelta(hours=n)
        if unit in ("minute", "min"):
            return now + timedelta(minutes=n)
        if unit == "day":
            return now + timedelta(days=n)

    # Bare time → today at that time.
    t = _parse_time(lower)
    if t is not None:
        return today.replace(hour=t[0], minute=t[1])

    # Last resort: dateutil.
    try:
        from dateutil import parser as dateutil_parser

        parsed = dateutil_parser.parse(s)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except Exception as exc:
        raise ValueError(f"could not parse datetime: {s!r}") from exc


def parse_dt_pair(s: str) -> Tuple[datetime, bool]:
    """Parse a date/datetime and report whether it carried explicit tz info."""
    s = (s or "").strip()
    if not s:
        raise ValueError("empty datetime string")
    try:
        if len(s) == 10:
            return datetime.fromisoformat(s), False
        candidate = s.replace("Z", "+00:00") if s.endswith("Z") else s
        parsed = datetime.fromisoformat(candidate)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None), True
        return parsed, False
    except ValueError:
        return parse_dt(s), False


def strip_tz(text: str) -> str:
    """Drop a trailing Z / ±HH:MM so an LLM-produced wall-clock time stays naive."""
    if not text:
        return text
    out = text.strip()
    if out.endswith(("Z", "z")):
        out = out[:-1]
    return re.sub(r"[+-]\d{2}:?\d{2}$", "", out)
