"""Calendar feature — local, SQLite-backed, no external service required.

Migrated from Odysseus's calendar (`routes/calendar_routes.py`,
`core/database.py` CalendarCal/CalendarEvent, `static/js/calendar.js`) with the
optional CalDAV subsystem left behind. Athena is an independent app: nothing here
imports Odysseus at runtime.
"""

from backend.calendar.dates import parse_dt, parse_dt_pair
from backend.calendar.store import CalendarStore

__all__ = ["CalendarStore", "parse_dt", "parse_dt_pair"]
