"""Calendar migration: storage semantics, routes, recurrence, ICS, quick add.

Behaviour is asserted against the same contracts Odysseus's calendar uses:
local SQLite is the source of truth, events overlap the query window, RRULEs are
expanded server-side into compound occurrence UIDs, and tz-aware input is stored
as UTC with an `is_utc` flag so it serializes with a `Z`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.calendar.dates import parse_dt, parse_dt_pair
from backend.calendar.store import CalendarStore
from backend.config import Settings
from backend.main import create_app

DAY = "2026-09-24"


def make_settings(tmp_path, **over):
    base = dict(
        _env_file=None,
        embedding_provider="hashing",
        chroma_mode="embedded",
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "docs"),
        chroma_path=str(tmp_path / "chroma"),
        rag_collection="calendar-test",
        nvidia_api_key="", groq_api_key="", openrouter_api_key="", google_api_key="",
    )
    base.update(over)
    return Settings(**base)


@pytest.fixture
def store(tmp_path):
    return CalendarStore(make_settings(tmp_path))


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


# ----------------------------------------------------------------------
# date parsing
# ----------------------------------------------------------------------
def test_natural_language_dates_parse():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    assert parse_dt("tomorrow 3pm") == today + timedelta(days=1, hours=15)
    assert parse_dt("3pm tomorrow") == today + timedelta(days=1, hours=15)
    assert parse_dt("today at 09:30") == today + timedelta(hours=9, minutes=30)
    assert parse_dt("2026-05-13T10:00:00") == datetime(2026, 5, 13, 10, 0, 0)


def test_tz_aware_input_is_stored_as_utc():
    value, is_utc = parse_dt_pair("2026-05-13T10:00:00+09:00")
    assert is_utc is True
    assert value == datetime(2026, 5, 13, 1, 0, 0)
    naive, naive_utc = parse_dt_pair("2026-05-13T10:00:00")
    assert naive_utc is False and naive == datetime(2026, 5, 13, 10, 0, 0)


# ----------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------
def test_default_calendar_is_created_lazily(store):
    calendars = store.list_calendars()
    assert len(calendars) == 1
    assert calendars[0]["name"] == "Personal"
    assert calendars[0]["source"] == "local"


def test_create_list_update_delete_calendar(store):
    created = store.create_calendar("Work", "#ff0000")
    assert any(c["name"] == "Work" for c in store.list_calendars())
    assert store.update_calendar(created["id"], name="Job", color="#00ff00")
    row = next(c for c in store.list_calendars() if c["href"] == created["id"])
    assert row["name"] == "Job" and row["color"] == "#00ff00"
    assert store.delete_calendar(created["id"])
    assert not any(c["href"] == created["id"] for c in store.list_calendars())


def test_event_crud_and_range_filtering(store):
    created = store.create_event({"summary": "Standup", "dtstart": f"{DAY}T09:00:00",
                                  "dtend": f"{DAY}T09:30:00", "location": "Room 1"})
    uid = created["uid"]

    window = store.list_events(datetime(2026, 9, 1), datetime(2026, 10, 1))
    assert [e["uid"] for e in window["events"]] == [uid]
    ev = window["events"][0]
    assert ev["summary"] == "Standup" and ev["location"] == "Room 1"
    assert ev["dtstart"] == f"{DAY}T09:00:00"
    assert ev["calendar"] == "Personal"

    # A window that doesn't overlap returns nothing.
    assert store.list_events(datetime(2026, 10, 1), datetime(2026, 11, 1))["events"] == []

    assert store.update_event(uid, {"summary": "Renamed", "dtstart": f"{DAY}T10:00:00",
                                    "dtend": f"{DAY}T10:30:00"})
    assert store.get_event(uid)["summary"] == "Renamed"

    assert store.delete_event(uid)["ok"] is True
    assert store.get_event(uid) is None


def test_all_day_event_serialises_as_date_only(store):
    store.create_event({"summary": "Holiday", "dtstart": DAY, "dtend": DAY, "all_day": True})
    ev = store.list_events(datetime(2026, 9, 1), datetime(2026, 10, 1))["events"][0]
    assert ev["all_day"] is True
    assert ev["dtstart"] == DAY and ev["dtend"] == DAY


def test_tz_aware_event_round_trips_with_z_suffix(store):
    store.create_event({"summary": "Call", "dtstart": "2026-05-13T10:00:00+09:00",
                        "dtend": "2026-05-13T11:00:00+09:00"})
    ev = store.list_events(datetime(2026, 5, 1), datetime(2026, 6, 1))["events"][0]
    assert ev["is_utc"] is True
    assert ev["dtstart"].endswith("Z")
    assert ev["dtstart"].startswith("2026-05-13T01:00:00")


def test_persistence_survives_a_new_store(tmp_path):
    settings = make_settings(tmp_path)
    CalendarStore(settings).create_event({"summary": "Persisted", "dtstart": f"{DAY}T08:00:00",
                                          "dtend": f"{DAY}T09:00:00"})
    reopened = CalendarStore(settings)
    events = reopened.list_events(datetime(2026, 9, 1), datetime(2026, 10, 1))["events"]
    assert [e["summary"] for e in events] == ["Persisted"]
    assert (settings.resolved_data_dir() / "calendar.db").exists()


def test_end_is_defaulted_to_one_hour(store):
    store.create_event({"summary": "Quick", "dtstart": f"{DAY}T09:00:00"})
    ev = store.list_events(datetime(2026, 9, 1), datetime(2026, 10, 1))["events"][0]
    assert ev["dtend"] == f"{DAY}T10:00:00"


# ----------------------------------------------------------------------
# recurrence
# ----------------------------------------------------------------------
def test_rrule_expands_into_occurrences_in_window(store):
    store.create_event({"summary": "Daily", "dtstart": "2026-09-21T09:00:00",
                        "dtend": "2026-09-21T09:30:00", "rrule": "FREQ=DAILY"})
    events = store.list_events(datetime(2026, 9, 23), datetime(2026, 9, 26))["events"]
    assert len(events) == 3
    assert all(e["is_recurrence"] for e in events)
    assert [e["dtstart"][:10] for e in events] == ["2026-09-23", "2026-09-24", "2026-09-25"]
    # compound occurrence uid keeps the series recoverable
    assert all(e["uid"].startswith(e["series_uid"] + "::") for e in events)


def test_rrule_recurs_beyond_the_dtstart_year(store):
    store.create_event({"summary": "Yearly-ish", "dtstart": "2024-03-01T09:00:00",
                        "dtend": "2024-03-01T10:00:00", "rrule": "FREQ=MONTHLY"})
    events = store.list_events(datetime(2026, 3, 1), datetime(2026, 4, 1))["events"]
    assert len(events) == 1
    assert events[0]["dtstart"].startswith("2026-03-01")


def test_deleting_one_occurrence_excludes_it_from_the_series(store):
    created = store.create_event({"summary": "Daily", "dtstart": "2026-09-21T09:00:00",
                                  "dtend": "2026-09-21T09:30:00", "rrule": "FREQ=DAILY"})
    window = (datetime(2026, 9, 23), datetime(2026, 9, 26))
    occurrences = store.list_events(*window)["events"]
    assert len(occurrences) == 3

    target = occurrences[1]["uid"]
    result = store.delete_event(target, scope="occurrence")
    assert result["scope"] == "occurrence"

    remaining = store.list_events(*window)["events"]
    assert len(remaining) == 2
    assert target not in [e["uid"] for e in remaining]
    # the series itself still exists
    assert store.get_event(created["uid"]) is not None


def test_deleting_the_series_removes_everything(store):
    created = store.create_event({"summary": "Daily", "dtstart": "2026-09-21T09:00:00",
                                  "dtend": "2026-09-21T09:30:00", "rrule": "FREQ=DAILY"})
    assert store.delete_event(f"{created['uid']}::2026-09-24")["scope"] == "series"
    assert store.list_events(datetime(2026, 9, 1), datetime(2026, 10, 1))["events"] == []


def test_malformed_rrule_degrades_to_a_single_event(store):
    store.create_event({"summary": "Broken", "dtstart": f"{DAY}T09:00:00",
                        "dtend": f"{DAY}T10:00:00", "rrule": "FREQ=NONSENSE"})
    events = store.list_events(datetime(2026, 9, 1), datetime(2026, 10, 1))["events"]
    assert len(events) == 1 and events[0]["summary"] == "Broken"


# ----------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------
def test_calendar_routes_crud(client):
    assert client.get("/api/calendar/calendars").status_code == 200
    created = client.post("/api/calendar/calendars", json={"name": "Work", "color": "#123456"}).json()
    assert created["ok"] is True

    event = client.post("/api/calendar/events", json={
        "summary": "Review", "dtstart": f"{DAY}T14:00:00", "dtend": f"{DAY}T15:00:00",
    }).json()
    assert event["ok"] is True

    listed = client.get(f"/api/calendar/events?start={DAY}&end=2026-09-25").json()
    assert [e["summary"] for e in listed["events"]] == ["Review"]

    uid = listed["events"][0]["uid"]
    assert client.put(f"/api/calendar/events/{uid}", json={"summary": "Reviewed"}).status_code == 200
    assert client.delete(f"/api/calendar/events/{uid}").json()["ok"] is True
    assert client.get(f"/api/calendar/events?start={DAY}&end=2026-09-25").json()["events"] == []


def test_malformed_range_returns_empty_not_500(client):
    body = client.get("/api/calendar/events?start=NaN-NaN-NaN&end=also-bad").json()
    assert body == {"events": []}


def test_delete_unknown_event_is_404(client):
    assert client.delete("/api/calendar/events/does-not-exist").status_code == 404


def test_ics_export_and_reimport_round_trip(client):
    client.post("/api/calendar/events", json={
        "summary": "Lunch", "dtstart": f"{DAY}T12:00:00", "dtend": f"{DAY}T13:00:00",
        "location": "Cafe, downtown",
    })
    cal_id = client.get("/api/calendar/calendars").json()["calendars"][0]["href"]
    exported = client.get(f"/api/calendar/export/{cal_id}")
    assert exported.status_code == 200
    assert "text/calendar" in exported.headers["content-type"]
    assert "BEGIN:VEVENT" in exported.text
    assert "SUMMARY:Lunch" in exported.text
    assert "LOCATION:Cafe\\, downtown" in exported.text  # comma escaped per RFC 5545

    imported = client.post(
        "/api/calendar/import",
        files={"file": ("roundtrip.ics", exported.text.encode(), "text/calendar")},
        params={"calendar_name": "Reimported"},
    ).json()
    assert imported["ok"] is True and imported["imported"] == 1
    assert any(c["name"] == "Reimported" for c in client.get("/api/calendar/calendars").json()["calendars"])


def test_ics_import_rejects_garbage(client):
    resp = client.post("/api/calendar/import",
                       files={"file": ("bad.ics", b"not an ics file", "text/calendar")})
    assert resp.status_code == 400


class JsonLLM:
    """Stub router that answers quick-parse with a fixed JSON event."""

    configured = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    @property
    def provider_names(self):
        return ["stub"]

    @property
    def skipped_names(self):
        return []

    def describe(self):
        return [{"name": "stub", "model": "stub", "label": "stub"}]

    def chat(self, messages, **kwargs):
        from backend.providers.router import LLMResult

        self.calls.append(messages)
        return LLMResult(content=json.dumps(self.payload), provider="stub", model="stub", label="stub")


def test_quick_parse_returns_structured_event(tmp_path):
    llm = JsonLLM({
        "summary": "Lunch with Sara", "dtstart": f"{DAY}T13:00:00", "dtend": f"{DAY}T14:00:00",
        "all_day": False, "location": "Downtown", "description": "", "confidence": 0.9,
    })
    app = create_app(make_settings(tmp_path), llm=llm)
    with TestClient(app) as c:
        body = c.post("/api/calendar/quick-parse", json={"text": "lunch with sara 1pm downtown"}).json()
    assert body["ok"] is True
    assert body["event"]["summary"] == "Lunch with Sara"
    assert body["event"]["dtstart"] == f"{DAY}T13:00:00"
    assert body["event"]["location"] == "Downtown"
    assert body["confidence"] == 0.9


def test_quick_parse_strips_time_tokens_from_summary(tmp_path):
    llm = JsonLLM({"summary": "Standup 9am (09:00)", "dtstart": f"{DAY}T09:00:00",
                   "dtend": f"{DAY}T09:15:00", "all_day": False})
    app = create_app(make_settings(tmp_path), llm=llm)
    with TestClient(app) as c:
        body = c.post("/api/calendar/quick-parse", json={"text": "standup 9am"}).json()
    assert body["event"]["summary"] == "Standup"


def test_quick_parse_handles_unparseable_model_output(tmp_path):
    class Garbage(JsonLLM):
        def chat(self, messages, **kwargs):
            from backend.providers.router import LLMResult

            return LLMResult(content="I'm sorry, I can't help with that.",
                             provider="stub", model="stub", label="stub")

    app = create_app(make_settings(tmp_path), llm=Garbage({}))
    with TestClient(app) as c:
        body = c.post("/api/calendar/quick-parse", json={"text": "???"}).json()
    assert body["ok"] is False
    assert "JSON" in body["error"]


def test_calendar_endpoints_are_not_cacheable(client):
    assert client.get("/api/calendar/calendars").headers.get("cache-control") == "no-store"
