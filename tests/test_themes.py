"""Theme migration: preset catalogue, persistence endpoint, and UI wiring.

The JavaScript behaviour (apply/persist/customize) is exercised manually in the
browser; these tests cover what the backend owns (the prefs store behind theme
sync) plus structural guarantees about the shipped theme module and markup.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import ATHENA_ROOT, Settings
from backend.main import create_app

FRONTEND = ATHENA_ROOT / "frontend"
THEME_JS = FRONTEND / "theme.js"

# Odysseus's preset ids, verbatim.
EXPECTED_THEMES = [
    "dark", "light", "midnight", "paper", "cyberpunk", "retrowave", "forest",
    "ocean", "ume", "copper", "terminal", "organs", "lavender", "gpt", "claude", "cute",
]


def make_settings(tmp_path):
    return Settings(
        _env_file=None, embedding_provider="hashing", chroma_mode="embedded",
        data_dir=str(tmp_path / "data"), documents_dir=str(tmp_path / "docs"),
        chroma_path=str(tmp_path / "chroma"), rag_collection="theme-test",
        nvidia_api_key="", groq_api_key="", openrouter_api_key="", google_api_key="",
    )


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


# ----------------------------------------------------------------------
# prefs store (theme sync target)
# ----------------------------------------------------------------------
def test_theme_pref_starts_empty(client):
    assert client.get("/api/prefs/theme").json() == {"key": "theme", "value": None}


def test_theme_pref_round_trips(client):
    payload = {"name": "ocean", "colors": {"bg": "#0b1a2c", "fg": "#64d2ff",
                                           "panel": "#091422", "border": "#1e5074", "red": "#4facfe"}}
    saved = client.put("/api/prefs/theme", json={"value": payload}).json()
    assert saved["value"] == payload
    assert client.get("/api/prefs/theme").json()["value"] == payload


def test_theme_pref_survives_restart(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        c.put("/api/prefs/theme", json={"value": {"name": "cyberpunk", "colors": {"bg": "#0a0a0f"}}})
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/api/prefs/theme").json()["value"]["name"] == "cyberpunk"


def test_custom_themes_pref_round_trips(client):
    custom = {"sakura": {"bg": "#2b1b2e", "fg": "#f5c2e7", "panel": "#1e1420",
                         "border": "#6c4675", "red": "#f5a0c0"}}
    client.put("/api/prefs/custom-themes", json={"value": custom})
    assert client.get("/api/prefs/custom-themes").json()["value"] == custom


def test_unknown_pref_key_is_rejected(client):
    assert client.get("/api/prefs/nonsense").status_code == 404
    assert client.put("/api/prefs/nonsense", json={"value": 1}).status_code == 404


def test_prefs_responses_are_not_cacheable(client):
    assert client.get("/api/prefs/theme").headers.get("cache-control") == "no-store"


# ----------------------------------------------------------------------
# shipped theme module / markup
# ----------------------------------------------------------------------
def test_theme_module_defines_every_odysseus_preset():
    source = THEME_JS.read_text(encoding="utf-8")
    for name in EXPECTED_THEMES:
        assert re.search(rf"^\s+{name}:\s*\{{", source, re.MULTILINE), f"missing theme: {name}"


def test_theme_module_carries_odysseus_palette_values():
    source = THEME_JS.read_text(encoding="utf-8")
    # Spot-check the exact palette values from Odysseus's THEMES object.
    assert '#282c34' in source and '#9cdef2' in source          # dark / "original"
    assert '#0a0a0f' in source and '#9b30ff' in source          # cyberpunk
    assert '#212121' in source and '#949494' in source          # gpt
    assert '"original"' in source and '"GPT"' in source         # display labels
    # Behaviour ported from Odysseus:
    assert "generateHarmonyColors" in source
    assert "MAX_CUSTOM_THEMES = 8" in source
    assert "athena-theme" in source and "athena-custom-themes" in source


def test_theme_module_derives_athena_tokens():
    source = THEME_JS.read_text(encoding="utf-8")
    for token in ("--panel-2", "--card", "--border-strong", "--fg-dim", "--muted",
                  "--accent", "--accent-2"):
        assert token in source, f"theme module does not set {token}"


def test_theme_markup_and_entry_point(client):
    html = client.get("/").text
    assert 'id="theme-modal"' in html
    assert 'id="open-theme"' in html
    assert 'data-tab="browse"' in html and 'data-tab="customize"' in html
    assert "Default Themes" in html
    assert 'id="themeGrid"' in html
    assert "/static/theme.js" in html
    assert client.get("/static/theme.js").status_code == 200


def test_favicon_is_served(client):
    assert client.get("/static/favicon.svg").status_code == 200
