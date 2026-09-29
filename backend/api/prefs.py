"""Small key/value preference store (UI state that should outlive the browser).

Athena is local-only, so this is a plain JSON file under `data/`. It exists so
things like the selected theme survive more than `localStorage` (a cleared
browser profile, a different browser, a fresh install over the same data dir) —
the same local-file + API shape Odysseus uses for its per-user prefs, without its
database or user model.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prefs", tags=["prefs"])

# Keys the UI is allowed to persist. Keeps this from becoming an arbitrary
# remote-write surface and documents what the frontend actually stores.
ALLOWED_KEYS = ("theme", "custom-themes", "calendar")


class PrefValue(BaseModel):
    value: Any = None


class PrefStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s: %s", self.path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="ui_prefs-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
            Path(temp_name).replace(self.path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def get(self, key: str) -> Optional[Any]:
        return self.load().get(key)

    def put(self, key: str, value: Any) -> Any:
        with self._lock:
            data = self.load()
            data[key] = value
            self.save(data)
        return value


def _store(request: Request) -> PrefStore:
    store = getattr(request.app.state, "pref_store", None)
    if store is None:
        store = PrefStore(request.app.state.settings.resolved_data_dir() / "ui_prefs.json")
        request.app.state.pref_store = store
    return store


def _validate(key: str) -> str:
    if key not in ALLOWED_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown preference: {key}")
    return key


@router.get("/{key}")
def get_pref(key: str, request: Request) -> dict:
    key = _validate(key)
    return {"key": key, "value": _store(request).get(key)}


@router.put("/{key}")
def put_pref(key: str, body: PrefValue, request: Request) -> dict:
    key = _validate(key)
    return {"key": key, "value": _store(request).put(key, body.value)}
