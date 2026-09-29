"""JSON persistence for completed and in-progress research records."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import Settings

logger = logging.getLogger(__name__)
_SESSION_ID_RE = re.compile(r"^rp-[a-f0-9]{12}$")


class ResearchStore:
    def __init__(self, settings: Settings) -> None:
        self.root = settings.resolved_research_data_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, research_id: str) -> Optional[Path]:
        if not isinstance(research_id, str) or not _SESSION_ID_RE.fullmatch(research_id):
            return None
        path = (self.root / f"{research_id}.json").resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            return None
        return path

    def save(self, research_id: str, record: Dict[str, Any]) -> None:
        path = self.path_for(research_id)
        if path is None:
            raise ValueError("Invalid research id")
        fd, temp_name = tempfile.mkstemp(prefix="research-", suffix=".tmp", dir=self.root)
        try:
            import os
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False)
            Path(temp_name).replace(path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def load(self, research_id: str) -> Optional[Dict[str, Any]]:
        path = self.path_for(research_id)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning("Could not read research record %s: %s", research_id, exc)
            return None

    def list(self, *, archived: Optional[bool] = None, limit: int = 50) -> List[Dict[str, Any]]:
        records = []
        for path in self.root.glob("rp-*.json"):
            data = self.load(path.stem)
            if not data:
                continue
            if archived is not None and bool(data.get("archived", False)) != archived:
                continue
            records.append({**data, "research_id": path.stem})
        records.sort(key=lambda row: row.get("completed_at", row.get("started_at", 0)), reverse=True)
        return records[:max(0, limit)]

    def delete(self, research_id: str) -> bool:
        path = self.path_for(research_id)
        if path is None or not path.exists():
            return False
        path.unlink()
        return True

    def detach_conversation(self, conversation_id: str) -> int:
        """Clear the conversation link on jobs that belong to it.

        Called when a conversation is deleted: the research record — and the
        report it produced — is preserved, only the association is dropped, so
        the report stays readable in the library.
        """
        changed = 0
        for path in self.root.glob("rp-*.json"):
            data = self.load(path.stem)
            if data and data.get("conversation_id") == conversation_id:
                data["conversation_id"] = None
                self.save(path.stem, data)
                changed += 1
        return changed
