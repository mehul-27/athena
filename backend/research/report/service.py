"""Generate report HTML from a persisted Athena research record."""

from __future__ import annotations

from typing import Any, Dict

from backend.research.report.renderer import generate_visual_report


def render_record(research_id: str, record: Dict[str, Any]) -> str:
    return generate_visual_report(
        question=record.get("query", ""),
        report_markdown=record.get("raw_report") or record.get("result", ""),
        sources=record.get("sources") or [],
        stats=record.get("stats") or {},
        category=record.get("category"),
        session_id=research_id,
        hidden_images=record.get("hidden_images") or [],
    )
