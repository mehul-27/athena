from __future__ import annotations

from typing import Any, Dict, List

from backend.research.quality import is_low_quality


def extract_sources(findings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen = set()
    sources = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        url = finding.get("url", "")
        title = finding.get("title", "") or url
        summary = finding.get("summary", "") or finding.get("evidence", "")
        if url and url not in seen and not is_low_quality(summary):
            seen.add(url)
            source = {"url": url, "title": title}
            image = finding.get("og_image", "")
            if image:
                source["image"] = image
            sources.append(source)
    return sources


def extract_raw_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    items = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        url = finding.get("url", "")
        title = finding.get("title", "") or "Untitled"
        summary = finding.get("summary", "")
        evidence = finding.get("evidence", "")
        content = summary if summary else (evidence[:2000] if evidence else "")
        if url and content and not is_low_quality(content):
            items.append({"url": url, "title": title, "summary": content})
    return items
