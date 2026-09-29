from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def strip_code_block(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_array(text: str) -> List[str]:
    text = strip_code_block(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass

    last_start = text.rfind("[")
    truncated = last_start != -1 and "]" not in text[last_start:]
    if truncated:
        complete_items = re.findall(r'"([^"]*)"', text[last_start:])
        if complete_items:
            logger.info("Repaired truncated JSON array: recovered %d items", len(complete_items))
            return complete_items

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass

    last_parsed = None
    for match in re.finditer(r"\[[\s\S]*?\]", text):
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                last_parsed = parsed
        except json.JSONDecodeError:
            continue
    if last_parsed is not None:
        return [str(item) for item in last_parsed]

    array_start = text.find("[")
    if array_start != -1:
        complete_items = re.findall(r'"([^"]*)"', text[array_start:])
        if complete_items:
            logger.info("Repaired truncated JSON array: recovered %d items", len(complete_items))
            return complete_items

    logger.warning("Could not parse JSON array from: %s", text[:200])
    return []


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = strip_code_block(text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def parse_json_array_items(text: str) -> List[Dict[str, Any]]:
    """Parse a JSON array of objects (used by batched extraction).

    Tolerant of code fences and surrounding prose; returns [] rather than
    raising so a failed batch can fall back to per-item extraction.
    """
    text = strip_code_block(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    # Last resort: recover individual top-level objects, in order.
    items: List[Dict[str, Any]] = []
    for obj_match in re.finditer(r"\{[^{}]*\}", text):
        try:
            parsed = json.loads(obj_match.group())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    if items:
        logger.info("Repaired batched JSON: recovered %d object(s)", len(items))
    return items

