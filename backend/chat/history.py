"""Conversation history helpers.

The UI keeps the conversation and sends prior turns back with each request so a
follow-up has context. History is untrusted input: it is filtered to the two
roles a chat may contain and capped so a long conversation can't crowd out the
system prompt or the retrieved/searched context.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

# How many prior turns to forward. Keeps the prompt bounded on long threads.
MAX_HISTORY_TURNS = 24

# A compaction summary is the one system turn that must survive history cleaning —
# it carries the context of the turns it replaced.
SUMMARY_PREFIX = "[Conversation summary]"


def clean_history(
    history: Optional[Iterable[Any]],
    *,
    max_turns: int = MAX_HISTORY_TURNS,
) -> List[Dict[str, str]]:
    """Return the tail of `history` as well-formed {role, content} turns."""
    cleaned: List[Dict[str, str]] = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if role not in ("user", "assistant") and not (
            role == "system" and content.startswith(SUMMARY_PREFIX)
        ):
            continue
        cleaned.append({"role": role, "content": content})
    if max_turns and len(cleaned) > max_turns:
        cleaned = cleaned[-max_turns:]
    return cleaned
