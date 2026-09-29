"""Reasoning-trace stripping for LLM output.

Models vary in how they mark up their chain of thought — `<think>`/`<thinking>`/
`<thought>` tags, Gemma's `<|channel>thought … <channel|>` framing, or Qwen's
leading "Thinking Process:" preamble. Athena only ever wants to show the answer,
so this normalises and removes those traces.

Lives outside the research package because every answering path needs it
(Chat, RAG and Search, not just Deep Research). `backend/research/text.py`
re-exports these names for compatibility.
"""

from __future__ import annotations

import re

_THINK_TAG_NAME = r"(?:think(?:ing)?|thought)"
_THINK_OPEN_TAG_RE = re.compile(rf"<{_THINK_TAG_NAME}(?:\s[^<>]*)?>", re.IGNORECASE)
_THINK_CLOSE_TAG_RE = re.compile(rf"</{_THINK_TAG_NAME}>\s*", re.IGNORECASE)
_THINK_TAG_RE = re.compile(rf"</?{_THINK_TAG_NAME}[^<>]*>\s*", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(rf"<{_THINK_TAG_NAME}(?:\s[^<>]*)?>[\s\S]*$", re.IGNORECASE)
_THINK_ATTR_RE = re.compile(rf"<{_THINK_TAG_NAME}\s[^<>]*>", re.IGNORECASE)
_THINK_ATTR_CLOSE_RE = re.compile(rf"</{_THINK_TAG_NAME}\s[^<>]*>", re.IGNORECASE)
_GEMMA_THOUGHT_OPEN_RE = re.compile(r"<\|channel>thought\s*\n?[\s\S]*$", re.IGNORECASE)
_GEMMA_RESPONSE_OPEN_RE = re.compile(r"<\|channel>response\s*\n?", re.IGNORECASE)
_GEMMA_CHANNEL_CLOSE_RE = re.compile(r"<channel\|>", re.IGNORECASE)
_THOUGHT_TAG_OPEN_RE = re.compile(r"<thought(\s[^<>]*)?>", re.IGNORECASE)
_THOUGHT_TAG_CLOSE_RE = re.compile(r"</thought>", re.IGNORECASE)
_GEMMA_THOUGHT_CHANNEL_OPEN_RE = re.compile(r"<\|channel>thought\s*\n?", re.IGNORECASE)
_GEMMA_CHANNEL_CLOSE_TRIM_RE = re.compile(r"<channel\|>\s*", re.IGNORECASE)
_QWEN_THINKING_RE = re.compile(r"^Thinking Process:.*?(?=\n\n#|\n\n\*\*|\Z)", re.IGNORECASE | re.DOTALL)
_PROMPT_ECHO_RES = (
    re.compile(r"^The user asks:.*?(?=\n\n#|\n\n\*\*[A-Z]|\Z)", re.DOTALL),
    re.compile(r"^We need to.*?(?=\n\n#|\n\n\*\*[A-Z]|\Z)", re.DOTALL),
)


def _sub_delimited(text: str, open_re, close_re, replace_inner) -> str:
    out = []
    pos = 0
    while True:
        opener = open_re.search(text, pos)
        if opener is None:
            break
        closer = close_re.search(text, opener.end())
        if closer is None:
            break
        out.append(text[pos:opener.start()])
        out.append(replace_inner(text[opener.end():closer.start()]))
        pos = closer.end()
    out.append(text[pos:])
    return "".join(out)


def normalize_thinking_markup(text: str) -> str:
    if not text:
        return text
    out = _THOUGHT_TAG_OPEN_RE.sub(lambda m: "<think" + (m.group(1) or "") + ">", text)
    out = _THOUGHT_TAG_CLOSE_RE.sub("</think>", out)

    def replace_gemma_thought(inner: str) -> str:
        thought = inner.strip()
        return f"<think>{thought}</think>\n" if thought else ""

    out = _sub_delimited(out, _GEMMA_THOUGHT_CHANNEL_OPEN_RE, _GEMMA_CHANNEL_CLOSE_TRIM_RE, replace_gemma_thought)
    out = _sub_delimited(out, _GEMMA_RESPONSE_OPEN_RE, _GEMMA_CHANNEL_CLOSE_RE, lambda inner: inner)
    out = _GEMMA_RESPONSE_OPEN_RE.sub("", out)
    return _GEMMA_CHANNEL_CLOSE_RE.sub("", out)


def strip_thinking(text: str | None) -> str:
    """Remove tagged reasoning and leading Qwen/prompt-echo prose from LLM output."""
    if not text:
        return ""
    out = normalize_thinking_markup(text)
    out = _GEMMA_THOUGHT_OPEN_RE.sub("", out)
    out = _THINK_ATTR_RE.sub("<think>", out)
    out = _THINK_ATTR_CLOSE_RE.sub("</think>", out)
    out = _sub_delimited(out, _THINK_OPEN_TAG_RE, _THINK_CLOSE_TAG_RE, lambda inner: "")
    out = _THINK_OPEN_RE.sub("", out)
    out = _THINK_TAG_RE.sub("", out)
    out = _QWEN_THINKING_RE.sub("", out)
    for matcher in _PROMPT_ECHO_RES:
        out = matcher.sub("", out)
    return out.strip()
