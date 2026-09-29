"""Backwards-compatible re-export.

The implementation moved to `backend/text_utils.py` because Chat, RAG and Search
need it too (not just Deep Research). This module keeps the original import path
working, so `backend.research.*` and the existing tests are unaffected.
"""

from backend.text_utils import normalize_thinking_markup, strip_thinking

__all__ = ["strip_thinking", "normalize_thinking_markup"]
