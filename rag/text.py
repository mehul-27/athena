"""Shared text helpers (tokenization + stop words).

The stop-word list mirrors Odysseus's `PersonalDocsConfig.STOP_WORDS` so the
keyword half of hybrid search and the offline `hashing` embedding provider agree
on which tokens carry signal. Filtering stop words is what lets a fixed
relevance threshold cleanly separate an on-topic query from an off-topic one.
"""

from __future__ import annotations

import re
from typing import List, Set

STOP_WORDS: Set[str] = set(
    """
    the a an is are was were be been being to of in for on at by with from
    and or if then else when while as it this that those these i you he she
    we they my your our their me him her us them
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, minus stop words and single characters."""
    if not isinstance(text, str):
        return []
    return [
        t
        for t in re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        if len(t) > 1 and t not in STOP_WORDS
    ]


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))
