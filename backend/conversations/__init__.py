"""Conversation persistence + usage tracking (Athena-native).

`ConversationStore` is the durable model behind the chat header, rename,
compact, delete, saved documents, and the Deep Research ↔ conversation link.
"""

from backend.conversations.store import DEFAULT_TITLE, ConversationStore

__all__ = ["ConversationStore", "DEFAULT_TITLE"]
