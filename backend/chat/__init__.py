from backend.chat import normal, rag_chat, search_chat
from backend.chat.prompts import GROUNDING_SYSTEM_PROMPT, NO_CONTEXT_MESSAGE

# NOTE: `capabilities` is intentionally not imported here — it imports the other
# chat modules, so importing it from the package initialiser would be circular.
# Callers do `from backend.chat import capabilities`.
__all__ = ["normal", "rag_chat", "search_chat", "GROUNDING_SYSTEM_PROMPT", "NO_CONTEXT_MESSAGE"]
