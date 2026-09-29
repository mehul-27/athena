"""System prompts and user-facing messages for the chat modes.

Keeping these in one module makes the "Chat is unconstrained, RAG Chat is
grounded-only" contract explicit and testable.
"""

# Chat: a plain assistant. May use general knowledge. No retrieval.
NORMAL_SYSTEM_PROMPT = "You are Athena, a helpful assistant. Answer the user's question directly."


def _identity_clause(identity: str) -> str:
    return f"If the user asks which model or AI you are, say you are running on {identity}."


def resolve_identity(llm, prefer: str | None = None) -> str:
    """Human-readable "Provider (model)" — but ONLY when it is guaranteed.

    Athena fails over between providers, so naming one up front can make the
    model assert something false: ask for Google Gemini, have it 404, let Groq
    answer — and a prompt that said "you are running on Google Gemini" makes the
    model claim Gemini while Groq served the request.

    So an identity is injected only when exactly **one** provider is usable and
    no failover is possible. Otherwise the model is told no vendor identity; the
    authoritative attribution is the response's `provider_label` ("via …").
    """
    describe = getattr(llm, "describe", None)
    if not callable(describe):
        return ""
    try:
        chain = list(describe() or [])
    except Exception:  # pragma: no cover - defensive
        return ""
    if not chain:
        return ""
    if prefer:
        # A preference is only safe to name if it is the sole candidate.
        target = prefer.strip().lower()
        if not any((e.get("name") or "").strip().lower() == target for e in chain):
            return ""
    if len(chain) != 1:
        return ""

    chosen = chain[0]
    label = (chosen.get("label") or chosen.get("name") or "").strip()
    model = (chosen.get("model") or "").strip()
    if label and model:
        return f"{label} ({model})"
    return model or label


def normal_system_prompt(identity: str = "") -> str:
    """Chat system prompt, telling the model which provider/model it is running on."""
    identity = (identity or "").strip()
    if not identity:
        return NORMAL_SYSTEM_PROMPT
    return (
        f"You are Athena, a helpful assistant running on {identity}. "
        "Answer the user's question directly. "
        f"{_identity_clause(identity)}"
    )


def search_system_prompt(identity: str = "") -> str:
    identity = (identity or "").strip()
    if not identity:
        return SEARCH_SYSTEM_PROMPT
    return (
        f"You are Athena, running on {identity}. Answer the user's question using the "
        "provided web search results. Cite the sources you use by their [n] numbers. "
        "If the results do not contain enough information, say so plainly instead of "
        f"guessing. {_identity_clause(identity)}"
    )

# RAG Chat: supplied verbatim per the RAG Chat requirements.
GROUNDING_SYSTEM_PROMPT = (
    "Answer only from the supplied document context. If the context does not "
    "contain enough information to answer the question, say that the "
    "information was not found in the documents. Do not use general knowledge "
    "to fill missing information."
)

# Returned without ever calling the LLM when no chunk clears the threshold.
NO_CONTEXT_MESSAGE = (
    "I couldn't find sufficiently relevant information in the documents "
    "available to this RAG chat."
)

# Search: grounded in search results (a separate capability from RAG Chat).
SEARCH_SYSTEM_PROMPT = (
    "You are Athena. Answer the user's question using the provided web search "
    "results. Cite the sources you use by their [n] numbers. If the results do "
    "not contain enough information, say so plainly instead of guessing."
)
