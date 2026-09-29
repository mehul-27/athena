"""Cost estimation for LLM usage.

Inspired by Odysseus's `MODEL_PRICING` table (`static/js/chatRenderer.js`), but
with the honesty rules this task requires:

* a **free** model is genuinely $0 — including OpenRouter's `:free` suffix, and
  any provider configured as a local server (loopback/LAN) which never bills;
* a model with **no known price reports no cost** (`None`) rather than a guess —
  the UI renders `$—` while still showing the token counts;
* matching is longest-substring so `gpt-4o-mini` never picks up `gpt-4o`'s price.

The table is intentionally small and operator-editable: it holds list prices
(USD per 1M tokens) that are published and stable. Anything missing is treated as
unknown, not free.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# USD per 1,000,000 tokens: (input, output). Extend freely — an unknown model is
# reported as "no cost", never as a fabricated number.
PRICES: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Anthropic
    "claude-3-haiku": (0.25, 1.25),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
    # Google
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    # DeepSeek / Mistral
    "deepseek-chat": (0.27, 1.10),
    "mistral-large": (2.00, 6.00),
}


def _match_key(model: str) -> Optional[str]:
    """Longest known key contained in `model` (so `gpt-4o-mini` beats `gpt-4o`)."""
    model = (model or "").lower()
    if not model:
        return None
    best: Optional[str] = None
    for key in PRICES:
        if key in model and (best is None or len(key) > len(best)):
            best = key
    return best


def is_free_model(model: str) -> bool:
    """True for models the provider serves at no cost (`:free`, `-free` suffix)."""
    name = (model or "").strip().lower()
    return name.endswith(":free") or name.endswith("-free") or name.endswith("/free")


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    is_local: bool = False,
) -> Tuple[Optional[float], bool]:
    """Return `(cost_usd, known)`.

    `known=False` means "we have no price for this" — the caller must render an
    unknown marker rather than treating it as free. A genuinely free model
    returns `(0.0, True)`.
    """
    if is_local:
        return 0.0, True
    if is_free_model(model):
        return 0.0, True

    key = _match_key(model)
    if key is None:
        return None, False

    in_price, out_price = PRICES[key]
    cost = (max(int(input_tokens or 0), 0) * in_price
            + max(int(output_tokens or 0), 0) * out_price) / 1_000_000
    return cost, True


def format_cost(cost: Optional[float]) -> str:
    """Display string: `$—` when unknown, else a sensible precision."""
    if cost is None:
        return "$—"
    if cost == 0:
        return "$0.0000"
    if cost < 0.01:
        return f"${cost:.4f}"
    if cost < 1:
        return f"${cost:.3f}"
    return f"${cost:.2f}"
