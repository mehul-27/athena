"""In-memory provider health/rate-limit state.

Runtime, per-process state only — never persisted. It exists so the Settings UI
can show something more useful than "configured or not": whether a provider is
currently rate-limited or erroring, and when it last succeeded.

Deliberately tiny. Odysseus has no equivalent (its 429 handling is purely a
call-time fallback trigger), so this is a small new capability, not a port.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# A provider that returned 429 is shown as "rate limited" for this long, then
# reverts to "available" on its own. No background probing is involved.
RATE_LIMIT_COOLDOWN = 60.0


@dataclass
class ProviderHealth:
    last_status: str = "unknown"  # unknown | ok | error
    last_status_code: Optional[int] = None
    last_error: str = ""
    last_error_at: float = 0.0
    last_success_at: float = 0.0
    last_model: str = ""
    rate_limited_until: float = 0.0
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0

    def rate_limited(self, now: Optional[float] = None) -> bool:
        return self.rate_limited_until > (now if now is not None else time.time())


class HealthTracker:
    """Records per-provider outcomes as the router runs."""

    def __init__(self, rate_limit_cooldown: float = RATE_LIMIT_COOLDOWN) -> None:
        self._cooldown = max(0.0, float(rate_limit_cooldown))
        self._by_name: Dict[str, ProviderHealth] = {}

    def _entry(self, name: str) -> ProviderHealth:
        entry = self._by_name.get(name)
        if entry is None:
            entry = ProviderHealth()
            self._by_name[name] = entry
        return entry

    def record_success(self, name: str, *, model: str = "") -> None:
        entry = self._entry(name)
        entry.last_status = "ok"
        entry.last_status_code = None
        entry.last_error = ""
        entry.last_success_at = time.time()
        entry.last_model = model or entry.last_model
        entry.rate_limited_until = 0.0
        entry.consecutive_failures = 0
        entry.total_calls += 1

    def record_failure(self, name: str, *, status: Optional[int] = None, message: str = "") -> None:
        entry = self._entry(name)
        entry.last_status = "error"
        entry.last_status_code = status
        entry.last_error = (message or "")[:300]
        entry.last_error_at = time.time()
        entry.consecutive_failures += 1
        entry.total_calls += 1
        entry.total_failures += 1
        if status == 429:
            entry.rate_limited_until = time.time() + self._cooldown

    def get(self, name: str) -> ProviderHealth:
        return self._by_name.get(name, ProviderHealth())

    def snapshot(self) -> Dict[str, ProviderHealth]:
        return dict(self._by_name)

    def reset_provider(self, name: str) -> None:
        """Forget a provider's runtime state (used when it is removed)."""
        self._by_name.pop(name, None)

    def reset(self) -> None:
        self._by_name.clear()


def health_summary(health: ProviderHealth, *, now: Optional[float] = None) -> Tuple[str, str]:
    """Return (status_token, last_error) for UI display."""
    now = now if now is not None else time.time()
    if health.rate_limited(now):
        return "rate_limited", health.last_error
    if health.last_status == "error":
        return "error", health.last_error
    if health.last_status == "ok":
        return "available", ""
    return "available", ""
