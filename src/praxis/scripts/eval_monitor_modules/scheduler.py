"""Scheduling decisions for eval monitor."""

from __future__ import annotations

import time
from typing import Any


def should_preempt_current_eval(
    *,
    current_tier: str | None,
    manifest: list[dict[str, Any]],
    evaluated: set[int],
    failed: set[int],
) -> bool:
    """Return whether a running eval should be preempted for a best-tier candidate."""
    if current_tier != "recent":
        return False
    return any(
        entry["tier"] == "best"
        and entry["step"] not in evaluated
        and entry["step"] not in failed
        for entry in manifest
    )


def stale_watchdog_expired(*, last_progress_ts: float, stale_timeout_sec: int) -> bool:
    """Return whether stale-timeout threshold has been exceeded."""
    if stale_timeout_sec <= 0:
        return False
    return (time.time() - last_progress_ts) >= stale_timeout_sec
