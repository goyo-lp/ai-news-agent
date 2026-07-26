"""Per-run spend guards for the research tools.

Both guards exist because of the 2026-07-25 production trace: 57 web searches
against a silently-dead SearXNG instance, and repeated re-verification of the
same brief. Each was first implemented as a module-level counter, which made
them process-global — the orchestrator had to remember to reset them between
runs, and two lanes in one process (`both`) shared one budget.

These are plain objects instead. A tool factory creates one and closes over it,
so a budget's lifetime is exactly the lifetime of the tools built for that run,
and nothing has to be reset.
"""
from __future__ import annotations


class SearchCircuit:
    """Trips after ``threshold`` consecutive empty-or-failed searches.

    A search that returns results resets the count: the guard is aimed at a
    backend that has stopped producing evidence entirely, not at individual
    queries that legitimately match nothing.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = max(1, threshold)
        self.consecutive_empty = 0

    @property
    def is_open(self) -> bool:
        return self.consecutive_empty >= self.threshold

    def record(self, *, found_results: bool) -> None:
        if found_results:
            self.consecutive_empty = 0
        else:
            self.consecutive_empty += 1


class AttemptBudget:
    """Caps how many times one key (a topic_id) may be retried per run."""

    def __init__(self, max_attempts: int) -> None:
        self.max_attempts = max(1, max_attempts)
        self._attempts: dict[str, int] = {}

    def exhausted(self, key: str) -> bool:
        return self._attempts.get(key, 0) >= self.max_attempts

    def record_attempt(self, key: str) -> None:
        self._attempts[key] = self._attempts.get(key, 0) + 1


__all__ = ["AttemptBudget", "SearchCircuit"]
