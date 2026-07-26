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

from typing import Any


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


class ResultMemo:
    """Remembers one run's tool results so an identical call is answered from
    memory instead of re-hitting the network.

    From the 2026-07-26 trace: within one research fan-out the same article was
    fetched 8 times, a second one 6 times, and whole 4-query search sets were
    re-issued verbatim 90 seconds apart. The researcher is not learning anything
    from the repeat — it has lost track of what it already has.

    Replaying the identical summary is only half the fix, because a tool that
    answers instantly can be re-called in a tighter loop. So a replay is marked
    ``repeated``, which tells the model it is going in circles; the prompt turns
    that into a stop.

    Failures are remembered too, deliberately. A URL that 403s or blocks will do
    so again, and the researcher does not back off — it retries immediately, in
    a loop, against a wall clock. Serving the remembered failure ends that loop.
    The cost is that a genuinely transient error is not retried within the run,
    which is the cheaper mistake: one lost citation beats a topic that produces
    no brief at all because it spent its budget re-asking a dead URL.

    Scoped by construction: a factory creates one and closes over it, so the
    memo's lifetime is the lifetime of the tools built for that run.
    """

    def __init__(self) -> None:
        self._results: dict[str, dict[str, Any]] = {}

    def replay(self, key: str) -> dict[str, Any] | None:
        """The remembered summary for ``key``, flagged as a repeat, or None on
        a first call. The flag is the whole point: the artifact is unchanged,
        so the only new information the caller gets is that it already asked."""
        remembered = self._results.get(key)
        if remembered is None:
            return None
        return {
            **remembered,
            "repeated": True,
            "note": (
                "You already ran this exact call in this run — this is the "
                "remembered result, not a new one. Read the artifact at `path` "
                "and move on; repeating it again cannot produce new evidence."
            ),
        }

    def put(self, key: str, result: dict[str, Any]) -> None:
        self._results[key] = result


class AttemptBudget:
    """Caps how many times one key (a topic_id) may be retried per run."""

    def __init__(self, max_attempts: int) -> None:
        self.max_attempts = max(1, max_attempts)
        self._attempts: dict[str, int] = {}

    def exhausted(self, key: str) -> bool:
        return self._attempts.get(key, 0) >= self.max_attempts

    def record_attempt(self, key: str) -> None:
        self._attempts[key] = self._attempts.get(key, 0) + 1


__all__ = ["AttemptBudget", "ResultMemo", "SearchCircuit"]
