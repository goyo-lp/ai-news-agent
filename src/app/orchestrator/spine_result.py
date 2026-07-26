"""The propose spine's run summary — the contract between the pipeline and its
caller.

Kept apart from :mod:`app.orchestrator.spine` so the CLI can depend on the shape
of a run's outcome without depending on how the run is produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SpineStatus = Literal[
    "ok",
    "no_articles",
    "rank_failed",
    "no_topics",
    "no_selection",
    "nothing_above_floor",
]
"""Every way a run can end. Enumerated here so a caller can see the full set
without reading the pipeline."""


@dataclass(frozen=True)
class SelectedTopic:
    topic_id: str
    title: str
    domain: str
    score: float


@dataclass(frozen=True)
class TaskOutcome:
    """One subagent invocation's result: ``ok`` | ``timeout`` | ``error``."""

    topic_id: str
    status: str


@dataclass(frozen=True)
class SkippedTopic:
    topic_id: str
    reason: str


@dataclass(frozen=True)
class WrittenPost:
    topic_id: str
    post_id: str
    writer_status: str
    gate_passed: bool


@dataclass
class SpineResult:
    """One run's outcome. The artifacts themselves live on disk under the
    orchestrator data dir; this is the summary the CLI reports and the run
    report corroborates."""

    run_id: str
    status: SpineStatus
    fetched: int = 0
    topics: int = 0
    vetoed: list[dict[str, str]] = field(default_factory=list)
    selected: list[SelectedTopic] = field(default_factory=list)
    research: list[TaskOutcome] = field(default_factory=list)
    skipped_floor: list[SkippedTopic] = field(default_factory=list)
    written: list[WrittenPost] = field(default_factory=list)
    error: str | None = None

    @property
    def drafts_passed(self) -> int:
        """Derived, not stored — it cannot disagree with ``written``."""
        return sum(1 for post in self.written if post.gate_passed)


__all__ = [
    "SelectedTopic",
    "SkippedTopic",
    "SpineResult",
    "SpineStatus",
    "TaskOutcome",
    "WrittenPost",
]
