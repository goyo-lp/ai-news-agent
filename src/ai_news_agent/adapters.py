"""Official-page adapter interface for sources without a native feed."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from ai_news_agent.sources import SourceConfig, SourceRouteStatus


@dataclass(slots=True)
class AdapterOutcome:
    """Result of attempting an official-page adapter fetch."""

    source_id: str
    available: bool
    reason: str
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class OfficialPageAdapter(Protocol):
    """Fetch article-like entries from an official page.

    Implementations must emit the same normalized article fields as RSS/Atom
    ingestion so downstream stages cannot distinguish the route. PR03 ships
    only the interface plus an explicit unavailable marker; per-publisher
    scraping lands in later work after validation.
    """

    source_id: str

    def fetch(self) -> AdapterOutcome:
        """Attempt collection, returning availability instead of raising."""
        ...


class UnavailableAdapter:
    """Default adapter that records why a source has no working route yet."""

    def __init__(self, source_id: str, reason: str) -> None:
        self.source_id = source_id
        self._reason = reason

    def fetch(self) -> AdapterOutcome:
        return AdapterOutcome(
            source_id=self.source_id,
            available=False,
            reason=self._reason,
        )


def adapter_for_source(config: SourceConfig) -> OfficialPageAdapter | None:
    """Return the adapter stub for adapter-route sources, else None."""

    if config.route.type.value != "official_page_adapter":
        return None
    if config.route.status == SourceRouteStatus.ACTIVE:
        # No validated adapter implementation ships in PR03; an active status
        # without an implementation is a configuration error surfaced as
        # unavailable rather than silent success.
        return UnavailableAdapter(
            config.id,
            "adapter marked active but no validated implementation ships in PR03",
        )
    return UnavailableAdapter(
        config.id,
        "official-page adapter not yet implemented; needs PR03 validation",
    )


__all__ = [
    "AdapterOutcome",
    "OfficialPageAdapter",
    "UnavailableAdapter",
    "adapter_for_source",
]
