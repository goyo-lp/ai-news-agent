"""Load and validate the 50-source ingestion roster."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, model_validator

from ai_news_agent.schemas import (
    RecordId,
    RouteType,
    Source,
    SourceCategory,
    SourceTier,
    Text,
)


class SourceRouteStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class SourceRoute(BaseModel):
    """Ingestion mechanism declared for one source."""

    model_config = ConfigDict(extra="forbid")

    type: RouteType
    url: AnyHttpUrl
    status: SourceRouteStatus = SourceRouteStatus.CANDIDATE


class SourceConfig(BaseModel):
    """One entry from config/sources.yaml with registry metadata."""

    model_config = ConfigDict(extra="forbid")

    id: RecordId
    name: Text
    publisher_family: RecordId
    category: SourceCategory
    tier: SourceTier
    homepage_url: AnyHttpUrl
    route: SourceRoute
    discovery_provenance: Text
    feed_scope: Text
    editorial_notes: tuple[Text, ...] = ()
    enabled: bool = False
    disabled_reason: Text | None = "pending-pr03-validation"
    last_checked_at: datetime | None = None
    last_success_at: datetime | None = None

    @model_validator(mode="after")
    def require_consistent_state(self) -> Self:
        if not self.enabled and self.disabled_reason is None:
            raise ValueError("disabled sources require disabled_reason")
        if self.enabled and self.disabled_reason is not None:
            raise ValueError("enabled sources cannot have disabled_reason")
        for value in (self.last_checked_at, self.last_success_at):
            if value is not None and value.utcoffset() is None:
                raise ValueError("timestamps must include a timezone")
        return self

    def to_source(self) -> Source:
        """Convert registry config into a validated Source record."""

        return Source(
            id=self.id,
            name=self.name,
            publisher_family=self.publisher_family,
            category=self.category,
            tier=self.tier,
            homepage_url=str(self.homepage_url),
            route_type=self.route.type,
            route_url=str(self.route.url),
            feed_scope=self.feed_scope,
            editorial_notes=self.editorial_notes,
            enabled=self.enabled,
            disabled_reason=self.disabled_reason,
            last_checked_at=self.last_checked_at,
            last_success_at=self.last_success_at,
        )


def load_source_configs(path: str | Path) -> tuple[SourceConfig, ...]:
    """Load and validate the YAML roster, rejecting duplicates and secrets."""

    raw_path = Path(path)
    payload = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("sources config must be a list of source entries")
    configs = tuple(SourceConfig.model_validate(item) for item in payload)
    _reject_duplicate_ids(configs)
    for config in configs:
        _reject_embedded_credentials(config)
    return configs


def load_sources_as_records(path: str | Path) -> tuple[Source, ...]:
    """Load the roster directly as validated Source records."""

    return tuple(config.to_source() for config in load_source_configs(path))


def _reject_duplicate_ids(configs: tuple[SourceConfig, ...]) -> None:
    seen: set[str] = set()
    for config in configs:
        if config.id in seen:
            raise ValueError(f"duplicate source id: {config.id}")
        seen.add(config.id)


def _reject_embedded_credentials(config: SourceConfig) -> None:
    for url in (str(config.homepage_url), str(config.route.url)):
        authority = url.split("://", 1)[-1].split("/", 1)[0]
        if "@" in authority:
            raise ValueError(f"source {config.id} URL must not embed credentials")


__all__ = [
    "SourceConfig",
    "SourceRoute",
    "SourceRouteStatus",
    "load_source_configs",
    "load_sources_as_records",
]
