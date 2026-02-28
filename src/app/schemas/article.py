from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class FetchRules(BaseModel):
    image_fallback_rss_enclosure: bool = True
    requires_user_agent: bool = True
    blocked_domains: list[str] = Field(default_factory=list)


class SourceFetchOverrides(BaseModel):
    image_fallback_rss_enclosure: bool | None = None
    requires_user_agent: bool | None = None
    blocked_domains: list[str] | None = None


class SourceConfig(BaseModel):
    name: str
    url: str
    rss: str
    fetch_overrides: SourceFetchOverrides | None = None

    def merged_rules(self, defaults: FetchRules) -> FetchRules:
        overrides = self.fetch_overrides
        if overrides is None:
            return defaults
        return FetchRules(
            image_fallback_rss_enclosure=(
                overrides.image_fallback_rss_enclosure
                if overrides.image_fallback_rss_enclosure is not None
                else defaults.image_fallback_rss_enclosure
            ),
            requires_user_agent=(
                overrides.requires_user_agent
                if overrides.requires_user_agent is not None
                else defaults.requires_user_agent
            ),
            blocked_domains=(
                overrides.blocked_domains
                if overrides.blocked_domains is not None
                else defaults.blocked_domains
            ),
        )


class SourcesFile(BaseModel):
    fetch_defaults: FetchRules = Field(default_factory=FetchRules)
    sources: list[SourceConfig]


class Article(BaseModel):
    id: str
    source_name: str
    source_rss: str
    source_url: str | None = None
    title: str
    url: str
    published_at: datetime | None = None
    description: str | None = None
    rss_image_url: str | None = None
    og_title: str | None = None
    og_description: str | None = None
    image_url: str | None = None
    summary: str | None = None
    score: float | None = None
    duplicate_count: int = 1
    cluster_id: str | None = None
    cluster_size: int = 1

    @property
    def effective_title(self) -> str:
        return self.og_title or self.title

    @property
    def effective_summary_source(self) -> str:
        return self.og_description or self.description or ""


def serialize_articles(articles: list[Article]) -> list[dict[str, Any]]:
    return [article.model_dump(mode="json") for article in articles]


def parse_articles(payload: list[dict[str, Any]] | None) -> list[Article]:
    if not payload:
        return []
    return [Article.model_validate(item) for item in payload]
