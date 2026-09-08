"""Generate grounded three-sentence summaries and render digest previews."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import httpx
import langsmith as ls
import structlog

from ai_news_agent.config import Settings
from ai_news_agent.observability import redact
from ai_news_agent.schemas import (
    Article,
    ArticleText,
    Evidence,
    EvidenceKind,
    Source,
    StoryCluster,
    Summary,
)
from ai_news_agent.storage import SQLiteStore

_LOGGER = structlog.get_logger(__name__)

POLICY_VERSION = "editorial-policy-2026-09-07"
PROMPT_VERSION = "summaries-v1"
DEFAULT_JUDGE_TIMEOUT = 30.0
DEFAULT_JUDGE_MAX_TOKENS = 600
MIN_EVIDENCE_CHARS = 400
MAX_EVIDENCE_CHARS = 6000

_SUMMARY_SYSTEM = (
    "You write grounded three-sentence news summaries for AI builders. "
    "Use only the provided article evidence, attribute claims, and never "
    "invent numbers, dates, or availability. "
    "Reply with exactly one JSON object and no other text."
)
_SUMMARY_OUTPUT_SCHEMA = (
    '{"sentence1": string, "sentence2": string, "sentence3": string, '
    '"evidence_by_sentence": [[string], [string], [string]]}'
)

_ABBREVIATIONS = (
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Inc.",
    "Ltd.",
    "Jr.",
    "Sr.",
    "St.",
    "U.S.",
    "U.K.",
    "E.U.",
    "e.g.",
    "i.e.",
    "vs.",
    "etc.",
    "Fig.",
    "No.",
)
_DECIMAL = re.compile(r"(?P<before>\d)\.(?P<after>\d)")


class SummaryError(RuntimeError):
    """Raised when the summary writer cannot be reached."""


class SummaryOutputError(SummaryError):
    """Raised when the writer returns malformed or invalid summaries."""


class SummaryConfigurationError(RuntimeError):
    """Raised when summary credentials are missing or rejected."""


@dataclass(slots=True)
class SummaryDraft:
    sentences: tuple[str, str, str]
    evidence_by_sentence: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


@dataclass(slots=True)
class SummaryDecision:
    cluster_id: str
    status: str
    reason: str
    summary_id: str | None = None


@dataclass(slots=True)
class SummaryResult:
    attempted: int = 0
    generated: int = 0
    insufficient: int = 0
    summaries: tuple[Summary, ...] = ()
    decisions: tuple[SummaryDecision, ...] = ()
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class DigestItem:
    cluster_id: str
    title: str
    url: str
    source_name: str
    published_at: datetime
    sentences: tuple[str, str, str]
    evidence_ids: tuple[str, ...]


class SummaryWriter(Protocol):
    """Write one three-sentence summary from article evidence."""

    @property
    def identity(self) -> str:
        """Return a stable model identifier for tracing."""
        ...

    def write(
        self, title: str, article_text: str, evidence_excerpts: str
    ) -> SummaryDraft:
        """Return three grounded sentences with per-sentence evidence."""
        ...


def summary_prompt(title: str, article_text: str, evidence_excerpts: str) -> str:
    """Build the versioned summary prompt for one representative article."""

    return (
        f"Headline: {title}\n\n"
        "ARTICLE TEXT (only summarizable source):\n"
        f"{article_text[:MAX_EVIDENCE_CHARS]}\n\n"
        "SUPPORTING EXCERPTS (context only):\n"
        f"{evidence_excerpts[:MAX_EVIDENCE_CHARS]}\n\n"
        "Write exactly three sentences about the linked representative article: "
        "(1) what happened and who did it; (2) the most decision-relevant "
        "supporting detail, number, scope, availability, or limitation; "
        "(3) why it matters for AI builders or a material uncertainty. "
        "Attribute vendor, government, and advocacy claims (for example, "
        "'the company said'). Frame inferred significance as analysis that "
        "follows directly from cited facts. Use plain prose with no bullets, "
        "no semicolon chains hiding extra sentences, and no promotional "
        "language. Map every sentence to evidence IDs from the provided text. "
        f"Reply with exactly one JSON object like {_SUMMARY_OUTPUT_SCHEMA}."
    )


def _protect(text: str) -> str:
    protected = text
    for token in _ABBREVIATIONS:
        protected = protected.replace(token, token.replace(".", "<DOT>"))
    return _DECIMAL.sub(
        lambda match: f"{match.group('before')}<DOT>{match.group('after')}",
        protected,
    )


def validate_sentence(text: str) -> str:
    """Validate one prose sentence, tolerating abbreviations and decimals."""

    cleaned = text.strip()
    if not cleaned:
        raise SummaryOutputError("summary sentence must not be empty")
    if "\n" in cleaned or "\r" in cleaned:
        raise SummaryOutputError("summary sentences must not contain line breaks")
    if cleaned.startswith(("-", "*", "•")):
        raise SummaryOutputError("summary sentences must not use bullets")
    if ";" in cleaned:
        raise SummaryOutputError("summary sentences must not hide clauses with ';'")
    protected = _protect(cleaned)
    terminators = re.findall(r"[.!?…]+", protected)
    if len(terminators) != 1 or not re.search(r"[.!?…]+[\"')\]]?\s*$", protected):
        raise SummaryOutputError("each summary field must be exactly one sentence")
    if len(cleaned.split()) < 5:
        raise SummaryOutputError("summary sentence looks like a fragment")
    return cleaned


def parse_summary_draft(raw: str) -> SummaryDraft:
    """Parse strict model JSON output into a validated three-sentence draft."""

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise SummaryOutputError(f"summary writer returned no JSON: {raw[:120]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SummaryOutputError(
            f"summary writer returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SummaryOutputError("summary writer returned a non-object draft")
    try:
        sentences = (payload["sentence1"], payload["sentence2"], payload["sentence3"])
        evidence = payload["evidence_by_sentence"]
    except KeyError as exc:
        raise SummaryOutputError(f"summary draft missing {exc}") from exc
    if not all(isinstance(item, str) for item in sentences):
        raise SummaryOutputError("summary sentences must be strings")
    validated = (
        validate_sentence(sentences[0]),
        validate_sentence(sentences[1]),
        validate_sentence(sentences[2]),
    )
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise SummaryOutputError("evidence_by_sentence must list three sentences")
    mapped: list[tuple[str, ...]] = []
    for group in evidence:
        if not isinstance(group, list) or not group:
            raise SummaryOutputError("each sentence needs at least one evidence ID")
        if not all(isinstance(item, str) and item.strip() for item in group):
            raise SummaryOutputError("evidence IDs must be non-empty strings")
        mapped.append(tuple(item.strip()[:200] for item in group))
    return SummaryDraft(
        sentences=validated,
        evidence_by_sentence=(mapped[0], mapped[1], mapped[2]),
    )


class OpenRouterSummaryWriter:
    """Grounded summary writer behind OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = DEFAULT_JUDGE_TIMEOUT,
        max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_tokens = max_tokens

    @property
    def identity(self) -> str:
        return self._model

    def write(
        self, title: str, article_text: str, evidence_excerpts: str
    ) -> SummaryDraft:
        """Call the model once and parse its strict JSON draft."""

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "temperature": 0,
                    "max_tokens": self._max_tokens,
                    "messages": [
                        {"role": "system", "content": _SUMMARY_SYSTEM},
                        {
                            "role": "user",
                            "content": summary_prompt(
                                title, article_text, evidence_excerpts
                            ),
                        },
                    ],
                },
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise SummaryError(
                f"summary writer timeout after {self._timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise SummaryError(f"summary writer transport error: {exc}") from exc
        if response.status_code in (401, 403):
            raise SummaryConfigurationError(
                f"summary writer credentials rejected (HTTP {response.status_code})"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise SummaryError(f"summary writer HTTP {response.status_code}, retryable")
        if response.status_code >= 400:
            raise SummaryError(
                f"summary writer HTTP {response.status_code}, not retryable"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise SummaryOutputError(
                f"summary writer returned an unexpected envelope: {exc}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise SummaryOutputError("summary writer returned empty content")
        return parse_summary_draft(content)


def build_summary_writer(
    settings: Settings,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> OpenRouterSummaryWriter:
    """Build the configured writer, failing fast without credentials."""

    if not settings.judge_ready or settings.openrouter_api_key is None:
        raise SummaryConfigurationError(
            "set OPENROUTER_API_KEY before model-written summaries"
        )
    return OpenRouterSummaryWriter(
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def article_evidence(
    cluster: StoryCluster,
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Return primary text plus excerpts, or None when evidence is thin."""

    article = articles_by_id.get(cluster.representative_article_id)
    if article is None:
        return None
    body = store.get(ArticleText, f"{article.id}-text")
    text = body.text.strip() if body is not None and body.text.strip() else ""
    if len(text) < MIN_EVIDENCE_CHARS:
        return None
    allowed = {
        EvidenceKind.ARTICLE_TEXT,
        EvidenceKind.PRIMARY_DOCUMENT,
        EvidenceKind.SUPPORTING_REPORT,
    }
    parts: list[str] = []
    ids: list[str] = [f"{article.id}-text"]
    for suffix in ("-article-text", "-feed"):
        record: Evidence | None = store.get(Evidence, f"{article.id}{suffix}")
        if record is None or record.kind not in allowed:
            continue
        parts.append(f"[{record.id}] {record.excerpt.strip()[:800]}")
    return text[:MAX_EVIDENCE_CHARS], "\n".join(parts), tuple(ids)


@ls.traceable(
    name="generate-summaries",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def generate_summaries(
    clusters: tuple[StoryCluster, ...],
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
    *,
    now: datetime | None = None,
    writer: SummaryWriter | None = None,
    digest_run_id: str = "manual-summaries",
) -> SummaryResult:
    """Generate one grounded draft summary per cluster with full evidence.

    Clusters without retrievable full text never receive a headline-only
    summary; they become explicit insufficient items instead.
    """

    del digest_run_id
    started = now or datetime.now(UTC)
    active = writer or build_summary_writer(Settings())
    summaries: list[Summary] = []
    decisions: list[SummaryDecision] = []

    for cluster in sorted(clusters, key=lambda item: item.id):
        article = articles_by_id.get(cluster.representative_article_id)
        if article is None:
            decisions.append(
                SummaryDecision(
                    cluster_id=cluster.id,
                    status="insufficient",
                    reason="representative article missing from store",
                )
            )
            continue
        evidence = article_evidence(cluster, articles_by_id, store)
        if evidence is None:
            decisions.append(
                SummaryDecision(
                    cluster_id=cluster.id,
                    status="insufficient",
                    reason="insufficient article text for a full-article summary",
                )
            )
            continue
        text, excerpts, _ids = evidence
        try:
            draft = active.write(article.title, text, excerpts)
        except SummaryConfigurationError:
            raise
        except SummaryOutputError as exc:
            decisions.append(
                SummaryDecision(
                    cluster_id=cluster.id,
                    status="insufficient",
                    reason=f"writer output unusable: {exc}",
                )
            )
            continue
        except SummaryError as exc:
            decisions.append(
                SummaryDecision(
                    cluster_id=cluster.id,
                    status="insufficient",
                    reason=f"writer unavailable: {exc}",
                )
            )
            continue
        record = Summary(
            id=f"{cluster.id}-summary",
            story_cluster_id=cluster.id,
            representative_article_id=article.id,
            sentences=draft.sentences,
            evidence_by_sentence=draft.evidence_by_sentence,
        )
        store.save(record)
        summaries.append(record)
        decisions.append(
            SummaryDecision(
                cluster_id=cluster.id,
                status="generated",
                reason="draft summary with sentence-level evidence",
                summary_id=record.id,
            )
        )

    result = SummaryResult(
        attempted=len(clusters),
        generated=len(summaries),
        insufficient=sum(1 for item in decisions if item.status == "insufficient"),
        summaries=tuple(summaries),
        decisions=tuple(decisions),
        model=active.identity,
        prompt_version=PROMPT_VERSION,
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    _LOGGER.info(
        "summaries_complete",
        attempted=result.attempted,
        generated=result.generated,
        insufficient=result.insufficient,
        model=result.model,
        prompt_version=result.prompt_version,
    )
    return result


def build_digest_items(
    selected_ids: Sequence[str],
    clusters_by_id: dict[str, StoryCluster],
    articles_by_id: dict[str, Article],
    summaries_by_cluster: dict[str, Summary],
    store: SQLiteStore,
) -> tuple[DigestItem, ...]:
    """Assemble one ordered structured digest shared by all renderers."""

    items: list[DigestItem] = []
    for cluster_id in selected_ids:
        cluster = clusters_by_id.get(cluster_id)
        summary = summaries_by_cluster.get(cluster_id)
        article = (
            articles_by_id.get(cluster.representative_article_id)
            if cluster is not None
            else None
        )
        if cluster is None or summary is None or article is None:
            continue
        source = store.get(Source, article.source_id)
        name = source.name if source is not None else article.source_id
        evidence = tuple(
            evidence_id
            for group in summary.evidence_by_sentence
            for evidence_id in group
        )
        items.append(
            DigestItem(
                cluster_id=cluster_id,
                title=article.title,
                url=str(article.url),
                source_name=name,
                published_at=article.published_at,
                sentences=summary.sentences,
                evidence_ids=evidence,
            )
        )
    return tuple(items)


def render_email_html(
    items: Sequence[DigestItem],
    *,
    digest_label: str = "Daily AI digest",
    shortfall_note: str | None = None,
) -> str:
    """Render an email-friendly HTML preview from the structured digest."""

    parts = [
        "<!doctype html><html><body>",
        '<div style="max-width:600px;margin:0 auto;font-family:sans-serif">',
        f"<h1>{html.escape(digest_label)}</h1>",
        "<p><strong>Draft &mdash; unverified.</strong> Review before any delivery.</p>",
    ]
    if shortfall_note:
        parts.append(f"<p>Coverage note: {html.escape(shortfall_note)}</p>")
    for index, item in enumerate(items, start=1):
        parts.append(
            f'<h2>{index}. <a href="{html.escape(item.url)}">'
            f"{html.escape(item.title)}</a></h2>"
        )
        parts.append(
            f"<p>{html.escape(item.source_name)} &middot; "
            f"{html.escape(item.published_at.isoformat())}</p>"
        )
        for sentence in item.sentences:
            parts.append(f"<p>{html.escape(sentence)}</p>")
        parts.append(f"<p>Evidence: {html.escape(', '.join(item.evidence_ids))}</p>")
    parts.append("</div></body></html>")
    return "".join(parts)


def render_email_text(
    items: Sequence[DigestItem],
    *,
    digest_label: str = "Daily AI digest",
    shortfall_note: str | None = None,
) -> str:
    """Render a plain-text email preview from the structured digest."""

    lines = [digest_label, "Draft — unverified. Review before any delivery.", ""]
    if shortfall_note:
        lines.append(f"Coverage note: {shortfall_note}")
        lines.append("")
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item.title}")
        lines.append(f"{item.url}")
        lines.append(f"{item.source_name} · {item.published_at.isoformat()}")
        lines.extend(item.sentences)
        lines.append(f"Evidence: {', '.join(item.evidence_ids)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_archive_html(
    items: Sequence[DigestItem],
    *,
    digest_label: str = "Daily AI digest",
    shortfall_note: str | None = None,
) -> str:
    """Render a dated HTML archive page from the structured digest."""

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(digest_label)}</title></head><body>",
        "<main>",
        f"<h1>{html.escape(digest_label)}</h1>",
        "<p><strong>Draft &mdash; unverified.</strong> Local preview only.</p>",
    ]
    if shortfall_note:
        parts.append(f"<p>Coverage note: {html.escape(shortfall_note)}</p>")
    parts.append("<ol>")
    for item in items:
        parts.append(
            f'<li><a href="{html.escape(item.url)}">{html.escape(item.title)}</a> '
            f"({html.escape(item.source_name)}, "
            f"{html.escape(item.published_at.isoformat())})"
        )
        parts.append("<ol>")
        for sentence in item.sentences:
            parts.append(f"<li>{html.escape(sentence)}</li>")
        parts.append("</ol></li>")
    parts.append("</ol></main></body></html>")
    return "".join(parts)


__all__ = [
    "DEFAULT_JUDGE_MAX_TOKENS",
    "DEFAULT_JUDGE_TIMEOUT",
    "MAX_EVIDENCE_CHARS",
    "MIN_EVIDENCE_CHARS",
    "POLICY_VERSION",
    "PROMPT_VERSION",
    "DigestItem",
    "OpenRouterSummaryWriter",
    "SummaryConfigurationError",
    "SummaryDecision",
    "SummaryDraft",
    "SummaryError",
    "SummaryOutputError",
    "SummaryResult",
    "SummaryWriter",
    "article_evidence",
    "build_digest_items",
    "build_summary_writer",
    "generate_summaries",
    "parse_summary_draft",
    "render_archive_html",
    "render_email_html",
    "render_email_text",
    "summary_prompt",
    "validate_sentence",
]
