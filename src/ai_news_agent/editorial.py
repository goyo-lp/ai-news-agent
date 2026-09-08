"""Investigate shortlisted stories and rank them with an editorial agent."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import httpx
import langsmith as ls
import structlog
from langchain.agents import create_agent
from langchain_core.tools import tool

from ai_news_agent.config import Settings
from ai_news_agent.observability import redact
from ai_news_agent.schemas import (
    Article,
    ArticleText,
    Evidence,
    EvidenceKind,
    Score,
    Source,
    StoryCluster,
)
from ai_news_agent.storage import SQLiteStore

_LOGGER = structlog.get_logger(__name__)

POLICY_VERSION = "editorial-policy-2026-09-07"
PROMPT_VERSION = "editorial-v1"
DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_MAX_SECONDS = 120.0
DEFAULT_JUDGE_TIMEOUT = 30.0
DEFAULT_JUDGE_MAX_TOKENS = 800
PRIMARY_TEXT_CHARS = 2000
SUPPORTING_EXCERPT_CHARS = 1000

_EDITORIAL_SYSTEM = (
    "You are an AI news editor ranking stories for AI builders. "
    "Score only what the provided evidence supports. "
    "Reply with exactly one JSON object and no other text."
)
_EDITORIAL_OUTPUT_SCHEMA = (
    '{"relevance": 0-5, "impact": 0-5, "novelty": 0-5, "evidence": 0-5, '
    '"timeliness": 0-5, "evidence_ids": [string], '
    '"uncertainty": 0.0-1.0, "rationale": string, '
    '"needs_more_evidence": true|false}'
)

EDITORIAL_SYSTEM_PROMPT = (
    "You are the AI News Agent editorial investigator. "
    "Use the available tools to read the representative article, inspect "
    "related coverage, fetch allowed primary evidence, and consult digest "
    "history. Then score relevance, impact, novelty, evidence, and timeliness "
    "0-5 with evidence references, uncertainty, and a concise inclusion "
    "rationale. Totals are computed in code, never by the model."
)


class EditorialError(RuntimeError):
    """Raised when the editorial judge cannot be reached."""


class EditorialOutputError(EditorialError):
    """Raised when the judge returns malformed or invalid scores."""


class EditorialConfigurationError(RuntimeError):
    """Raised when editorial credentials are missing or rejected."""


@dataclass(slots=True)
class EditorialVerdict:
    relevance: int
    impact: int
    novelty: int
    evidence: int
    timeliness: int
    evidence_ids: tuple[str, ...]
    uncertainty: float
    rationale: str
    needs_more_evidence: bool = False


@dataclass(slots=True)
class EditorialDecision:
    cluster_id: str
    verdict: str
    rationale: str
    uncertainty: float
    weighted_total: float = 0.0
    score_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    tool_calls: int = 0
    revisions: int = 0
    termination_reason: str = "complete"


@dataclass(slots=True)
class EditorialSummary:
    attempted: int = 0
    ranked: int = 0
    incomplete: int = 0
    tool_calls: int = 0
    judge_calls: int = 0
    revisions: int = 0
    ranking: tuple[str, ...] = ()
    decisions: tuple[EditorialDecision, ...] = ()
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    termination_reason: str = "complete"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class EditorialJudge(Protocol):
    """Score one story cluster from gathered evidence."""

    @property
    def identity(self) -> str:
        """Return a stable model identifier for tracing."""
        ...

    def score(
        self,
        title: str,
        article_text: str,
        supporting: str,
        history_note: str,
    ) -> EditorialVerdict:
        """Return per-criterion scores with evidence references."""
        ...


def editorial_prompt(
    title: str,
    article_text: str,
    supporting: str,
    history_note: str,
) -> str:
    """Build the versioned scoring prompt for one story cluster."""

    return (
        f"Headline: {title}\n"
        f"Digest history: {history_note}\n\n"
        f"PRIMARY ARTICLE (representative, to be summarized):\n"
        f"{article_text[:PRIMARY_TEXT_CHARS]}\n\n"
        f"SUPPORTING COVERAGE (investigation only, never summarize instead):\n"
        f"{supporting[: SUPPORTING_EXCERPT_CHARS * 2]}\n\n"
        "Score relevance, impact, novelty, evidence, timeliness 0-5 using the "
        "editorial rubric (5 high, 3 medium, 1 low, 0 absent). Cite evidence_ids "
        "for every factual claim, state uncertainty 0.0-1.0, give a concise "
        "inclusion rationale, and set needs_more_evidence when an uncertain "
        "claim could be resolved with the available tools. "
        f"Reply with exactly one JSON object like {_EDITORIAL_OUTPUT_SCHEMA}."
    )


def parse_editorial_verdict(raw: str) -> EditorialVerdict:
    """Parse strict model JSON output into a validated verdict."""

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise EditorialOutputError(f"editorial judge returned no JSON: {raw[:120]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise EditorialOutputError(
            f"editorial judge returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EditorialOutputError("editorial judge returned a non-object verdict")
    try:
        scores = {
            key: payload[key]
            for key in ("relevance", "impact", "novelty", "evidence", "timeliness")
        }
        evidence_ids = payload["evidence_ids"]
        uncertainty = payload["uncertainty"]
        rationale = payload["rationale"]
        needs_more = payload.get("needs_more_evidence", False)
    except KeyError as exc:
        raise EditorialOutputError(f"editorial verdict missing {exc}") from exc
    for key, value in scores.items():
        if not isinstance(value, int) or not 0 <= value <= 5:
            raise EditorialOutputError(f"editorial score {key} must be int 0-5")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or not all(isinstance(item, str) and item.strip() for item in evidence_ids)
    ):
        raise EditorialOutputError("editorial verdict needs non-empty evidence_ids")
    if (
        not isinstance(uncertainty, (int, float))
        or not 0.0 <= float(uncertainty) <= 1.0
    ):
        raise EditorialOutputError("editorial uncertainty must be 0.0-1.0")
    if not isinstance(rationale, str) or not rationale.strip():
        raise EditorialOutputError("editorial verdict needs a non-empty rationale")
    if not isinstance(needs_more, bool):
        raise EditorialOutputError("needs_more_evidence must be a boolean")
    return EditorialVerdict(
        relevance=scores["relevance"],
        impact=scores["impact"],
        novelty=scores["novelty"],
        evidence=scores["evidence"],
        timeliness=scores["timeliness"],
        evidence_ids=tuple(item.strip()[:200] for item in evidence_ids),
        uncertainty=float(uncertainty),
        rationale=rationale.strip()[:1000],
        needs_more_evidence=needs_more,
    )


class OpenRouterEditorialJudge:
    """Per-criterion editorial judge behind OpenRouter's OpenAI-compatible API."""

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

    def score(
        self, title: str, article_text: str, supporting: str, history_note: str
    ) -> EditorialVerdict:
        """Call the model once and parse its strict JSON scores."""

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "temperature": 0,
                    "max_tokens": self._max_tokens,
                    "messages": [
                        {"role": "system", "content": _EDITORIAL_SYSTEM},
                        {
                            "role": "user",
                            "content": editorial_prompt(
                                title, article_text, supporting, history_note
                            ),
                        },
                    ],
                },
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise EditorialError(
                f"editorial judge timeout after {self._timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise EditorialError(f"editorial judge transport error: {exc}") from exc
        if response.status_code in (401, 403):
            raise EditorialConfigurationError(
                f"editorial judge credentials rejected (HTTP {response.status_code})"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise EditorialError(
                f"editorial judge HTTP {response.status_code}, retryable"
            )
        if response.status_code >= 400:
            raise EditorialError(
                f"editorial judge HTTP {response.status_code}, not retryable"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise EditorialOutputError(
                f"editorial judge returned an unexpected envelope: {exc}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise EditorialOutputError("editorial judge returned empty content")
        return parse_editorial_verdict(content)


def build_editorial_judge(
    settings: Settings,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> OpenRouterEditorialJudge:
    """Build the configured judge, failing fast without credentials."""

    if not settings.judge_ready or settings.openrouter_api_key is None:
        raise EditorialConfigurationError(
            "set OPENROUTER_API_KEY before model-judged editorial ranking"
        )
    return OpenRouterEditorialJudge(
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def read_article_text(
    cluster: StoryCluster,
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
) -> str:
    """Return the representative article text (primary evidence)."""

    article = articles_by_id.get(cluster.representative_article_id)
    if article is None:
        return ""
    body = store.get(ArticleText, f"{article.id}-text")
    if body is None or not body.text.strip():
        return article.title
    return body.text.strip()[:PRIMARY_TEXT_CHARS]


def inspect_related_coverage(
    cluster: StoryCluster,
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
) -> str:
    """Summarize supporting articles in the same cluster (investigation only)."""

    lines: list[str] = []
    for article_id in cluster.article_ids:
        if article_id == cluster.representative_article_id:
            continue
        article = articles_by_id.get(article_id)
        if article is None:
            continue
        source = store.get(Source, article.source_id)
        name = source.name if source is not None else article.source_id
        lines.append(f"- {article.title} ({name}, {article.id})")
    if not lines:
        return "no additional cluster coverage"
    return "related coverage:\n" + "\n".join(lines[:10])


def fetch_primary_evidence(cluster: StoryCluster, store: SQLiteStore) -> str:
    """Collect allowed primary evidence excerpts for a cluster."""

    allowed = {
        EvidenceKind.ARTICLE_TEXT,
        EvidenceKind.PRIMARY_DOCUMENT,
        EvidenceKind.SUPPORTING_REPORT,
    }
    parts: list[str] = []
    for article_id in cluster.article_ids:
        for suffix in ("-article-text", "-feed"):
            record = store.get(Evidence, f"{article_id}{suffix}")
            if record is None or record.kind not in allowed:
                continue
            if "-investigation-" in record.id:
                continue
            parts.append(f"[{record.id}] {record.excerpt.strip()[:500]}")
            if len(parts) >= 6:
                break
        if len(parts) >= 6:
            break
    if not parts:
        return "no stored primary evidence excerpts"
    return "primary evidence:\n" + "\n".join(parts)


def check_digest_history(
    cluster: StoryCluster,
    published_article_ids: frozenset[str],
) -> str:
    """Note whether a cluster overlaps already-published digest history."""

    if not published_article_ids:
        return "no digest history provided; treat as new"
    overlap = [item for item in cluster.article_ids if item in published_article_ids]
    if overlap:
        return f"overlaps published history: {', '.join(sorted(overlap))}"
    return "new relative to digest history"


def make_editorial_tools(
    store: SQLiteStore,
    articles_by_id: dict[str, Article],
    published_article_ids: frozenset[str] = frozenset(),
) -> Sequence[Callable]:
    """Build LangChain tools bound to one editorial evidence store."""

    @tool
    def read_article(cluster_id: str) -> str:
        """Read the representative article text for a story cluster.

        Args:
            cluster_id: StoryCluster ID whose representative text is needed.
        """
        for record in store.iter_records(StoryCluster):
            if record.id == cluster_id:
                return read_article_text(record, articles_by_id, store)
        return f"unknown cluster: {cluster_id}"

    @tool
    def inspect_coverage(cluster_id: str) -> str:
        """Inspect related coverage for a story cluster.

        Args:
            cluster_id: StoryCluster ID whose supporting articles to list.
        """
        for record in store.iter_records(StoryCluster):
            if record.id == cluster_id:
                return inspect_related_coverage(record, articles_by_id, store)
        return f"unknown cluster: {cluster_id}"

    @tool
    def fetch_evidence(cluster_id: str) -> str:
        """Fetch allowed primary evidence excerpts for a story cluster.

        Args:
            cluster_id: StoryCluster ID whose evidence excerpts to fetch.
        """
        for record in store.iter_records(StoryCluster):
            if record.id == cluster_id:
                return fetch_primary_evidence(record, store)
        return f"unknown cluster: {cluster_id}"

    @tool
    def consult_history(cluster_id: str) -> str:
        """Consult digest history for overlap with a story cluster.

        Args:
            cluster_id: StoryCluster ID to compare against published history.
        """
        for record in store.iter_records(StoryCluster):
            if record.id == cluster_id:
                return check_digest_history(record, published_article_ids)
        return f"unknown cluster: {cluster_id}"

    return [read_article, inspect_coverage, fetch_evidence, consult_history]


def build_editorial_agent(
    model: str,
    tools: Sequence[Callable] | None = None,
    *,
    store: SQLiteStore | None = None,
    articles_by_id: dict[str, Article] | None = None,
    published_article_ids: frozenset[str] = frozenset(),
):  # type: ignore[no-untyped-def]
    """Build a LangChain agent wired to the editorial evidence tools."""

    resolved = list(tools) if tools is not None else []
    if not resolved and store is not None and articles_by_id is not None:
        resolved = list(
            make_editorial_tools(store, articles_by_id, published_article_ids)
        )
    return create_agent(
        model=model,
        tools=resolved,
        system_prompt=EDITORIAL_SYSTEM_PROMPT,
    )


Sleeper = Callable[[float], None]
Clock = Callable[[], float]


def _supporting_context(
    cluster: StoryCluster,
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
) -> str:
    related = inspect_related_coverage(cluster, articles_by_id, store)
    evidence = fetch_primary_evidence(cluster, store)
    return f"{related}\n{evidence}"


@ls.traceable(
    name="editorial-ranking",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def rank_clusters(
    clusters: tuple[StoryCluster, ...],
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
    *,
    now: datetime | None = None,
    judge: EditorialJudge | None = None,
    digest_run_id: str = "manual-editorial",
    published_article_ids: frozenset[str] = frozenset(),
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sleep: Sleeper = time.sleep,
    clock: Clock = time.monotonic,
) -> EditorialSummary:
    """Investigate each cluster with bounded tools and rank the results.

    Evidence tools run first (representative text, related coverage, primary
    excerpts, history). The judge returns per-criterion scores, evidence
    references, uncertainty, and a rationale; the weighted total is computed in
    code and persisted as a Score. Uncertain verdicts get one targeted
    reinvestigation with supporting context kept separate from the primary
    article. Budget exhaustion marks remaining clusters incomplete instead of
    looping, and every decision records model identity, prompt version, calls,
    and termination reason for LangSmith.
    """

    del sleep, digest_run_id
    started = now or datetime.now(UTC)
    start_mark = clock()
    active_judge = judge or build_editorial_judge(Settings())
    decisions: list[EditorialDecision] = []
    tool_calls = 0
    judge_calls = 0
    revisions = 0
    exhausted = False

    for cluster in sorted(clusters, key=lambda item: item.id):
        elapsed = max(0.0, clock() - start_mark)
        if (
            exhausted
            or tool_calls >= max_tool_calls
            or judge_calls >= max_iterations
            or elapsed >= max_seconds
        ):
            exhausted = True
            decisions.append(
                EditorialDecision(
                    cluster_id=cluster.id,
                    verdict="incomplete",
                    rationale="budget exhausted before investigation",
                    uncertainty=1.0,
                    tool_calls=0,
                    termination_reason="budget-exhausted",
                )
            )
            continue
        article = articles_by_id.get(cluster.representative_article_id)
        if article is None:
            decisions.append(
                EditorialDecision(
                    cluster_id=cluster.id,
                    verdict="incomplete",
                    rationale="representative article missing from store",
                    uncertainty=1.0,
                    termination_reason="missing-representative",
                )
            )
            continue

        primary = read_article_text(cluster, articles_by_id, store)
        supporting = _supporting_context(cluster, articles_by_id, store)
        history = check_digest_history(cluster, published_article_ids)
        cluster_tool_calls = 3
        tool_calls += 3

        try:
            verdict = active_judge.score(article.title, primary, supporting, history)
            judge_calls += 1
        except EditorialConfigurationError:
            raise
        except EditorialOutputError as exc:
            decisions.append(
                EditorialDecision(
                    cluster_id=cluster.id,
                    verdict="incomplete",
                    rationale=f"judge output unusable: {exc}",
                    uncertainty=1.0,
                    tool_calls=cluster_tool_calls,
                    termination_reason="malformed-output",
                )
            )
            continue
        except EditorialError as exc:
            decisions.append(
                EditorialDecision(
                    cluster_id=cluster.id,
                    verdict="incomplete",
                    rationale=f"judge unavailable: {exc}",
                    uncertainty=1.0,
                    tool_calls=cluster_tool_calls,
                    termination_reason="judge-unavailable",
                )
            )
            continue

        cluster_revisions = 0
        elapsed = max(0.0, clock() - start_mark)
        if (
            verdict.needs_more_evidence
            and judge_calls < max_iterations
            and tool_calls < max_tool_calls
            and elapsed < max_seconds
        ):
            extra = fetch_primary_evidence(cluster, store)
            tool_calls += 1
            cluster_tool_calls += 1
            enriched = f"{supporting}\nreinvestigation:\n{extra}"
            try:
                revised = active_judge.score(article.title, primary, enriched, history)
                judge_calls += 1
            except (EditorialOutputError, EditorialError):
                pass
            else:
                verdict = revised
                cluster_revisions = 1
                revisions += 1

        record = Score(
            id=f"{cluster.id}-editorial",
            story_cluster_id=cluster.id,
            relevance=verdict.relevance,
            impact=verdict.impact,
            novelty=verdict.novelty,
            evidence=verdict.evidence,
            timeliness=verdict.timeliness,
            evidence_ids=verdict.evidence_ids,
            rationale=verdict.rationale,
        )
        store.save(record)
        decisions.append(
            EditorialDecision(
                cluster_id=cluster.id,
                verdict="ranked",
                rationale=verdict.rationale,
                uncertainty=verdict.uncertainty,
                weighted_total=record.weighted_total,
                score_id=record.id,
                evidence_ids=verdict.evidence_ids,
                tool_calls=cluster_tool_calls,
                revisions=cluster_revisions,
                termination_reason="complete",
            )
        )

    ranked = sorted(
        (item for item in decisions if item.verdict == "ranked"),
        key=lambda item: (-item.weighted_total, item.cluster_id),
    )
    summary = EditorialSummary(
        attempted=len(clusters),
        ranked=len(ranked),
        incomplete=sum(1 for item in decisions if item.verdict == "incomplete"),
        tool_calls=tool_calls,
        judge_calls=judge_calls,
        revisions=revisions,
        ranking=tuple(item.cluster_id for item in ranked),
        decisions=tuple(decisions),
        model=active_judge.identity,
        prompt_version=PROMPT_VERSION,
        termination_reason="budget-exhausted" if exhausted else "complete",
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    _LOGGER.info(
        "editorial_ranking_complete",
        attempted=summary.attempted,
        ranked=summary.ranked,
        incomplete=summary.incomplete,
        revisions=summary.revisions,
        model=summary.model,
        prompt_version=summary.prompt_version,
        termination_reason=summary.termination_reason,
    )
    return summary


__all__ = [
    "DEFAULT_JUDGE_MAX_TOKENS",
    "DEFAULT_JUDGE_TIMEOUT",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_TOOL_CALLS",
    "EDITORIAL_SYSTEM_PROMPT",
    "POLICY_VERSION",
    "PROMPT_VERSION",
    "EditorialConfigurationError",
    "EditorialDecision",
    "EditorialError",
    "EditorialJudge",
    "EditorialOutputError",
    "EditorialSummary",
    "EditorialVerdict",
    "OpenRouterEditorialJudge",
    "build_editorial_agent",
    "build_editorial_judge",
    "check_digest_history",
    "editorial_prompt",
    "fetch_primary_evidence",
    "inspect_related_coverage",
    "make_editorial_tools",
    "parse_editorial_verdict",
    "rank_clusters",
    "read_article_text",
]
