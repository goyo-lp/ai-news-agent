"""Shortlist story clusters with date rules and a low-cost relevance judge."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

import httpx
import langsmith as ls
import structlog

from ai_news_agent.clustering import stable_cluster_id
from ai_news_agent.config import Settings
from ai_news_agent.extraction import retrieve_articles
from ai_news_agent.observability import redact
from ai_news_agent.schemas import Article, ArticleText, Evidence, Source, StoryCluster
from ai_news_agent.storage import SQLiteStore

if TYPE_CHECKING:
    from ai_news_agent.extraction import Fetcher

_LOGGER = structlog.get_logger(__name__)

POLICY_VERSION = "editorial-policy-2026-09-07"
PROMPT_VERSION = "screening-v1"
FRESH_HOURS = 24.0
MAX_AGE_HOURS = 36.0
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_JUDGE_TIMEOUT = 30.0
DEFAULT_JUDGE_MAX_TOKENS = 300

_JUDGE_SYSTEM = (
    "You screen AI news stories for a technical audience of AI builders. "
    "Reply with exactly one JSON object and no other text."
)
_JUDGE_OUTPUT_SCHEMA = (
    '{"relevant": true|false, "borderline": true|false, "reason": string}'
)


class JudgeError(RuntimeError):
    """Raised when the relevance judge cannot be reached."""


class JudgeOutputError(JudgeError):
    """Raised when the judge returns malformed or invalid output."""


class JudgeConfigurationError(RuntimeError):
    """Raised when judge credentials are missing or rejected."""


@dataclass(slots=True)
class RelevanceVerdict:
    relevant: bool
    borderline: bool
    reason: str


@dataclass(slots=True)
class ScreeningDecision:
    cluster_id: str
    verdict: str
    reason: str
    late_arrival: bool = False
    cached: bool = False


@dataclass(slots=True)
class ScreeningSummary:
    attempted: int = 0
    shortlisted: int = 0
    borderline: int = 0
    rejected: int = 0
    judge_calls: int = 0
    judge_cached: int = 0
    judge_errors: int = 0
    merged: int = 0
    shortlist: tuple[StoryCluster, ...] = ()
    decisions: tuple[ScreeningDecision, ...] = ()
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class RelevanceJudge(Protocol):
    """Assess whether one story matters to AI builders."""

    @property
    def identity(self) -> str:
        """Return a stable model identifier for cache keys."""
        ...

    def judge(self, title: str, excerpt: str, source_name: str) -> RelevanceVerdict:
        """Return a relevance verdict for one story."""
        ...


def judge_prompt(title: str, excerpt: str, source_name: str) -> str:
    """Build the versioned relevance prompt for one story."""

    return (
        f"Story headline: {title}\n"
        f"Source: {source_name}\n"
        f"Excerpt: {excerpt}\n\n"
        "Is this directly about artificial intelligence with a concrete new "
        "development (release, result, incident, policy step, or market change "
        "affecting builders)? Exclude non-AI stories, AI-as-marketing-label, "
        "evergreen explainers, tutorials, rumors without evidence, and funding "
        "news with no effect beyond the transaction. Mark genuinely uncertain "
        "cases borderline rather than guessing. Reply with exactly one JSON "
        f"object like {_JUDGE_OUTPUT_SCHEMA} and no other text."
    )


def parse_verdict(raw: str) -> RelevanceVerdict:
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
        raise JudgeOutputError(f"judge returned no JSON object: {raw[:120]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeOutputError(f"judge returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeOutputError("judge returned a non-object verdict")
    try:
        relevant = payload["relevant"]
        borderline = payload["borderline"]
        reason = payload["reason"]
    except KeyError as exc:
        raise JudgeOutputError(f"judge verdict missing {exc}") from exc
    if not isinstance(relevant, bool) or not isinstance(borderline, bool):
        raise JudgeOutputError("judge verdict flags must be booleans")
    if not isinstance(reason, str) or not reason.strip():
        raise JudgeOutputError("judge verdict needs a non-empty reason")
    return RelevanceVerdict(
        relevant=relevant, borderline=borderline, reason=reason.strip()[:500]
    )


class OpenRouterJudge:
    """Low-cost relevance judge behind OpenRouter's OpenAI-compatible API."""

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

    def judge(self, title: str, excerpt: str, source_name: str) -> RelevanceVerdict:
        """Call the model once and parse its strict JSON verdict."""

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "temperature": 0,
                    "max_tokens": self._max_tokens,
                    "messages": [
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {
                            "role": "user",
                            "content": judge_prompt(title, excerpt, source_name),
                        },
                    ],
                },
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise JudgeError(f"judge timeout after {self._timeout}s") from exc
        except httpx.HTTPError as exc:
            raise JudgeError(f"judge transport error: {exc}") from exc
        if response.status_code in (401, 403):
            raise JudgeConfigurationError(
                f"judge credentials rejected (HTTP {response.status_code})"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise JudgeError(f"judge HTTP {response.status_code}, retryable")
        if response.status_code >= 400:
            raise JudgeError(f"judge HTTP {response.status_code}, not retryable")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise JudgeOutputError(
                f"judge returned an unexpected envelope: {exc}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise JudgeOutputError("judge returned empty content")
        return parse_verdict(content)


def build_judge(
    settings: Settings, timeout: float = DEFAULT_JUDGE_TIMEOUT
) -> OpenRouterJudge:
    """Build the configured judge, failing fast without credentials."""

    if not settings.judge_ready or settings.openrouter_api_key is None:
        raise JudgeConfigurationError(
            "set OPENROUTER_API_KEY before model-judged screening"
        )
    return OpenRouterJudge(
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        timeout=timeout,
    )


def cache_key(
    title: str,
    excerpt: str,
    *,
    model: str,
    policy_version: str = POLICY_VERSION,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """Key cached verdicts by content, policy, prompt, and model version."""

    digest = hashlib.sha256(
        "\n".join([prompt_version, policy_version, model, title, excerpt]).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"screening-{digest[:32]}"


def cache_lookup(store: SQLiteStore, key: str) -> RelevanceVerdict | None:
    """Return a cached verdict, or None on a miss or corrupt entry."""

    try:
        row = store.connection.execute(
            "SELECT verdict_json FROM screening_cache WHERE cache_key = ?", (key,)
        ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    try:
        payload = json.loads(row["verdict_json"])
        return RelevanceVerdict(
            relevant=bool(payload["relevant"]),
            borderline=bool(payload["borderline"]),
            reason=str(payload["reason"]),
        )
    except (ValueError, KeyError, TypeError):
        return None


def cache_store(store: SQLiteStore, key: str, verdict: RelevanceVerdict) -> None:
    """Persist one verdict for future runs with identical inputs."""

    store.connection.execute(
        """
        INSERT INTO screening_cache (cache_key, verdict_json)
        VALUES (?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            verdict_json = excluded.verdict_json,
            created_at = CURRENT_TIMESTAMP
        """,
        (
            key,
            json.dumps(
                {
                    "relevant": verdict.relevant,
                    "borderline": verdict.borderline,
                    "reason": verdict.reason,
                }
            ),
        ),
    )
    store.connection.commit()


Sleeper = Callable[[float], None]


def judge_with_retries(
    judge: RelevanceJudge,
    title: str,
    excerpt: str,
    source_name: str,
    *,
    max_retries: int = 1,
    sleep: Sleeper = time.sleep,
) -> RelevanceVerdict:
    """Retry retryable judge errors once; configuration errors fail fast."""

    last_error: JudgeError | None = None
    for attempt in range(max_retries + 1):
        try:
            return judge.judge(title, excerpt, source_name)
        except JudgeConfigurationError:
            raise
        except JudgeOutputError:
            raise
        except JudgeError as exc:
            last_error = exc
            message = str(exc)
            if "not retryable" in message or attempt >= max_retries:
                raise
            sleep(float(2**attempt))
    raise last_error or JudgeError("judge failed without a verdict")


def _source_name(article: Article, store: SQLiteStore) -> str:
    """Return the publisher name for judge context, falling back to the ID."""

    source = store.get(Source, article.source_id)
    return source.name if source is not None else article.source_id


def _cluster_excerpt(
    cluster: StoryCluster, articles_by_id: dict[str, Article], store: SQLiteStore
) -> str:
    """Return feed-excerpt context for the representative article."""

    article = articles_by_id.get(cluster.representative_article_id)
    if article is None:
        return ""
    evidence = store.get(Evidence, f"{article.id}-feed")
    if evidence is not None and evidence.excerpt.strip():
        return evidence.excerpt.strip()[:1000]
    return article.title


def _merge_identical_texts(
    kept: list[StoryCluster],
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
    now: datetime,
) -> tuple[list[StoryCluster], int]:
    """Merge kept clusters whose representatives share identical full text."""

    groups: dict[str, list[StoryCluster]] = {}
    for cluster in kept:
        article = articles_by_id.get(cluster.representative_article_id)
        body = (
            store.get(ArticleText, f"{article.id}-text")
            if article is not None
            else None
        )
        key = body.content_hash if body is not None else cluster.id
        groups.setdefault(key, []).append(cluster)
    merged: list[StoryCluster] = []
    absorbed = 0
    for _key, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        member_ids = tuple(
            sorted(
                {article_id for cluster in group for article_id in cluster.article_ids}
            )
        )
        best = min(
            (articles_by_id[article_id] for article_id in member_ids),
            key=lambda item: (item.published_at, item.id),
        )
        rationale = (
            f"merged after retrieval: identical full text across {len(group)} "
            f"clusters ({', '.join(cluster.id for cluster in group)}); "
            f"{group[0].rationale}"
        )
        cluster = StoryCluster(
            id=stable_cluster_id(member_ids),
            article_ids=member_ids,
            representative_article_id=best.id,
            rationale=rationale,
            created_at=now,
        )
        store.save(cluster)
        merged.append(cluster)
        absorbed += len(group) - 1
    return merged, absorbed


@ls.traceable(
    name="screen-clusters",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def screen_clusters(
    clusters: tuple[StoryCluster, ...],
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
    *,
    now: datetime | None = None,
    judge: RelevanceJudge | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    published_article_ids: frozenset[str] = frozenset(),
    policy_version: str = POLICY_VERSION,
    retrieve_texts: bool = True,
    fetcher: Fetcher | None = None,
) -> ScreeningSummary:
    """Apply date/history rules, judge relevance, and shortlist stories.

    Deterministic rules run first (missing representative, recycled coverage,
    36-hour age cap with a recorded 24-36h late-arrival lane). Survivors go to
    the judge with verdicts cached by content, policy, prompt, and model
    version. Malformed judge output and transport failures become
    review-preserved borderline items; only transport failures skip the cache.
    Kept stories fill the cap with relevant items first, then borderline, get
    full-text retrieval, and merge on identical retrieved text.
    """

    started = now or datetime.now(UTC)
    active_judge = judge or build_judge(Settings())
    relevant: list[tuple[StoryCluster, Article, bool]] = []
    borderline: list[tuple[StoryCluster, Article, bool]] = []
    decisions: list[ScreeningDecision] = []
    judge_calls = 0
    judge_cached = 0
    judge_errors = 0

    for cluster in sorted(clusters, key=lambda item: item.id):
        article = articles_by_id.get(cluster.representative_article_id)
        if article is None:
            decisions.append(
                ScreeningDecision(
                    cluster_id=cluster.id,
                    verdict="rejected",
                    reason="representative article missing from store",
                )
            )
            continue
        if published_article_ids and all(
            article_id in published_article_ids for article_id in cluster.article_ids
        ):
            decisions.append(
                ScreeningDecision(
                    cluster_id=cluster.id,
                    verdict="rejected",
                    reason="already published; no meaningful follow-up",
                )
            )
            continue
        age_hours = max(0.0, (started - article.published_at).total_seconds() / 3600.0)
        if age_hours > MAX_AGE_HOURS:
            decisions.append(
                ScreeningDecision(
                    cluster_id=cluster.id,
                    verdict="rejected",
                    reason=f"published {age_hours:.0f}h ago, outside 36h window",
                )
            )
            continue
        late = age_hours > FRESH_HOURS
        excerpt = _cluster_excerpt(cluster, articles_by_id, store)
        key = cache_key(
            article.title,
            excerpt,
            model=active_judge.identity,
            policy_version=policy_version,
        )
        cached = cache_lookup(store, key)
        if cached is not None:
            verdict = cached
            was_cached = True
            judge_cached += 1
        else:
            judge_calls += 1
            try:
                verdict = judge_with_retries(
                    active_judge,
                    article.title,
                    excerpt,
                    _source_name(article, store),
                )
            except JudgeOutputError as exc:
                verdict = RelevanceVerdict(
                    relevant=False,
                    borderline=True,
                    reason=f"judge output unusable, preserved for review: {exc}",
                )
                cache_store(store, key, verdict)
            except JudgeError as exc:
                judge_errors += 1
                verdict = RelevanceVerdict(
                    relevant=False,
                    borderline=True,
                    reason=f"judge unavailable, preserved for review: {exc}",
                )
            else:
                cache_store(store, key, verdict)
            was_cached = False
        if verdict.borderline:
            borderline.append((cluster, article, late))
            decisions.append(
                ScreeningDecision(
                    cluster_id=cluster.id,
                    verdict="borderline",
                    reason=verdict.reason,
                    late_arrival=late,
                    cached=was_cached,
                )
            )
        elif verdict.relevant:
            relevant.append((cluster, article, late))
            decisions.append(
                ScreeningDecision(
                    cluster_id=cluster.id,
                    verdict="shortlisted",
                    reason=verdict.reason,
                    late_arrival=late,
                    cached=was_cached,
                )
            )
        else:
            decisions.append(
                ScreeningDecision(
                    cluster_id=cluster.id,
                    verdict="rejected",
                    reason=verdict.reason,
                    late_arrival=late,
                    cached=was_cached,
                )
            )

    relevant.sort(key=lambda item: item[1].published_at, reverse=True)
    borderline.sort(key=lambda item: item[1].published_at, reverse=True)
    kept = (relevant + borderline)[:max_candidates]
    kept_ids = {cluster.id for cluster, _, _ in kept}
    for decision in decisions:
        if decision.verdict != "rejected" and decision.cluster_id not in kept_ids:
            decision.verdict = "rejected"
            decision.reason = (
                f"outside top-{max_candidates} shortlist: {decision.reason}"
            )
    kept_clusters = [cluster for cluster, _, _ in kept]
    if retrieve_texts and kept_clusters:
        retrieve_articles(
            tuple(
                articles_by_id[cluster.representative_article_id]
                for cluster in kept_clusters
            ),
            store,
            now=started,
            fetcher=fetcher,
        )
        kept_clusters, absorbed = _merge_identical_texts(
            kept_clusters, articles_by_id, store, started
        )
    else:
        absorbed = 0
    summary = ScreeningSummary(
        attempted=len(clusters),
        shortlisted=sum(1 for item in decisions if item.verdict == "shortlisted"),
        borderline=sum(1 for item in decisions if item.verdict == "borderline"),
        rejected=sum(1 for item in decisions if item.verdict == "rejected"),
        judge_calls=judge_calls,
        judge_cached=judge_cached,
        judge_errors=judge_errors,
        merged=absorbed,
        shortlist=tuple(kept_clusters),
        decisions=tuple(decisions),
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    _LOGGER.info(
        "screening_complete",
        attempted=summary.attempted,
        shortlisted=summary.shortlisted,
        borderline=summary.borderline,
        rejected=summary.rejected,
        merged=summary.merged,
    )
    return summary


__all__ = [
    "DEFAULT_JUDGE_MAX_TOKENS",
    "DEFAULT_JUDGE_TIMEOUT",
    "DEFAULT_MAX_CANDIDATES",
    "FRESH_HOURS",
    "MAX_AGE_HOURS",
    "POLICY_VERSION",
    "PROMPT_VERSION",
    "JudgeConfigurationError",
    "JudgeError",
    "JudgeOutputError",
    "OpenRouterJudge",
    "RelevanceJudge",
    "RelevanceVerdict",
    "ScreeningDecision",
    "ScreeningSummary",
    "build_judge",
    "cache_key",
    "cache_lookup",
    "cache_store",
    "judge_prompt",
    "judge_with_retries",
    "parse_verdict",
    "screen_clusters",
]
