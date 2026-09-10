"""Verify draft summaries with deterministic checks and bounded repair."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, TypedDict

import httpx
import langsmith as ls
import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ai_news_agent.config import Settings
from ai_news_agent.observability import redact
from ai_news_agent.schemas import (
    Article,
    ArticleText,
    Evidence,
    FindingSeverity,
    Score,
    StoryCluster,
    Summary,
    SummaryStatus,
    VerificationFinding,
    VerificationResult,
    VerificationVerdict,
)
from ai_news_agent.storage import SQLiteStore
from ai_news_agent.summaries import SummaryWriter, generate_summaries

_LOGGER = structlog.get_logger(__name__)

POLICY_VERSION = "editorial-policy-2026-09-07"
PROMPT_VERSION = "verification-v1"
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_JUDGE_TIMEOUT = 30.0
DEFAULT_JUDGE_MAX_TOKENS = 600
MAX_FRESH_HOURS = 36.0
MIN_VERIFIED_FOR_DELIVERY = 3

_VERIFIER_SYSTEM = (
    "You verify grounded news summaries against their cited evidence. "
    "Check grounding, relevance, freshness, duplication, and whether an "
    "excluded contender is clearly stronger. "
    "Reply with exactly one JSON object and no other text."
)
_VERIFIER_OUTPUT_SCHEMA = (
    '{"verdict": "pass"|"revise"|"replace"|"reject", '
    '"findings": [{"severity": "info"|"warning"|"error", "message": string, '
    '"claim": string|null, "evidence_ids": [string]}]}'
)

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


class VerificationError(RuntimeError):
    """Raised when the verifier cannot be reached."""


class VerificationOutputError(VerificationError):
    """Raised when the verifier returns malformed output."""


class VerificationConfigurationError(RuntimeError):
    """Raised when verifier credentials are missing or rejected."""


@dataclass(slots=True)
class ModelVerdict:
    verdict: VerificationVerdict
    findings: tuple[VerificationFinding, ...]


@dataclass(slots=True)
class VerificationDecision:
    cluster_id: str
    outcome: str
    attempts: int = 0
    findings: tuple[VerificationFinding, ...] = ()
    summary_id: str | None = None
    replacement_id: str | None = None


@dataclass(slots=True)
class VerificationSummary:
    attempted: int = 0
    verified: int = 0
    rejected: int = 0
    replaced: int = 0
    shortfall: int = 0
    verified_ids: tuple[str, ...] = ()
    decisions: tuple[VerificationDecision, ...] = ()
    meets_minimum: bool = False
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    termination_reason: str = "complete"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ModelVerifier(Protocol):
    """Judge one draft summary against evidence and contenders."""

    @property
    def identity(self) -> str:
        """Return a stable model identifier for tracing."""
        ...

    def check(
        self,
        sentences: Sequence[str],
        evidence_text: str,
        contenders_note: str,
    ) -> ModelVerdict:
        """Return an explicit verdict with claim-level findings."""
        ...


def verifier_prompt(
    sentences: Sequence[str],
    evidence_text: str,
    contenders_note: str,
) -> str:
    """Build the versioned verification prompt for one draft summary."""

    numbered = "\n".join(f"{index}. {item}" for index, item in enumerate(sentences, 1))
    return (
        f"DRAFT SUMMARY (three sentences):\n{numbered}\n\n"
        f"CITED EVIDENCE:\n{evidence_text[:6000]}\n\n"
        f"EXCLUDED CONTENDERS:\n{contenders_note}\n\n"
        "Check grounding (numbers, dates, names, status match evidence), "
        "relevance, freshness, duplication, and missed stronger candidates. "
        "Return one finding per claim with severity, message, claim, and "
        "evidence IDs. Verdict pass only when every sentence is supported; "
        "revise for fixable grounding; replace when a contender is stronger; "
        "reject when unsalvageable. "
        f"Reply with exactly one JSON object like {_VERIFIER_OUTPUT_SCHEMA}."
    )


def parse_model_verdict(raw: str) -> ModelVerdict:
    """Parse strict verifier JSON into a verdict with findings."""

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise VerificationOutputError(f"verifier returned no JSON: {raw[:120]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VerificationOutputError(f"verifier returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationOutputError("verifier returned a non-object verdict")
    try:
        verdict_raw = payload["verdict"]
        findings_raw = payload.get("findings", [])
    except AttributeError as exc:
        raise VerificationOutputError(f"verifier verdict malformed: {exc}") from exc
    try:
        verdict = VerificationVerdict(verdict_raw)
    except ValueError as exc:
        raise VerificationOutputError(f"unknown verdict {verdict_raw!r}") from exc
    if not isinstance(findings_raw, list):
        raise VerificationOutputError("findings must be a list")
    findings: list[VerificationFinding] = []
    for entry in findings_raw:
        if not isinstance(entry, dict):
            raise VerificationOutputError("each finding must be an object")
        try:
            findings.append(
                VerificationFinding(
                    severity=FindingSeverity(entry["severity"]),
                    message=str(entry["message"]),
                    claim=entry.get("claim"),
                    evidence_ids=tuple(entry.get("evidence_ids", ())),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise VerificationOutputError(f"invalid finding: {exc}") from exc
    return ModelVerdict(verdict=verdict, findings=tuple(findings))


class OpenRouterVerifier:
    """Separate verifier call behind OpenRouter's OpenAI-compatible API."""

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

    def check(
        self,
        sentences: Sequence[str],
        evidence_text: str,
        contenders_note: str,
    ) -> ModelVerdict:
        """Call the verifier once and parse its strict JSON verdict."""

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "temperature": 0,
                    "max_tokens": self._max_tokens,
                    "messages": [
                        {"role": "system", "content": _VERIFIER_SYSTEM},
                        {
                            "role": "user",
                            "content": verifier_prompt(
                                sentences, evidence_text, contenders_note
                            ),
                        },
                    ],
                },
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise VerificationError(f"verifier timeout after {self._timeout}s") from exc
        except httpx.HTTPError as exc:
            raise VerificationError(f"verifier transport error: {exc}") from exc
        if response.status_code in (401, 403):
            raise VerificationConfigurationError(
                f"verifier credentials rejected (HTTP {response.status_code})"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise VerificationError(f"verifier HTTP {response.status_code}, retryable")
        if response.status_code >= 400:
            raise VerificationError(
                f"verifier HTTP {response.status_code}, not retryable"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise VerificationOutputError(
                f"verifier returned an unexpected envelope: {exc}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise VerificationOutputError("verifier returned empty content")
        return parse_model_verdict(content)


def build_verifier(
    settings: Settings,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> OpenRouterVerifier:
    """Build the configured verifier, failing fast without credentials."""

    if not settings.judge_ready or settings.openrouter_api_key is None:
        raise VerificationConfigurationError(
            "set OPENROUTER_API_KEY before model-judged verification"
        )
    return OpenRouterVerifier(
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def evidence_text_for(
    summary: Summary,
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
) -> str:
    """Collect the cited evidence text for one draft summary."""

    parts: list[str] = []
    article = articles_by_id.get(summary.representative_article_id)
    if article is not None:
        body = store.get(ArticleText, f"{article.id}-text")
        if body is not None:
            parts.append(body.text[:4000])
    for group in summary.evidence_by_sentence:
        for evidence_id in group:
            record = store.get(Evidence, evidence_id)
            if record is not None:
                parts.append(record.excerpt[:1000])
                continue
            text = store.get(ArticleText, evidence_id)
            if text is not None:
                parts.append(text.text[:1000])
    return "\n".join(parts)


def deterministic_findings(
    summary: Summary,
    article: Article | None,
    evidence_text: str,
    seen_canonicals: set[str],
    *,
    now: datetime,
) -> list[VerificationFinding]:
    """Run code checks for evidence, numbers, freshness, and duplication."""

    findings: list[VerificationFinding] = []
    if article is None:
        findings.append(
            VerificationFinding(
                severity=FindingSeverity.ERROR,
                message="representative article missing from store",
            )
        )
        return findings
    canonical = str(article.canonical_url).strip().lower()
    if canonical in seen_canonicals:
        findings.append(
            VerificationFinding(
                severity=FindingSeverity.ERROR,
                message="duplicate event already represented in selection",
            )
        )
    else:
        seen_canonicals.add(canonical)
    flat_ids = [item for group in summary.evidence_by_sentence for item in group]
    if not flat_ids:
        findings.append(
            VerificationFinding(
                severity=FindingSeverity.ERROR,
                message="summary cites no evidence",
            )
        )
    lowered = evidence_text.lower()
    for sentence in summary.sentences:
        for number in _NUMBER.findall(sentence):
            if number.lower() not in lowered:
                findings.append(
                    VerificationFinding(
                        severity=FindingSeverity.ERROR,
                        message=f"unsupported number {number!r} not in evidence",
                        claim=sentence[:200],
                    )
                )
    age_hours = max(0.0, (now - article.published_at).total_seconds() / 3600.0)
    if age_hours > MAX_FRESH_HOURS:
        findings.append(
            VerificationFinding(
                severity=FindingSeverity.ERROR,
                message=f"stale story published {age_hours:.0f}h ago",
            )
        )
    return findings


def contenders_note_for(
    cluster_id: str,
    scores_by_cluster: dict[str, Score],
    reserve_ids: Sequence[str],
) -> str:
    """Describe excluded contenders for missed-candidate checks."""

    lines = []
    for reserve_id in reserve_ids:
        score = scores_by_cluster.get(reserve_id)
        if score is None or reserve_id == cluster_id:
            continue
        lines.append(f"- {reserve_id}: total {score.weighted_total:.1f}")
    return "\n".join(lines) if lines else "no excluded contenders"


@ls.traceable(
    name="verify-summary",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def verify_one(
    summary: Summary,
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
    *,
    now: datetime,
    verifier: ModelVerifier | None = None,
    scores_by_cluster: dict[str, Score] | None = None,
    reserve_ids: Sequence[str] = (),
    seen_canonicals: set[str] | None = None,
    attempt: int = 1,
) -> VerificationResult:
    """Verify one draft: deterministic checks first, then the model judge."""

    active = verifier or build_verifier(Settings())
    article = articles_by_id.get(summary.representative_article_id)
    evidence_text = evidence_text_for(summary, articles_by_id, store)
    findings = deterministic_findings(
        summary, article, evidence_text, seen_canonicals or set(), now=now
    )
    fatal = [item for item in findings if item.severity is FindingSeverity.ERROR]
    if fatal:
        verdict = VerificationVerdict.REJECT
        if any("duplicate" in item.message for item in fatal):
            verdict = VerificationVerdict.REPLACE
        model_findings: tuple[VerificationFinding, ...] = ()
    else:
        try:
            checked = active.check(
                summary.sentences,
                evidence_text,
                contenders_note_for(
                    summary.story_cluster_id, scores_by_cluster or {}, reserve_ids
                ),
            )
        except VerificationConfigurationError:
            raise
        except VerificationOutputError as exc:
            findings.append(
                VerificationFinding(
                    severity=FindingSeverity.ERROR,
                    message=f"verifier output unusable: {exc}",
                )
            )
            checked = ModelVerdict(verdict=VerificationVerdict.REJECT, findings=())
        except VerificationError as exc:
            findings.append(
                VerificationFinding(
                    severity=FindingSeverity.WARNING,
                    message=f"verifier unavailable: {exc}",
                )
            )
            checked = ModelVerdict(verdict=VerificationVerdict.REJECT, findings=())
        verdict = checked.verdict
        model_findings = checked.findings
    result = VerificationResult(
        id=f"{summary.id}-v{attempt}",
        summary_id=summary.id,
        verdict=verdict,
        findings=tuple(list(findings) + list(model_findings)),
        checked_at=datetime.now(UTC),
        attempt=attempt,
    )
    store.save(result)
    return result


class _GraphState(TypedDict):
    pending: list[str]
    attempts: dict[str, int]
    verified: list[str]
    outcomes: dict[str, str]
    replacements: dict[str, str]


def _set_status(store: SQLiteStore, summary_id: str, status: SummaryStatus) -> None:
    for record in store.iter_records(Summary):
        if record.id == summary_id:
            store.save(
                Summary(
                    id=record.id,
                    story_cluster_id=record.story_cluster_id,
                    representative_article_id=record.representative_article_id,
                    sentences=record.sentences,
                    evidence_by_sentence=record.evidence_by_sentence,
                    status=status,
                )
            )
            return


@ls.traceable(
    name="verification-workflow",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def run_verification(
    selected_ids: Sequence[str],
    clusters_by_id: dict[str, StoryCluster],
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
    *,
    now: datetime | None = None,
    verifier: ModelVerifier | None = None,
    writer: SummaryWriter | None = None,
    scores_by_cluster: dict[str, Score] | None = None,
    reserve_ids: Sequence[str] = (),
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    digest_run_id: str = "manual-verification",
) -> VerificationSummary:
    """Verify selected drafts with bounded revision and reserve replacement.

    Deterministic checks run before any model call. Revise regenerates the
    draft once through the summary writer and reverifies; replace swaps in
    the next reserve candidate. Retries terminate after max_attempts and
    unresolved items are excluded with a shortfall. Only verified summaries
    reach the final digest; resumed executions upsert by stable IDs.
    """

    started = now or datetime.now(UTC)
    active_verifier = verifier or build_verifier(Settings())
    scores = scores_by_cluster or {}
    checkpointer = MemorySaver()
    graph = StateGraph(_GraphState)

    def _verify_node(state: _GraphState) -> _GraphState:
        pending = list(state["pending"])
        attempts = dict(state["attempts"])
        verified = list(state["verified"])
        outcomes = dict(state["outcomes"])
        replacements = dict(state["replacements"])
        seen: set[str] = set()
        for cluster_id in verified:
            summary = store.get(Summary, f"{cluster_id}-summary")
            article = None
            if summary is not None:
                article = articles_by_id.get(summary.representative_article_id)
            if article is not None:
                seen.add(str(article.canonical_url).strip().lower())
        reserve_pool = [item for item in reserve_ids if item not in verified]
        for cluster_id in list(pending):
            cluster = clusters_by_id.get(cluster_id)
            summary = store.get(Summary, f"{cluster_id}-summary")
            if cluster is None or summary is None:
                outcomes[cluster_id] = "rejected"
                pending.remove(cluster_id)
                continue
            attempt = attempts.get(cluster_id, 0) + 1
            attempts[cluster_id] = attempt
            result = verify_one(
                summary,
                articles_by_id,
                store,
                now=started,
                verifier=active_verifier,
                scores_by_cluster=scores,
                reserve_ids=tuple(reserve_pool),
                seen_canonicals=seen,
                attempt=attempt,
            )
            if result.verdict is VerificationVerdict.PASS:
                _set_status(store, summary.id, SummaryStatus.VERIFIED)
                verified.append(cluster_id)
                outcomes[cluster_id] = "verified"
                pending.remove(cluster_id)
            elif (
                result.verdict is VerificationVerdict.REVISE and attempt < max_attempts
            ):
                outcomes[cluster_id] = "revise"
            elif result.verdict is VerificationVerdict.REPLACE:
                swapped = False
                while reserve_pool:
                    candidate = reserve_pool.pop(0)
                    if candidate in verified or candidate in pending:
                        continue
                    candidate_cluster = clusters_by_id.get(candidate)
                    if candidate_cluster is None:
                        continue
                    candidate_summary = store.get(Summary, f"{candidate}-summary")
                    if candidate_summary is None and writer is not None:
                        generate_summaries(
                            (candidate_cluster,),
                            articles_by_id,
                            store,
                            now=started,
                            writer=writer,
                        )
                        candidate_summary = store.get(Summary, f"{candidate}-summary")
                    if candidate_summary is None:
                        continue
                    replacements[cluster_id] = candidate
                    outcomes[cluster_id] = "replaced"
                    pending.remove(cluster_id)
                    pending.append(candidate)
                    attempts.setdefault(candidate, 0)
                    swapped = True
                    break
                if not swapped:
                    outcomes[cluster_id] = "rejected"
                    _set_status(store, summary.id, SummaryStatus.REJECTED)
                    pending.remove(cluster_id)
            else:
                if (
                    attempt >= max_attempts
                    and result.verdict
                    in (
                        VerificationVerdict.REVISE,
                        VerificationVerdict.REPLACE,
                    )
                ) or result.verdict is VerificationVerdict.REJECT:
                    outcomes[cluster_id] = "rejected"
                    _set_status(store, summary.id, SummaryStatus.REJECTED)
                    pending.remove(cluster_id)
        return {
            "pending": pending,
            "attempts": attempts,
            "verified": verified,
            "outcomes": outcomes,
            "replacements": replacements,
        }

    def _repair_node(state: _GraphState) -> _GraphState:
        pending = list(state["pending"])
        for cluster_id in list(pending):
            if state["outcomes"].get(cluster_id) != "revise":
                continue
            cluster = clusters_by_id.get(cluster_id)
            if cluster is None or writer is None:
                continue
            generate_summaries(
                (cluster,), articles_by_id, store, now=started, writer=writer
            )
            state["outcomes"][cluster_id] = "reverified"
        return state

    def _should_repair(state: _GraphState) -> str:
        repairable = [
            item
            for item in state["pending"]
            if state["outcomes"].get(item) in ("revise", "reverified")
            and state["attempts"].get(item, 0) < max_attempts
        ]
        unexamined = [
            item for item in state["pending"] if item not in state["outcomes"]
        ]
        return "repair" if repairable or unexamined else END

    graph.add_node("verify", _verify_node)
    graph.add_node("repair", _repair_node)
    graph.add_edge(START, "verify")
    graph.add_conditional_edges(
        "verify", _should_repair, {"repair": "repair", END: END}
    )
    graph.add_edge("repair", "verify")
    compiled = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": digest_run_id}}
    final = compiled.invoke(
        {
            "pending": list(selected_ids),
            "attempts": {},
            "verified": [],
            "outcomes": {},
            "replacements": {},
        },
        config=config,
    )

    verified_ids = tuple(final["verified"])
    decisions: list[VerificationDecision] = []
    replaced = 0
    for cluster_id in list(selected_ids) + [
        item for item in final["verified"] if item not in selected_ids
    ]:
        outcome = final["outcomes"].get(cluster_id, "rejected")
        summary = store.get(Summary, f"{cluster_id}-summary")
        findings: tuple[VerificationFinding, ...] = ()
        latest = store.get(
            VerificationResult,
            f"{cluster_id}-summary-v{final['attempts'].get(cluster_id, 1)}",
        )
        if latest is not None:
            findings = latest.findings
        if outcome == "replaced":
            replaced += 1
        decisions.append(
            VerificationDecision(
                cluster_id=cluster_id,
                outcome=outcome,
                attempts=final["attempts"].get(cluster_id, 0),
                findings=findings,
                summary_id=summary.id if summary is not None else None,
                replacement_id=final["replacements"].get(cluster_id),
            )
        )
    shortfall = max(0, len(selected_ids) - len(verified_ids))
    summary = VerificationSummary(
        attempted=len(selected_ids),
        verified=len(verified_ids),
        rejected=sum(1 for item in decisions if item.outcome == "rejected"),
        replaced=replaced,
        shortfall=shortfall,
        verified_ids=verified_ids,
        decisions=tuple(sorted(decisions, key=lambda item: item.cluster_id)),
        meets_minimum=len(verified_ids) >= MIN_VERIFIED_FOR_DELIVERY,
        model=active_verifier.identity,
        prompt_version=PROMPT_VERSION,
        termination_reason="complete",
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    _LOGGER.info(
        "verification_complete",
        attempted=summary.attempted,
        verified=summary.verified,
        rejected=summary.rejected,
        replaced=summary.replaced,
        shortfall=summary.shortfall,
        model=summary.model,
        prompt_version=summary.prompt_version,
    )
    return summary


def verified_summaries_for_delivery(
    verified_ids: Sequence[str],
    store: SQLiteStore,
) -> tuple[Summary, ...]:
    """Return only verified summaries, so drafts never enter delivery."""

    ordered: list[Summary] = []
    for cluster_id in verified_ids:
        summary = store.get(Summary, f"{cluster_id}-summary")
        if summary is not None and summary.status is SummaryStatus.VERIFIED:
            ordered.append(summary)
    return tuple(ordered)


__all__ = [
    "DEFAULT_JUDGE_MAX_TOKENS",
    "DEFAULT_JUDGE_TIMEOUT",
    "DEFAULT_MAX_ATTEMPTS",
    "MAX_FRESH_HOURS",
    "MIN_VERIFIED_FOR_DELIVERY",
    "POLICY_VERSION",
    "PROMPT_VERSION",
    "ModelVerdict",
    "ModelVerifier",
    "OpenRouterVerifier",
    "VerificationConfigurationError",
    "VerificationDecision",
    "VerificationError",
    "VerificationOutputError",
    "VerificationSummary",
    "build_verifier",
    "contenders_note_for",
    "deterministic_findings",
    "evidence_text_for",
    "parse_model_verdict",
    "run_verification",
    "verified_summaries_for_delivery",
    "verifier_prompt",
    "verify_one",
]
