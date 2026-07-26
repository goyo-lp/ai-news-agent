"""Quality gate for LinkedIn post proposals — deterministic rules only.

The P2.5 plan asks for word-count 105–182, single-topic, anti-hype checks.
The reference LinkedIn agent had these baked into
``post_generator._validate_posts`` (a method tied to the post-generation node)
rather than a standalone gate; pulling them out into a deterministic service
makes the rule set invokable at any point in the coordinator's loop —
pre-delivery retry path, post-write audit, parity snapshots — and testable in
isolation.

Checks (each returns its own reason-on-fail note, merged into the result's
``reasons`` list):

  1. **word count**: ``PostProposal.body`` (after a light whitespace clean and
     jargon simplification, mirroring the reference's cleaner) is between
     ``_MIN_POST_WORDS`` (105) and ``_MAX_POST_WORDS`` (182). A draft under the
     minimum is *not* auto-padded here — that's a writer-side concern; the gate
     reports the gap, the coordinator's retry loop asks the writer to expand.
  2. **single topic**: ``supporting_topic_ids`` must contain exactly one id — a
     proposal that backs onto two topics is ambiguous voice; the writer's prompt
     is per-topic.
  3. **anti-hype**: ``has_hype`` flags the sales/marketing-shibboleth set the
     LinkedIn voice skill (P3.1) also calls out (``game changer``,
     ``revolutionary``, ``must-have``, ``10x``, ``act now``, ``can't miss``,
     ``unlock``). Hype markers fail the gate rather than silently rewrite — the
     writer is supposed to avoid them; a fail-with-reason is the retry signal.
  4. **citation fidelity** (:func:`citation_fidelity_reasons`, merged by the
     tool layer): every ``citation_urls`` entry must trace to the brief's
     ``citations``. The coordinator guardrail always *claimed* this check as
     the no-fabricated-URLs backstop; before 2026-07-26 the gate had no such
     check and URL fidelity rested entirely on LLM honesty.

Helpers ported verbatim from the reference where their behavior is documented:

  * :func:`_strip_salesy_language`, :func:`_strip_access_failure_language`,
    :func:`_simplify_jargon_for_general_audience` — the cleaning pipeline a
    body runs through *before* its word count is measured, so the gate's count
    matches the count a reader sees post-cleaning.
  * :func:`_normalize_hashtags` — used for the secondary check that hashtags
    are well-formed. NOT used to silently pad a too-short hashtag list (that's
    a writer-side concern; the gate reports, the writer fixes).

Intentional divergences from the reference (both documented):
  1. The reference silently **rewrote** posts that violated these rules; the
     gate reports ``passed=False`` + ``reasons`` and lets the coordinator ask
     the writer (P4.2) to retry. This is what a deterministic-gate tool *is* —
     the writer owns the model-output churn, the gate owns the verdict.
  2. ``_validate_posts`` also repaired missing hashtags / citations /
     supporting_topic_ids from the briefs. The gate doesn't mutate the
     proposal — those fallbacks are writer-side. The gate just reports.

The output :class:`QualityResult` is orchestrator-internal (not a boundary
contract in :mod:`app.orchestrator.schemas`) — parity with the boundary is
the :class:`PostProposal` and its verified-brief back-reference, not the gate
verdict. Promoting ``QualityResult`` into ``schemas.py`` is deferred until a
downstream consumer other than the tool that wraps this service depends on its
shape.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.orchestrator.schemas import PostProposal, ResearchBrief
from app.services.rss_client import normalize_url

logger = logging.getLogger(__name__)

_MIN_POST_WORDS = 105
_MAX_POST_WORDS = 182

_HYPE_MARKERS = {
    "game changer",
    "revolutionary",
    "must-have",
    "10x",
    "act now",
    "can't miss",
    "cant miss",
    "unlock",
    "break the internet",
    "unbelievable",
}

_CTA_ENDINGS = re.compile(
    r"\b(what do you think\??|curious to hear your take\.?|let me know\.?|thoughts\?)$",
    re.IGNORECASE,
)

_SALESY_REPLACEMENTS = {
    "game changer": "notable shift",
    "must-have": "worth evaluating",
    "revolutionary": "meaningful",
    "unlock": "enable",
    "10x": "significant",
    "can't miss": "relevant",
    "act now": "monitor closely",
}

_ACCESS_FAILURE_PATTERNS = (
    r"\b(i|we)\s+(can(?:not|'t)|could(?:\s+not|n't)|did(?:\s+not|n't))\s+(access|verify|find|retrieve)[^.!?]*[.!?]?",
    r"\bno\s+(abstract|methodology|full\s+text|authors)\b[^.!?]*[.!?]?",
    r"\bfrom\s+what\s+is\s+publicly\s+indexed\b[^.!?]*[.!?]?",
)
_ACCESS_FAILURE_REPLACEMENT = (
    "Public evidence is still limited and should be treated as directional."
)

_JARGON_REPLACEMENTS = {
    "implementation": "real-world use",
    "orchestration": "coordination",
    "inference": "model use",
    "latency": "response time",
    "throughput": "how much work it can handle",
    "benchmark": "test result",
    "multimodal": "works with text, images, or audio",
    "retrieval": "finding the right information",
    "context window": "working memory",
    "architecture": "system design",
    "evaluation": "testing",
    "tradeoffs": "pros and cons",
    "directional signal": "early signal",
}


@dataclass
class QualityResult:
    """Verdict from ``check_proposal``. ``passed`` is the binary gate; each
    failing check contributes to ``reasons`` (human-readable strings surfaced
    to the LLM and to LangSmith traces). The cleaned body and computed metrics
    ride along so the tool that wraps this service can persist them once
    without re-running the cleaner."""

    passed: bool
    word_count: int
    hashtags_count: int
    single_topic: bool
    has_hype: bool
    hype_markers: list[str] = field(default_factory=list)
    cleaned_body: str = ""
    reasons: list[str] = field(default_factory=list)


def count_words(text: str) -> int:
    """Whitespace-split word count — the same count ``_enforce_word_window``
    would use to size a draft, so the gate and any future writer-side window
    enforcer agree on definition."""
    return len(text.split())


def has_hype(text: str) -> tuple[bool, list[str]]:
    """Scan lowercased text for any hype marker; returns (present?, markers).
    Multiple markers each surface distinctly in the result so the writer knows
    which phrasing to remove."""
    lowered = text.lower()
    found: list[str] = []
    for marker in _HYPE_MARKERS:
        if marker in lowered:
            found.append(marker)
    return bool(found), found


def _strip_salesy_language(text: str) -> str:
    cleaned = text
    for source, target in _SALESY_REPLACEMENTS.items():
        cleaned = re.sub(source, target, cleaned, flags=re.IGNORECASE)
    cleaned = _CTA_ENDINGS.sub("", cleaned)
    return " ".join(cleaned.split())


def _strip_access_failure_language(text: str) -> str:
    cleaned = text
    for pattern in _ACCESS_FAILURE_PATTERNS:
        cleaned = re.sub(pattern, _ACCESS_FAILURE_REPLACEMENT, cleaned, flags=re.IGNORECASE)
    return cleaned


def _simplify_jargon_for_general_audience(text: str) -> str:
    simplified = text
    for source, target in _JARGON_REPLACEMENTS.items():
        simplified = re.sub(
            rf"\b{re.escape(source)}\b", target, simplified, flags=re.IGNORECASE
        )
    simplified = simplified.replace(";", ".")
    return re.sub(r"\s+", " ", simplified).strip()


def _clean_post_body(text: str) -> str:
    """Run the same cleaning pipeline the reference's post_generator applies
    before its own word-count check, so the gate measures the body a reader
    sees post-cleanup, not the raw writer draft."""
    if not text:
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text).splitlines()]
    merged = "\n".join(line for line in lines if line)
    merged = _strip_salesy_language(merged)
    merged = _strip_access_failure_language(merged)
    merged = _simplify_jargon_for_general_audience(merged)
    return " ".join(merged.split())


def normalize_hashtags(hashtags: list[str]) -> list[str]:
    """Canonical hashtag form: ``#``-prefixed, no internal whitespace, deduped,
    preserves input order. Ported from reference; used by the gate only for a
    well-formedness check (each entry, once normalized, must remain non-empty)."""
    normalized: list[str] = []
    for tag in hashtags:
        trimmed = (tag or "").strip()
        if not trimmed:
            continue
        if not trimmed.startswith("#"):
            trimmed = f"#{trimmed}"
        trimmed = re.sub(r"\s+", "", trimmed)
        if trimmed and trimmed not in normalized:
            normalized.append(trimmed)
    return normalized


def check_proposal(proposal: PostProposal) -> QualityResult:
    """Run the deterministic gate on one :class:`PostProposal` and return a
    :class:`QualityResult` with the verdict, computed metrics, and per-check
    failure reasons. Pure (no network, no mutation) — testable with synthetic
    fixtures.

    The body is run through :func:`_clean_post_body` *before* the word-count
    check, so the gate measures the body a reader sees post-cleaning. The
    cleaned body is returned in the result so the tool layer can persist it
    once without re-running the cleaner.

    Hype detection runs on the **raw** body — the cleaner legitimately rewrites
    some hype markers (`game changer` → `notable shift`, `revolutionary` →
    `meaningful`) before the word-count step, and that rewrite is the *correct*
    thing for a writer-retry path's first pass. But running the detector on the
    cleaned body would silently launder those 7 of 10 markers past the gate —
    `passed=True, has_hype=False` for `"This is a game changer."` — which
    contradicts the docstring's "hype markers fail the gate rather than
    silently rewrite" contract. So detection sees the writer's actual phrasing
    and fails; the cleaner only affects the persisted ``cleaned_body`` + the
    word-count metric."""
    reasons: list[str] = []
    cleaned = _clean_post_body(proposal.body)
    wc = count_words(cleaned)

    if wc < _MIN_POST_WORDS:
        reasons.append(f"body too short: {wc} words (need at least {_MIN_POST_WORDS})")
    elif wc > _MAX_POST_WORDS:
        reasons.append(f"body too long: {wc} words (cap {_MAX_POST_WORDS})")

    topic_ids = proposal.supporting_topic_ids
    if not topic_ids or len(topic_ids) != 1:
        reasons.append(
            f"expected exactly one supporting_topic_id, got {len(topic_ids)}"
        )

    hype_present, hype_markers = has_hype(proposal.body)
    if hype_present:
        reasons.append("hype language detected: " + ", ".join(sorted(hype_markers)))

    hashtags = normalize_hashtags(proposal.hashtags)
    if not hashtags:
        reasons.append("no usable hashtags after normalization")
    elif len(hashtags) < 3:
        reasons.append(
            f"only {len(hashtags)} usable hashtag(s); writer should add at least 3"
        )

    return QualityResult(
        passed=not reasons,
        word_count=wc,
        hashtags_count=len(hashtags),
        single_topic=len(topic_ids) == 1,
        has_hype=hype_present,
        hype_markers=hype_markers,
        cleaned_body=cleaned,
        reasons=reasons,
    )


def citation_fidelity_reasons(
    proposal: PostProposal, brief: ResearchBrief | None
) -> list[str]:
    """Check every draft citation traces to the brief's own citations.

    This is the deterministic backstop for the coordinator's "no fabricated
    URLs" guardrail — the check the guardrail always claimed the gate ran.
    URLs are compared after ``normalize_url`` (tracking params stripped,
    scheme/host normalized) so a citation that differs only by UTM noise
    still matches. A missing brief fails: an untraceable citation set is
    exactly what this check exists to catch.

    Kept separate from :func:`check_proposal` (which stays pure over the
    proposal alone) and merged by the tool layer, which owns brief loading.
    """
    if brief is None:
        topic = proposal.supporting_topic_ids[0] if proposal.supporting_topic_ids else "<none>"
        return [
            f"citation fidelity unverifiable: no brief found for topic {topic!r} "
            "— citations cannot be traced"
        ]
    allowed = {normalize_url(c.url) for c in brief.citations if c.url.strip()}
    reasons: list[str] = []
    for url in proposal.citation_urls:
        if not url.strip():
            continue
        if normalize_url(url) not in allowed:
            reasons.append(
                f"citation not in brief: {url!r} is not among the brief's "
                f"{len(allowed)} citation(s) — remove it or cite a sourced URL"
            )
    return reasons


# NOTE: reserved for future use. The current tool layer catches all
# load/parse exceptions inline and surfaces them as `status=error` summaries
# rather than raising a typed exception into the agent loop; a future refactor
# toward explicit exception-handling call sites can re-introduce a typed
# exception here. Left intentionally NOT exported via ``__all__`` until it has a
# raiser so the contract doesn't lie about current behavior.
class QualityGateError(Exception):
    """Reserved."""


__all__ = [
    "QualityResult",
    "check_proposal",
    "citation_fidelity_reasons",
    "count_words",
    "has_hype",
    "normalize_hashtags",
]