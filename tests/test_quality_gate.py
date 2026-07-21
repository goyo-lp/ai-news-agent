from __future__ import annotations

from app.orchestrator.schemas import PostProposal
from app.orchestrator.services.quality_gate import (
    QualityResult,
    check_proposal,
    count_words,
    has_hype,
    normalize_hashtags,
)


def _words(n: int, *, theme: str = "the writer keeps watching for clearer results from this early signal which matters across the next several weeks for adopters and observers alike") -> str:
    """Build an exactly-N-word body in a register that AVOIDS the jargon-
    replacement keys (``implementation``, ``throughput``, ``architecture`` …)
    so the cleaner's `_simplify_jargon` step can't change the body's word
    count. The other cleaner steps (`_strip_salesy_language`,
    `_strip_access_failure_language`) are no-ops on this theme too. That keeps
    `result.word_count == n` deterministically — important for the in-window
    and out-of-window assertions."""
    base = theme.split()
    if len(base) >= n:
        return " ".join(base[:n])
    reps = (n + len(base) - 1) // len(base)
    full = " ".join(base * reps)
    return " ".join(full.split()[:n])


def _proposal(
    *,
    body: str = "",
    topic_ids: list[str] | None = None,
    hashtags: list[str] | None = None,
    post_id: str = "post-1",
) -> PostProposal:
    return PostProposal(
        post_id=post_id,
        angle="reflection",
        headline="A technical note on: AI architecture",
        body=body,
        hashtags=hashtags if hashtags is not None else ["#AI", "#AIAgents", "#MachineLearning"],
        supporting_topic_ids=topic_ids if topic_ids is not None else ["t1"],
        citation_urls=["https://example.com/a"],
    )


# --- pure helpers -----------------------------------------------------------


def test_count_words_simple() -> None:
    assert count_words("") == 0
    assert count_words("one two three") == 3
    assert count_words("   spaced   words   ") == 2


def test_has_hype_finds_each_marker_case_insensitively() -> None:
    for marker in ("game changer", "Revolutionary", "MUST-HAVE", "10X", "Act Now"):
        present, found = has_hype(f"{marker} in a sentence")
        assert present
        assert any(m in marker.lower() or marker.lower() in m for m in found)


def test_has_hype_clean_text_returns_empty_markers() -> None:
    present, found = has_hype("the architecture improves throughput materially")
    assert not present
    assert found == []


def test_normalize_hashtags_strips_whitespace_and_dedupes() -> None:
    normalized = normalize_hashtags(["  AI  ", "AIAgents", "#AI", "", "  ", "#MachineLearning"])
    assert normalized == ["#AI", "#AIAgents", "#MachineLearning"]


# --- check_proposal verdict -------------------------------------------------


def test_check_proposal_passes_clean_in_window_single_topic_three_hashtags() -> None:
    body = _words(120)  # in [105, 182], several tech markers, no hype
    result = check_proposal(_proposal(body=body))
    assert isinstance(result, QualityResult)
    assert result.passed
    assert 105 <= result.word_count <= 182
    assert result.single_topic
    assert not result.has_hype
    assert result.reasons == []


def test_check_proposal_fails_too_short() -> None:
    body = _words(50)  # way under the 105 floor
    result = check_proposal(_proposal(body=body))
    assert not result.passed
    assert any("too short" in r for r in result.reasons)
    assert result.word_count == 50


def test_check_proposal_fails_too_long() -> None:
    body = _words(250)  # over the 182 cap
    result = check_proposal(_proposal(body=body))
    assert not result.passed
    assert any("too long" in r for r in result.reasons)


def test_check_proposal_fails_multiple_supporting_topic_ids() -> None:
    body = _words(120)
    result = check_proposal(_proposal(body=body, topic_ids=["t1", "t2"]))
    assert not result.passed
    assert any("exactly one supporting_topic_id" in r for r in result.reasons)
    assert not result.single_topic


def test_check_proposal_fails_missing_topic_ids() -> None:
    body = _words(120)
    result = check_proposal(_proposal(body=body, topic_ids=[]))
    assert not result.passed
    assert any("exactly one supporting_topic_id" in r for r in result.reasons)


def test_check_proposal_fails_hype_with_marker_names() -> None:
    """Detection runs on the **raw** body — so it catches the markers
    ``_strip_salesy_language`` would have rewritten before counting
    (`revolutionary` → `meaningful`, `game changer` → `notable shift`). The
    body mixes hype with non-jargon prose to stay in window after cleaning so
    only the hype-as-hype reason fires; both rewritten-by-cleaner and never-
    rewritten marker flavors are caught."""
    body = (
        "This announcement is a revolutionary game changer that is an "
        "unbelievable must-have unlock for sure. "
        + _words(150)  # pad above 105 with non-hype, non-jargon prose
    )
    result = check_proposal(_proposal(body=body))
    assert not result.passed
    assert result.has_hype
    assert "revolutionary" in result.hype_markers
    assert "game changer" in result.hype_markers
    assert "unbelievable" in result.hype_markers
    assert "must-have" in result.hype_markers
    assert "unlock" in result.hype_markers
    assert any("hype language detected" in r for r in result.reasons)


def test_check_proposal_cleaned_count_cascade_can_blow_past_window() -> None:
    """m1: a raw draft in [105, 182] words that the cleaner pushes over 182
    (because jargon replacements balloon word count: `throughput` → 6 words,
    `multimodal` → 6 words, etc.) fails `too long`. The documented guarantee
    is 'count the cleaned body, not the raw'. Pin it."""
    # Build a raw body of exactly 130 words, each a single-word jargon key that
    # expands to multiple words under `_simplify_jargon_for_general_audience`.
    # 130 raw → ~3-4 expanded words per key on average → ~400+ cleaned words.
    jargon_keys = [
        "implementation", "orchestration", "inference", "latency", "throughput",
        "benchmark", "multimodal", "retrieval", "architecture", "evaluation",
    ]
    body = " ".join((jargon_keys * 13)[:130])  # 130 list items, each 1 word
    # Sanity: raw count is exactly 130 (in-window).
    assert len(body.split()) == 130

    result = check_proposal(_proposal(body=body))
    assert not result.passed
    assert any("too long" in r for r in result.reasons)
    # The cleaned body literally contains the expanded replacements (sanity).
    assert "how much work it can handle" in result.cleaned_body


def test_check_proposal_fails_too_few_hashtags() -> None:
    body = _words(120)
    result = check_proposal(_proposal(body=body, hashtags=["#AI"]))
    assert not result.passed
    assert any("at least 3" in r for r in result.reasons)
    assert result.hashtags_count == 1


def test_check_proposal_fails_no_usable_hashtags() -> None:
    body = _words(120)
    result = check_proposal(_proposal(body=body, hashtags=["", "   "]))
    assert not result.passed
    assert any("no usable hashtags" in r for r in result.reasons)
    assert result.hashtags_count == 0


def test_check_proposal_aggregates_multiple_reasons() -> None:
    """A draft failing several checks reports each in `reasons` — order isn't
    pinned (per-check ordering is incidental), but each failing check is
    represented so the writer knows every fix to make."""
    body = _words(50)  # too short
    result = check_proposal(
        _proposal(body=body, topic_ids=["t1", "t2"], hashtags=["#AI"])
    )
    assert not result.passed
    # Three distinct failing checks represented.
    assert any("too short" in r for r in result.reasons)
    assert any("exactly one supporting_topic_id" in r for r in result.reasons)
    assert any("at least 3" in r for r in result.reasons)
    assert len(result.reasons) >= 3


def test_check_proposal_cleaned_body_runs_through_jargon_simplifier() -> None:
    """The cleaned body should have 'implementation' replaced with
    'real-world use' etc., so the word count reported is post-cleanup —
    matching what a reader sees."""
    body = (
        "The model's implementation improves inference latency and context window "
        "size for retrieval. " * 12  # ~120 words of jargon-laden text
    )
    result = check_proposal(_proposal(body=body))
    assert "implementation" not in result.cleaned_body
    assert "inference" not in result.cleaned_body
    assert "context window" not in result.cleaned_body
    assert "real-world use" in result.cleaned_body
    assert "model use" in result.cleaned_body
    assert "working memory" in result.cleaned_body


def test_check_proposal_strips_salesy_language_from_cleaned_body() -> None:
    body = " ".join(
        [
            "This model is a game changer.",
            "It's a revolutionary must-have unlock for sure.",
            "What do you think?",
        ]
        + ["the architecture is solid for production use."] * 30
    )
    result = check_proposal(_proposal(body=body))
    # The cleaner rewrites the salesy markers; the gate then detects hype —
    # wait no: hype detection runs on the *cleaned* body. So the rewritten
    # versions should NOT trip the hype detector (e.g. "notable shift" isn't a
    # marker). Verify that's the actual behavior: rewriting defuses the hype
    # detection. Important to pin so a maintainer knows not to flip the order.
    # -- but "unlock" -> "enable"; does the cleaned body still trigger hype?
    assert "game changer" not in result.cleaned_body.lower()
    # `unlock` is a hype marker and rewrites to `enable`; the cleaned body
    # shouldn't contain `unlock` anymore.
    assert "unlock" not in result.cleaned_body.lower()
    assert "enable" in result.cleaned_body.lower()


def test_check_proposal_on_empty_body() -> None:
    result = check_proposal(_proposal(body=""))
    assert not result.passed
    assert any("too short" in r for r in result.reasons)
    assert result.word_count == 0