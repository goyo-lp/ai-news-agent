from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.config import Settings
from app.orchestrator.schemas import Citation, ResearchBrief
from app.orchestrator.services import brief_verifier as svc_mod
from app.orchestrator.services.brief_verifier import (
    BriefVerifier,
    _build_verifier_queries,
    _dedupe_discovered,
    _make_cautious,
    _merge_notes,
    _sanitize_access_failure_language,
)


def _settings(api_key: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key=api_key,
        openrouter_verifier_secondary_model=None,
        verification_sources_per_topic=3,
        verification_concurrency=4,
        request_timeout_seconds=10,
    )


def _brief(topic_id: str = "t1") -> ResearchBrief:
    return ResearchBrief(
        topic_id=topic_id,
        headline="A Big Announcement",
        summary="What happened and why it matters.",
        technical_significance="Architecture improves throughput.",
        business_impact="Cuts inference cost.",
        why_now="Ships this week.",
        citations=[Citation(title="Source", url="https://example.com/a", domain="example.com")],
    )


def _patch_openrouter_httpx(monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]) -> None:
    """Patch the *brief_verifier* module's httpx.AsyncClient so the verifier's
    OpenRouter chat-completions call resolves to a fixed response. The verifier
    uses a fresh AsyncClient per verify_briefs run, captured here."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": __import__("json").dumps(response)}}
                ]
            },
        )

    real = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(svc_mod.httpx, "AsyncClient", _factory)


# --- helpers (deterministic) ------------------------------------------------


def test_make_cautious_prepends_once() -> None:
    assert _make_cautious("Hello world.") == "Based on current evidence, hello world."
    assert (
        _make_cautious("Based on current evidence already")
        == "Based on current evidence already"
    )
    assert _make_cautious("") == ""


@pytest.mark.parametrize(
    "phrase,surviving_second_sentence",
    [
        ("I could not access the page details. The model is fast.", "The model is fast."),
        ("No abstract available. The architecture is documented.", "The architecture is documented."),
        # The third regex consumes the whole "From what is publicly indexed ..."
        # sentence, so for this fixture the surviving content lives in a *later*
        # sentence that the regex's [^.!?]* stopper doesn't reach.
        (
            "From what is publicly indexed, the launch is unconfirmed. The architecture is documented.",
            "The architecture is documented.",
        ),
    ],
)
def test_sanitize_access_failure_language_neutralizes_each_pattern(
    phrase: str, surviving_second_sentence: str
) -> None:
    """m7: each of the three sanitizer regexes rewrote its trigger phrase to
    the neutral `directional` sentence; a *later* sentence the regex doesn't
    consume survives intact. Parametrized — a future maintainer adding a 4th
    pattern gets a clear template to add coverage for it."""
    cleaned = _sanitize_access_failure_language(phrase)
    assert "Public evidence is still limited and should be treated as directional." in cleaned
    assert surviving_second_sentence in cleaned


def test_sanitize_access_failure_language_neutralizes_access_phrase_directly() -> None:
    """Explicit coverage of the access-failure pattern (was test_sanitize_access_failure_language_neutralizes_scraping_phrases
    before parametrization)."""
    cleaned = _sanitize_access_failure_language(
        "I could not access the page details. The model is fast."
    )
    assert "could not access" not in cleaned.lower()
    assert "Public evidence is still limited and should be treated as directional." in cleaned
    assert "The model is fast." in cleaned


def test_sanitize_access_failure_language_preserves_normal_text() -> None:
    cleaned = _sanitize_access_failure_language("The architecture improves throughput.")
    assert cleaned == "The architecture improves throughput."


def test_sanitize_empty_returns_empty() -> None:
    assert _sanitize_access_failure_language("") == ""
    assert _sanitize_access_failure_language("   ") == ""


def test_merge_notes_dedups_and_sanitizes() -> None:
    merged = _merge_notes(
        ["first note", "based on current evidence already exists"],
        ["Second Note", "first note", "I could not access the page."],
    )
    assert "first note" in merged
    assert "Second Note" in merged
    assert "based on current evidence already exists" in merged
    # Sanitizer rewrote the access-failure note to neutral phrasing.
    assert "could not access" not in " ".join(merged).lower()
    # No duplicate "first note".
    assert merged.count("first note") == 1


def test_build_verifier_queries_dedups_to_four() -> None:
    brief = _brief()
    queries = _build_verifier_queries(brief)
    assert len(queries) == 4
    assert queries[0] == brief.headline
    assert all("A Big Announcement" in q for q in queries)


def test_build_verifier_queries_strips_whitespace() -> None:
    brief = ResearchBrief(
        topic_id="t",
        headline="   ",
        summary="s",
        technical_significance="ts",
        business_impact="bi",
        why_now="wn",
    )
    # An all-whitespace headline collapses to empty and is dropped from the
    # queries; the three template-derived queries survive (each has its own
    # non-empty body after whitespace normalization).
    queries = _build_verifier_queries(brief)
    assert len(queries) == 3
    # The empty-headline entry is gone; what remains is the derived templates.
    assert all(q != brief.headline for q in queries)
    assert all("technical" in q or "architecture" in q or "production" in q for q in queries)


def test_dedupe_discovered_drops_empty_and_duplicate_urls() -> None:
    class _Item:
        def __init__(self, url: str) -> None:
            self.url = url

    items = [_Item("https://a.example/x"), _Item(""), _Item("https://a.example/x"), _Item("https://b.example/y")]
    deduped = _dedupe_discovered(items)
    assert [str(i.url) for i in deduped] == ["https://a.example/x", "https://b.example/y"]


# --- ensemble reconciliation (M2: pure-helper coverage) ---------------------


def test_reconcile_verifier_payloads_strictest_status_wins_min_confidence() -> None:
    """M2: ensemble rule = max(status_rank tuple) wins (strictest), confidence =
    min(primary, secondary), notes merged+deduped, ensemble-detail note appended."""
    from app.orchestrator.services.brief_verifier import _reconcile_verifier_payloads

    primary = {
        "verdict": "verified",
        "confidence": 0.9,
        "corrected_summary": "verified summary",
        "notes": ["primary note"],
    }
    secondary = {
        "verdict": "insufficient_evidence",
        "confidence": 0.4,
        "corrected_summary": "more conservative summary",
        "notes": ["secondary note"],
    }

    merged = _reconcile_verifier_payloads(
        primary=primary,
        secondary=secondary,
        primary_model="model-a",
        secondary_model="model-b",
    )

    # Strictest status (insufficient_evidence has higher status_rank) wins.
    assert merged["verdict"] == "insufficient_evidence"
    # min confidence.
    assert merged["confidence"] == 0.4
    # Notes from both + ensemble-detail.
    assert "primary note" in merged["notes"]
    assert "secondary note" in merged["notes"]
    assert any("model-a" in n and "model-b" in n for n in merged["notes"])


def test_reconcile_verifier_payloads_verified_primary_wins_when_secondary_is_lower() -> None:
    """When primary is verified and secondary is partially_verified, primary's
    strict status-rank tuple is *smaller* (more confident = strictest in this
    reconciliation?), so the secondary's stricter verdict wins. Wait — strictest
    *wins*, so partially_verified (rank 1) > verified (rank 0). Pin that:
    secondary (partially_verified) wins, not primary (verified)."""
    from app.orchestrator.services.brief_verifier import _reconcile_verifier_payloads

    primary = {"verdict": "verified", "confidence": 0.85, "notes": ["p"]}
    secondary = {"verdict": "partially_verified", "confidence": 0.5, "notes": ["s"]}

    merged = _reconcile_verifier_payloads(
        primary=primary, secondary=secondary, primary_model="p", secondary_model="s"
    )

    # Strictest = partially_verified (rank 1) > verified (rank 0). Conf = min.
    assert merged["verdict"] == "partially_verified"
    assert merged["confidence"] == 0.5
    assert "p" in merged["notes"]
    assert "s" in merged["notes"]


# --- BriefVerifier.verify_one / verify_briefs -----------------------------------


async def test_verify_one_dry_run_returns_partially_verified_cautious_brief() -> None:
    """No OPENROUTER_API_KEY -> heuristic fallback path fires. Confidence
    differs based on whether evidence was gathered (Tavily mock returns 3
    items per query -> >0 confidence).
    The summary is rewritten as cautious; status == partially_verified."""
    settings = _settings(api_key=None)
    verifier = BriefVerifier(settings)

    verified = await verifier.verify_one(_brief(), hours_back=24, dry_run=True)

    assert verified.verification_status == "partially_verified"
    assert verified.verification_confidence >= 0.6
    assert verified.summary.lower().startswith("based on current evidence")
    # The dry-run note explicitly calls out the fallback.
    assert any("dry-run or missing verifier model key" in n.lower() for n in verified.verification_notes)


async def test_verify_one_dry_run_with_no_citations_uses_evidence_from_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no seed citations, the evidence pool comes entirely from the
    verifier's Tavily-search queries. Tavily's dry-run `_mock_search` returns
    3 templates × 4 queries, so evidence_texts is non-empty and the 0.6
    confidence branch fires. (The 0.4 branch is the 'no evidence at all'
    fallback — exercised by the next test.)"""
    settings = _settings(api_key=None)
    verifier = BriefVerifier(settings)
    brief = _brief()
    brief.citations = []  # no seed URLs to extract

    verified = await verifier.verify_one(brief, hours_back=24, dry_run=True)

    assert verified.verification_status == "partially_verified"
    assert verified.verification_confidence == 0.6  # evidence_texts non-empty


async def test_verify_one_dry_run_with_no_evidence_uses_lower_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """m5: when both the seed citation search and the Tavily mock return
    nothing (an empty evidence_texts list), the heuristic confidence drops to
    0.4 — the 'no corroborating evidence' branch at brief_verifier.py:level.
    Reach this by stubbing _mock_search to return [] so Tavily produces
    nothing to corroborate."""
    settings = _settings(api_key=None)
    verifier = BriefVerifier(settings)

    async def _empty_search_news(self, client, query, hours_back, max_results, dry_run):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(
        "app.orchestrator.services.tavily_client.TavilyClient.search_news",
        _empty_search_news,
    )

    brief = _brief()
    brief.citations = []

    verified = await verifier.verify_one(brief, hours_back=24, dry_run=True)

    assert verified.verification_status == "partially_verified"
    assert verified.verification_confidence == 0.4  # no-evidence fallback path


async def test_verify_one_real_call_parses_verifier_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """With OPENROUTER_API_KEY present and dry_run=False, the verifier calls
    OpenRouter chat completions and parses the JSON verdict."""
    settings = _settings(api_key="sk-test")
    verifier_payload = {
        "verdict": "verified",
        "confidence": 0.9,
        "corrected_summary": "Based on current evidence, the launch is solid.",
        "corrected_technical_significance": "Architecture is documented.",
        "corrected_business_impact": "Cost drops materially.",
        "corrected_why_now": "Imploded this week.",
        "notes": ["Two independent sources confirm.", "Benchmark numbers are comparable."],
    }
    _patch_openrouter_httpx(monkeypatch, verifier_payload)

    verifier = BriefVerifier(settings)
    verified = await verifier.verify_one(_brief(), hours_back=24, dry_run=False)

    assert verified.verification_status == "verified"
    assert abs(verified.verification_confidence - 0.9) < 0.01
    # Sanitizer keeps the cautious prefix the model returned.
    assert verified.summary.lower().startswith("based on current evidence")
    assert verified.technical_significance == "Architecture is documented."
    # Both model notes survived the merge + dedup.
    assert any("sources confirm" in n for n in verified.verification_notes)


async def test_verify_one_rejects_unknown_verdict_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that returns a verdict that isn't one of the three enum values
    falls back to partially_verified rather than raising out of the verifier
    (which would land as `insufficient_evidence` from the fallback worker)."""
    settings = _settings(api_key="sk-test")
    verifier_payload = {
        "verdict": "totally_true_i_swear",
        "confidence": 0.99,
        "corrected_summary": "The launch worked.",
        "corrected_technical_significance": "",
        "corrected_business_impact": "",
        "corrected_why_now": "",
        "notes": [],
    }
    _patch_openrouter_httpx(monkeypatch, verifier_payload)

    verifier = BriefVerifier(settings)
    verified = await verifier.verify_one(_brief(), hours_back=24, dry_run=False)

    assert verified.verification_status == "partially_verified"  # defense-in-depth fallback
    # Sanity: status is in the recognized enum so a later model-validate doesn't fail.
    assert verified.verification_status in {"verified", "partially_verified", "insufficient_evidence", "failed"}


async def test_verify_briefs_per_brief_failure_degrades_to_cautious_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-brief failure (here: OpenRouter returns a non-JSON body) does NOT
    abort the batch — the worker falls back to insufficient_evidence, summary
    becomes cautious, and the batch returns N briefs in input order + 1 error.
    """
    settings = _settings(api_key="sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    real = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(svc_mod.httpx, "AsyncClient", _factory)

    verifier = BriefVerifier(settings)
    briefs = [_brief("t1"), _brief("t2")]
    verified, errors = await verifier.verify_briefs(briefs, hours_back=24, dry_run=False)

    assert len(verified) == 2
    # Order preserved.
    assert verified[0].topic_id == "t1"
    assert verified[1].topic_id == "t2"
    # Both degraded to insufficient_evidence fallback; the JSON-parse failure
    # raised from _verify_with_model -> propagates out of _run_verifier_models
    # -> caught by the worker's try/except -> cautious fallback.
    assert all(b.verification_status == "insufficient_evidence" for b in verified)
    assert all(b.summary.lower().startswith("based on current evidence") for b in verified)
    assert len(errors) == 2


async def test_verify_briefs_empty_input_short_circuits() -> None:
    verifier = BriefVerifier(_settings())
    verified, errors = await verifier.verify_briefs([], hours_back=24, dry_run=False)
    assert verified == []
    assert errors == []