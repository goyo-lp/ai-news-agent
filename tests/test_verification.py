import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from ai_news_agent.config import Settings
from ai_news_agent.schemas import (
    Article,
    ArticleText,
    Evidence,
    EvidenceKind,
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
from ai_news_agent.summaries import SummaryDraft
from ai_news_agent.verification import (
    ModelVerdict,
    OpenRouterVerifier,
    VerificationConfigurationError,
    VerificationError,
    VerificationOutputError,
    build_verifier,
    contenders_note_for,
    deterministic_findings,
    evidence_text_for,
    parse_model_verdict,
    run_verification,
    verified_summaries_for_delivery,
    verifier_prompt,
    verify_one,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
HASH = "ab" * 32
BODY = "Example Lab released Model A with a 20 percent gain. " * 30


def make_article(article_id: str, hours_before: float = 1) -> Article:
    published = NOW - timedelta(hours=hours_before)
    url = f"https://example.com/ai/{article_id}"
    return Article(
        id=article_id,
        source_id="source-01",
        url=url,
        canonical_url=url,
        title=f"Story {article_id}",
        published_at=published,
        first_seen_at=published,
    )


def make_cluster(cluster_id: str, article_id: str) -> StoryCluster:
    return StoryCluster(
        id=cluster_id,
        article_ids=(article_id,),
        representative_article_id=article_id,
        rationale="seed",
        created_at=NOW,
    )


def make_summary(cluster_id: str, article_id: str, **overrides) -> Summary:
    payload = {
        "id": f"{cluster_id}-summary",
        "story_cluster_id": cluster_id,
        "representative_article_id": article_id,
        "sentences": (
            "Example Lab released Model A today.",
            "The report notes broad availability this week.",
            "Builders may benefit once replication lands.",
        ),
        "evidence_by_sentence": (
            (f"{article_id}-text",),
            (f"{article_id}-text",),
            (f"{article_id}-text",),
        ),
        "status": SummaryStatus.DRAFT,
    }
    payload.update(overrides)
    return Summary(**payload)


def make_score(cluster_id: str, total: float = 80.0) -> Score:
    _ = total
    return Score(
        id=f"{cluster_id}-editorial",
        story_cluster_id=cluster_id,
        relevance=4,
        impact=4,
        novelty=4,
        evidence=4,
        timeliness=4,
        evidence_ids=("e1",),
        rationale="Strong.",
    )


def make_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "verify.db")
    store.migrate()
    return store


def save_evidence(store: SQLiteStore, article_id: str, text: str = BODY) -> None:
    store.save(
        ArticleText(
            id=f"{article_id}-text",
            article_id=article_id,
            text=text,
            content_hash=HASH,
            fetched_at=NOW,
        )
    )
    store.save(
        Evidence(
            id=f"{article_id}-article-text",
            article_id=article_id,
            kind=EvidenceKind.ARTICLE_TEXT,
            source_url=f"https://example.com/ai/{article_id}",
            locator="article-body",
            excerpt=text[:800],
            content_hash=HASH,
            captured_at=NOW,
        )
    )


def pass_verdict() -> ModelVerdict:
    return ModelVerdict(verdict=VerificationVerdict.PASS, findings=())


class FakeVerifier:
    def __init__(self, verdicts: list[ModelVerdict] | None = None) -> None:
        self.verdicts = list(verdicts) if verdicts else [pass_verdict()]
        self.calls: list[dict] = []

    @property
    def identity(self) -> str:
        return "fake-verifier-v1"

    def check(self, sentences, evidence_text, contenders_note) -> ModelVerdict:
        self.calls.append({"sentences": tuple(sentences)})
        if len(self.verdicts) > 1:
            return self.verdicts.pop(0)
        return self.verdicts[0]


class FakeWriter:
    def __init__(self, draft: SummaryDraft | None = None) -> None:
        self.draft = draft or SummaryDraft(
            sentences=(
                "Example Lab released Model A today.",
                "The report notes broad availability this week.",
                "Builders may benefit once replication lands.",
            ),
            evidence_by_sentence=(("a1-text",), ("a1-text",), ("a1-text",)),
        )
        self.calls = 0

    @property
    def identity(self) -> str:
        return "fake-writer-v1"

    def write(self, title, article_text, evidence_excerpts) -> SummaryDraft:
        self.calls += 1
        return self.draft


def seed(
    store: SQLiteStore, cluster_id: str = "c1", article_id: str = "a1", **summary_kwargs
) -> tuple[StoryCluster, Article, Summary]:
    article = make_article(article_id)
    cluster = make_cluster(cluster_id, article_id)
    summary = make_summary(cluster_id, article_id, **summary_kwargs)
    store.save(article)
    store.save(cluster)
    save_evidence(store, article_id)
    store.save(summary)
    return cluster, article, summary


def test_prompt_covers_policy_checks() -> None:
    prompt = verifier_prompt(["S1.", "S2.", "S3."], "evidence", "contenders")

    assert "grounding" in prompt
    assert "S1." in prompt


def test_parse_verdict_accepts_findings() -> None:
    raw = json.dumps(
        {
            "verdict": "revise",
            "findings": [
                {
                    "severity": "error",
                    "message": "number mismatch",
                    "claim": "20%",
                    "evidence_ids": ["e1"],
                }
            ],
        }
    )
    parsed = parse_model_verdict(raw)

    assert parsed.verdict is VerificationVerdict.REVISE
    assert parsed.findings[0].severity is FindingSeverity.ERROR
    assert (
        parse_model_verdict(f"```json\n{raw}\n```").verdict
        is VerificationVerdict.REVISE
    )


def test_parse_verdict_rejects_bad_shapes() -> None:
    for bad in (
        "no json",
        json.dumps({"verdict": "maybe", "findings": []}),
        json.dumps(
            {"verdict": "pass", "findings": [{"severity": "bogus", "message": "x"}]}
        ),
        json.dumps({"verdict": "pass", "findings": "nope"}),
        "[1,2]",
    ):
        with pytest.raises(VerificationOutputError):
            parse_model_verdict(bad)


def test_deterministic_catches_wrong_numbers(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        _, article, _summary = seed(store)
        bad = make_summary(
            "c1",
            "a1",
            sentences=(
                "Example Lab released Model A today.",
                "The report shows a 99% gain on Benchmark Z.",
                "Builders may benefit once replication lands.",
            ),
        )
        findings = deterministic_findings(bad, article, BODY, set(), now=NOW)

    assert any("99" in item.message for item in findings)


def test_deterministic_catches_stale_and_duplicates(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        old_article = make_article("old", hours_before=48)
        old_summary = make_summary("c-old", "old")
        findings = deterministic_findings(
            old_summary, old_article, BODY, set(), now=NOW
        )
        assert any("stale" in item.message for item in findings)

        _, article, summary = seed(store)
        seen: set[str] = set()
        assert not [
            f
            for f in deterministic_findings(summary, article, BODY, seen, now=NOW)
            if "duplicate" in f.message
        ]
        again = deterministic_findings(summary, article, BODY, seen, now=NOW)
        assert any("duplicate" in item.message for item in again)

        missing = deterministic_findings(summary, None, BODY, set(), now=NOW)
        assert "missing" in missing[0].message


def test_evidence_text_collects_cited_records(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        _, _, summary = seed(store)
        text = evidence_text_for(summary, {"a1": make_article("a1")}, store)

    assert "Example Lab" in text


def test_contenders_note_lists_reserves() -> None:
    scores = {"c2": make_score("c2")}
    assert "c2" in contenders_note_for("c1", scores, ("c2",))
    assert "no excluded" in contenders_note_for("c1", {}, ())


def test_verify_pass_marks_no_model_call_on_fatal(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        _, article, _ = seed(store)
        bad = make_summary(
            "c1",
            "a1",
            sentences=(
                "Example Lab released Model A today.",
                "The report shows a 99% gain on Benchmark Z.",
                "Builders may benefit once replication lands.",
            ),
        )
        store.save(bad)
        verifier = FakeVerifier()
        result = verify_one(
            bad, {"a1": article}, store, now=NOW, verifier=verifier, attempt=1
        )

    assert result.verdict is VerificationVerdict.REJECT
    assert verifier.calls == []
    assert any("99" in item.message for item in result.findings)


def test_verifier_failures_become_findings(tmp_path: Path) -> None:
    class _Broken:
        @property
        def identity(self) -> str:
            return "broken"

        def check(self, sentences, evidence_text, contenders_note):
            raise VerificationOutputError("no JSON")

    class _Down:
        @property
        def identity(self) -> str:
            return "down"

        def check(self, sentences, evidence_text, contenders_note):
            raise VerificationError("timeout")

    with make_store(tmp_path) as store:
        _, article, summary = seed(store)
        broken = verify_one(
            summary, {"a1": article}, store, now=NOW, verifier=_Broken(), attempt=1
        )
        assert broken.verdict is VerificationVerdict.REJECT
        assert any("unusable" in item.message for item in broken.findings)
    with make_store(tmp_path) as store:
        _, article, summary = seed(store)
        down = verify_one(
            summary, {"a1": article}, store, now=NOW, verifier=_Down(), attempt=1
        )
        assert any("unavailable" in item.message for item in down.findings)


def test_configuration_errors_fail_fast(tmp_path: Path) -> None:
    from ai_news_agent.verification import VerificationConfigurationError as ConfigError

    class _Misconfigured:
        @property
        def identity(self) -> str:
            return "misconfigured"

        def check(self, sentences, evidence_text, contenders_note):
            raise ConfigError("no key")

    with make_store(tmp_path) as store:
        _, article, summary = seed(store)
        with pytest.raises(ConfigError):
            verify_one(
                summary,
                {"a1": article},
                store,
                now=NOW,
                verifier=_Misconfigured(),
                attempt=1,
            )


def test_build_verifier_requires_credentials() -> None:
    with pytest.raises(VerificationConfigurationError, match="OPENROUTER_API_KEY"):
        build_verifier(Settings(_env_file=None))


def test_run_pass_marks_verified_for_delivery(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        cluster, article, _ = seed(store)
        summary = run_verification(
            ("c1",),
            {"c1": cluster},
            {"a1": article},
            store,
            now=NOW,
            verifier=FakeVerifier(),
            writer=FakeWriter(),
        )

        assert summary.verified_ids == ("c1",)
        assert summary.meets_minimum is False
        assert summary.shortfall == 0
        delivered = verified_summaries_for_delivery(summary.verified_ids, store)
        assert len(delivered) == 1 and delivered[0].status is SummaryStatus.VERIFIED


def test_revise_repairs_and_reverifies(tmp_path: Path) -> None:
    revise = ModelVerdict(
        verdict=VerificationVerdict.REVISE,
        findings=(
            VerificationFinding(
                severity=FindingSeverity.WARNING, message="tighten attribution"
            ),
        ),
    )
    with make_store(tmp_path) as store:
        cluster, article, _ = seed(store)
        writer = FakeWriter()
        summary = run_verification(
            ("c1",),
            {"c1": cluster},
            {"a1": article},
            store,
            now=NOW,
            verifier=FakeVerifier([revise, pass_verdict()]),
            writer=writer,
            max_attempts=2,
        )

        assert summary.verified_ids == ("c1",)
        assert writer.calls == 1
        assert store.get(VerificationResult, "c1-summary-v1") is not None
        assert store.get(VerificationResult, "c1-summary-v2") is not None


def test_replace_swaps_reserve_candidate(tmp_path: Path) -> None:
    replace = ModelVerdict(
        verdict=VerificationVerdict.REPLACE,
        findings=(
            VerificationFinding(
                severity=FindingSeverity.WARNING, message="contender stronger"
            ),
        ),
    )
    with make_store(tmp_path) as store:
        c1, a1, _ = seed(store, "c1", "a1")
        a2 = make_article("a2")
        c2 = make_cluster("c2", "a2")
        store.save(a2)
        store.save(c2)
        save_evidence(store, "a2")
        store.save(make_summary("c2", "a2"))
        summary = run_verification(
            ("c1",),
            {"c1": c1, "c2": c2},
            {"a1": a1, "a2": a2},
            store,
            now=NOW,
            verifier=FakeVerifier([replace, pass_verdict(), pass_verdict()]),
            writer=FakeWriter(),
            scores_by_cluster={"c1": make_score("c1"), "c2": make_score("c2")},
            reserve_ids=("c2",),
        )

        assert "c2" in summary.verified_ids
        assert summary.replaced == 1


def test_retries_terminate_with_shortfall(tmp_path: Path) -> None:
    revise = ModelVerdict(
        verdict=VerificationVerdict.REVISE,
        findings=(
            VerificationFinding(severity=FindingSeverity.WARNING, message="again"),
        ),
    )
    with make_store(tmp_path) as store:
        cluster, article, _ = seed(store)
        summary = run_verification(
            ("c1",),
            {"c1": cluster},
            {"a1": article},
            store,
            now=NOW,
            verifier=FakeVerifier([revise, revise]),
            writer=FakeWriter(),
            max_attempts=2,
        )

        assert summary.verified_ids == ()
        assert summary.shortfall == 1
        assert verified_summaries_for_delivery(summary.verified_ids, store) == ()


def test_resumed_run_does_not_duplicate_artifact(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        cluster, article, _ = seed(store)
        kwargs = {
            "clusters_by_id": {"c1": cluster},
            "articles_by_id": {"a1": article},
            "now": NOW,
            "verifier": FakeVerifier(),
            "writer": FakeWriter(),
            "digest_run_id": "run-resume-1",
        }
        first = run_verification(("c1",), store=store, **kwargs)
        second = run_verification(("c1",), store=store, **kwargs)

        assert first.verified_ids == second.verified_ids == ("c1",)
        assert sum(1 for _ in store.iter_records(Summary)) == 1


def test_verify_command_runs_graph(tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    import ai_news_agent.cli as cli
    from ai_news_agent.cli import app

    runner = CliRunner()
    database = tmp_path / "verify.db"
    fresh_now = datetime.now(UTC)
    fresh_article = Article(
        id="a1",
        source_id="source-01",
        url="https://example.com/ai/a1",
        canonical_url="https://example.com/ai/a1",
        title="Story a1",
        published_at=fresh_now,
        first_seen_at=fresh_now,
    )
    fresh_cluster = StoryCluster(
        id="c1",
        article_ids=("a1",),
        representative_article_id="a1",
        rationale="seed",
        created_at=fresh_now,
    )
    with SQLiteStore(database) as store:
        store.migrate()
        store.save(fresh_article)
        store.save(fresh_cluster)
        save_evidence(store, "a1")
        store.save(make_summary("c1", "a1"))

    monkeypatch_verifier = FakeVerifier()
    monkeypatch_writer = FakeWriter()
    original_verifier = cli.build_verifier
    original_writer = cli.build_summary_writer
    cli.build_verifier = lambda settings, timeout=30.0: monkeypatch_verifier
    cli.build_summary_writer = lambda settings, timeout=30.0: monkeypatch_writer
    try:
        result = runner.invoke(app, ["verify", "--database", str(database)])
    finally:
        cli.build_verifier = original_verifier
        cli.build_summary_writer = original_writer

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["verified"] == 1
    assert payload["verified_ids"] == ["c1"]


class _FakePostResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _verdict_payload(**kwargs) -> dict:
    verdict = {"verdict": "pass", "findings": []}
    verdict.update(kwargs)
    return {"choices": [{"message": {"content": json.dumps(verdict)}}]}


def _verifier_client() -> OpenRouterVerifier:
    return OpenRouterVerifier(api_key="test-key", model="test-model")


def test_openrouter_verifier_posts_structured_request(monkeypatch) -> None:
    seen = {}

    def _post(url, *, headers, json, timeout):
        seen.update({"url": url, "headers": headers, "json": json})
        return _FakePostResponse(payload=_verdict_payload())

    monkeypatch.setattr(httpx, "post", _post)
    checked = _verifier_client().check(["S1.", "S2.", "S3."], "evidence", "none")

    assert checked.verdict is VerificationVerdict.PASS
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert _verifier_client().identity == "test-model"


def test_openrouter_verifier_maps_failures(monkeypatch) -> None:
    verifier = _verifier_client()
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=401)
    )
    with pytest.raises(VerificationConfigurationError, match="credentials rejected"):
        verifier.check(["S."], "e", "c")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=503)
    )
    with pytest.raises(VerificationError, match="retryable"):
        verifier.check(["S."], "e", "c")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=404)
    )
    with pytest.raises(VerificationError, match="not retryable"):
        verifier.check(["S."], "e", "c")

    def _timeout(*a, **k):
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(httpx, "post", _timeout)
    with pytest.raises(VerificationError, match="timeout"):
        verifier.check(["S."], "e", "c")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(payload={"nope": 1})
    )
    with pytest.raises(VerificationOutputError, match="envelope"):
        verifier.check(["S."], "e", "c")

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakePostResponse(
            payload={"choices": [{"message": {"content": "  "}}]}
        ),
    )
    with pytest.raises(VerificationOutputError, match="empty content"):
        verifier.check(["S."], "e", "c")
