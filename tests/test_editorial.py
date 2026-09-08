from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ai_news_agent.config import Settings
from ai_news_agent.editorial import (
    EditorialConfigurationError,
    EditorialError,
    EditorialOutputError,
    EditorialVerdict,
    OpenRouterEditorialJudge,
    build_editorial_agent,
    build_editorial_judge,
    check_digest_history,
    editorial_prompt,
    fetch_primary_evidence,
    inspect_related_coverage,
    make_editorial_tools,
    parse_editorial_verdict,
    rank_clusters,
    read_article_text,
)
from ai_news_agent.schemas import (
    Article,
    ArticleText,
    Evidence,
    EvidenceKind,
    Score,
    StoryCluster,
)
from ai_news_agent.storage import SQLiteStore

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
HASH = "ab" * 32


def make_article(article_id: str, title: str | None = None) -> Article:
    url = f"https://example.com/ai/{article_id}"
    return Article(
        id=article_id,
        source_id="source-01",
        url=url,
        canonical_url=url,
        title=title or f"Story {article_id}",
        published_at=NOW,
        first_seen_at=NOW,
    )


def make_cluster(cluster_id: str, *article_ids: str) -> StoryCluster:
    return StoryCluster(
        id=cluster_id,
        article_ids=tuple(article_ids),
        representative_article_id=article_ids[0],
        rationale="test cluster",
        created_at=NOW,
    )


def make_verdict(**overrides) -> EditorialVerdict:
    payload = {
        "relevance": 4,
        "impact": 4,
        "novelty": 3,
        "evidence": 4,
        "timeliness": 5,
        "evidence_ids": ("a1-article-text",),
        "uncertainty": 0.2,
        "rationale": "Consequential release with inspectable evidence.",
        "needs_more_evidence": False,
    }
    payload.update(overrides)
    return EditorialVerdict(**payload)


class FakeJudge:
    """Return scripted verdicts while recording investigation inputs."""

    def __init__(self, verdicts: list[EditorialVerdict] | None = None) -> None:
        self.verdicts = list(verdicts) if verdicts else [make_verdict()]
        self.calls: list[dict] = []

    @property
    def identity(self) -> str:
        return "fake-editorial-v1"

    def score(
        self, title: str, article_text: str, supporting: str, history_note: str
    ) -> EditorialVerdict:
        self.calls.append(
            {
                "title": title,
                "article_text": article_text,
                "supporting": supporting,
                "history": history_note,
            }
        )
        if len(self.verdicts) > 1:
            return self.verdicts.pop(0)
        return self.verdicts[0]


def make_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "editorial.db")
    store.migrate()
    return store


def save_text(store: SQLiteStore, article_id: str, text: str) -> None:
    store.save(
        ArticleText(
            id=f"{article_id}-text",
            article_id=article_id,
            text=text,
            content_hash=HASH,
            fetched_at=NOW,
        )
    )


def save_evidence(store: SQLiteStore, evidence_id: str, article_id: str) -> None:
    store.save(
        Evidence(
            id=evidence_id,
            article_id=article_id,
            kind=EvidenceKind.ARTICLE_TEXT,
            source_url=f"https://example.com/ai/{article_id}",
            locator="article-body",
            excerpt="Supporting excerpt with concrete numbers.",
            content_hash=HASH,
            captured_at=NOW,
        )
    )


def rank(
    clusters: tuple[StoryCluster, ...],
    articles: tuple[Article, ...],
    store: SQLiteStore,
    judge: FakeJudge,
    **kwargs,
):
    return rank_clusters(
        clusters,
        {article.id: article for article in articles},
        store,
        now=NOW,
        judge=judge,
        **kwargs,
    )


# Verdict parsing


def test_parse_verdict_accepts_strict_and_fenced_json() -> None:
    raw = (
        '{"relevance": 5, "impact": 4, "novelty": 4, "evidence": 5, '
        '"timeliness": 5, "evidence_ids": ["e1"], "uncertainty": 0.1, '
        '"rationale": "Strong release.", "needs_more_evidence": false}'
    )
    verdict = parse_editorial_verdict(raw)

    assert verdict.relevance == 5
    assert verdict.evidence_ids == ("e1",)
    assert verdict.needs_more_evidence is False
    fenced = parse_editorial_verdict(f"```json\n{raw}\n```")
    assert fenced.rationale == "Strong release."


def test_parse_verdict_rejects_malformed_output() -> None:
    for bad in (
        "no json here",
        '{"relevance": 5}',
        '{"relevance": 6, "impact": 1, "novelty": 1, "evidence": 1, '
        '"timeliness": 1, "evidence_ids": ["e"], "uncertainty": 0.1, '
        '"rationale": "x", "needs_more_evidence": false}',
        '{"relevance": 5, "impact": 4, "novelty": 4, "evidence": 5, '
        '"timeliness": 5, "evidence_ids": [], "uncertainty": 0.1, '
        '"rationale": "x", "needs_more_evidence": false}',
        '{"relevance": 5, "impact": 4, "novelty": 4, "evidence": 5, '
        '"timeliness": 5, "evidence_ids": ["e"], "uncertainty": 2.0, '
        '"rationale": "x", "needs_more_evidence": false}',
        '{"relevance": 5, "impact": 4, "novelty": 4, "evidence": 5, '
        '"timeliness": 5, "evidence_ids": ["e"], "uncertainty": 0.1, '
        '"rationale": "  ", "needs_more_evidence": false}',
        '{"relevance": "high", "impact": 1, "novelty": 1, "evidence": 1, '
        '"timeliness": 1, "evidence_ids": ["e"], "uncertainty": 0.1, '
        '"rationale": "x", "needs_more_evidence": false}',
        "[1, 2, 3]",
        "{oops",
    ):
        with pytest.raises(EditorialOutputError):
            parse_editorial_verdict(bad)


def test_prompt_separates_primary_from_investigation() -> None:
    prompt = editorial_prompt("Title", "Primary body", "Supporting notes", "new")

    assert "PRIMARY ARTICLE" in prompt
    assert "SUPPORTING COVERAGE" in prompt
    assert "Primary body" in prompt
    assert "Title" in prompt


# Evidence tools


def test_read_article_prefers_stored_text(tmp_path: Path) -> None:
    article = make_article("a1")
    with make_store(tmp_path) as store:
        assert read_article_text(make_cluster("c1", "a1"), {"a1": article}, store) == (
            "Story a1"
        )
        save_text(store, "a1", "Full representative body.")
        assert "Full representative" in read_article_text(
            make_cluster("c1", "a1"), {"a1": article}, store
        )
        assert read_article_text(make_cluster("c1", "ghost"), {}, store) == ""


def test_related_coverage_and_history_notes(tmp_path: Path) -> None:
    articles = (make_article("a1"), make_article("a2", "Second angle"))
    cluster = make_cluster("c1", "a1", "a2")
    with make_store(tmp_path) as store:
        related = inspect_related_coverage(
            cluster, {item.id: item for item in articles}, store
        )
        assert "Second angle" in related
        solo = inspect_related_coverage(
            make_cluster("c2", "a1"), {"a1": articles[0]}, store
        )
        assert "no additional" in solo
        assert "no digest history" in check_digest_history(cluster, frozenset())
        assert "new relative" in check_digest_history(cluster, frozenset({"other"}))
        assert "overlaps published" in check_digest_history(cluster, frozenset({"a2"}))


def test_primary_evidence_skips_investigation_records(tmp_path: Path) -> None:
    cluster = make_cluster("c1", "a1")
    with make_store(tmp_path) as store:
        assert "no stored primary" in fetch_primary_evidence(cluster, store)
        save_evidence(store, "a1-article-text", "a1")
        assert "a1-article-text" in fetch_primary_evidence(cluster, store)


def test_langchain_tools_bind_to_store(tmp_path: Path) -> None:
    article = make_article("a1")
    with make_store(tmp_path) as store:
        store.save(article)
        store.save(make_cluster("c1", "a1"))
        save_text(store, "a1", "Representative body.")
        tools = make_editorial_tools(store, {"a1": article})

        assert [item.name for item in tools] == [
            "read_article",
            "inspect_coverage",
            "fetch_evidence",
            "consult_history",
        ]
        assert "Representative body" in tools[0].invoke({"cluster_id": "c1"})
        assert "unknown cluster" in tools[0].invoke({"cluster_id": "missing"})
        assert "no additional" in tools[1].invoke({"cluster_id": "c1"})
        assert "unknown cluster" in tools[1].invoke({"cluster_id": "missing"})
        assert "no stored primary" in tools[2].invoke({"cluster_id": "c1"})
        assert "unknown cluster" in tools[2].invoke({"cluster_id": "missing"})
        assert "no digest history" in tools[3].invoke({"cluster_id": "c1"})
        assert "unknown cluster" in tools[3].invoke({"cluster_id": "missing"})


def test_build_editorial_agent_wires_tools(monkeypatch, tmp_path: Path) -> None:
    import ai_news_agent.editorial as editorial

    seen: dict = {}

    def _fake_create_agent(*, model, tools, system_prompt):
        seen.update({"model": model, "tools": tools, "prompt": system_prompt})
        return {"model": model}

    monkeypatch.setattr(editorial, "create_agent", _fake_create_agent)
    article = make_article("a1")
    with make_store(tmp_path) as store:
        agent = build_editorial_agent(
            "test-model", store=store, articles_by_id={"a1": article}
        )

    assert agent == {"model": "test-model"}
    assert len(seen["tools"]) == 4
    assert "investigator" in seen["prompt"].lower()


# Ranking


def test_ranking_orders_by_code_computed_total(tmp_path: Path) -> None:
    low = make_verdict(relevance=3, impact=3, novelty=3, evidence=3, timeliness=3)
    high = make_verdict(relevance=5, impact=5, novelty=4, evidence=5, timeliness=5)
    articles = (make_article("a1"), make_article("a2"))
    clusters = (make_cluster("c1", "a1"), make_cluster("c2", "a2"))

    class _Ordered(FakeJudge):
        def score(self, title, article_text, supporting, history_note):
            self.calls.append({"title": title})
            return high if title == "Story a2" else low

    with make_store(tmp_path) as store:
        summary = rank(clusters, articles, store, _Ordered())

    assert summary.ranked == 2
    assert summary.ranking == ("c2", "c1")
    with make_store(tmp_path) as _unused:
        pass
    # Totals come from Score weights, not model arithmetic.
    expected = Score(
        id="x",
        story_cluster_id="c2",
        relevance=5,
        impact=5,
        novelty=4,
        evidence=5,
        timeliness=5,
        rationale="r",
    ).weighted_total
    by_id = {item.cluster_id: item for item in summary.decisions}
    assert by_id["c2"].weighted_total == pytest.approx(expected)
    assert by_id["c2"].score_id == "c2-editorial"
    assert summary.model == "fake-editorial-v1"
    assert summary.termination_reason == "complete"


def test_stable_tie_break_by_cluster_id(tmp_path: Path) -> None:
    articles = (make_article("a1"), make_article("a2"))
    clusters = (make_cluster("c-b", "a1"), make_cluster("c-a", "a2"))
    with make_store(tmp_path) as store:
        summary = rank(clusters, articles, store, FakeJudge())

    assert summary.ranking == ("c-a", "c-b")


def test_uncertain_claim_triggers_revision_with_separate_evidence(
    tmp_path: Path,
) -> None:
    first = make_verdict(
        relevance=3,
        impact=3,
        novelty=3,
        evidence=2,
        timeliness=4,
        rationale="Uncertain without supporting numbers.",
        needs_more_evidence=True,
    )
    second = make_verdict(
        relevance=4,
        impact=4,
        novelty=3,
        evidence=4,
        timeliness=4,
        rationale="Confirmed by supporting excerpt.",
    )
    article = make_article("a1")
    with make_store(tmp_path) as store:
        save_text(store, "a1", "Primary representative body.")
        save_evidence(store, "a1-article-text", "a1")
        summary = rank(
            (make_cluster("c1", "a1"),), (article,), store, FakeJudge([first, second])
        )

    assert summary.revisions == 1
    assert summary.decisions[0].revisions == 1
    assert summary.decisions[0].rationale == "Confirmed by supporting excerpt."
    judge = FakeJudge([first, second])
    with make_store(tmp_path) as store:
        save_text(store, "a1", "Primary representative body.")
        rank((make_cluster("c1", "a1"),), (article,), store, judge)
    # Primary text stays primary; supporting context carries investigation.
    assert "Primary representative" in judge.calls[0]["article_text"]
    assert "primary evidence" in judge.calls[0]["supporting"]
    assert len(judge.calls) == 2


def test_revision_skipped_when_budgets_exhausted(tmp_path: Path) -> None:
    verdict = make_verdict(needs_more_evidence=True)
    article = make_article("a1")
    with make_store(tmp_path) as store:
        summary = rank(
            (make_cluster("c1", "a1"),),
            (article,),
            store,
            FakeJudge([verdict]),
            max_tool_calls=3,
        )

    assert summary.revisions == 0
    assert summary.decisions[0].revisions == 0


def test_budget_exhaustion_marks_remaining_incomplete(tmp_path: Path) -> None:
    articles = tuple(make_article(f"a{i}") for i in range(3))
    clusters = tuple(make_cluster(f"c{i}", f"a{i}") for i in range(3))
    with make_store(tmp_path) as store:
        summary = rank(clusters, articles, store, FakeJudge(), max_iterations=1)

    assert summary.ranked == 1
    assert summary.incomplete == 2
    assert summary.termination_reason == "budget-exhausted"
    assert summary.tool_calls == 3
    assert summary.judge_calls == 1
    exhausted = [item for item in summary.decisions if item.verdict == "incomplete"]
    assert all(item.termination_reason == "budget-exhausted" for item in exhausted)


def test_zero_time_budget_marks_all_incomplete(tmp_path: Path) -> None:
    article = make_article("a1")
    with make_store(tmp_path) as store:
        summary = rank(
            (make_cluster("c1", "a1"),), (article,), store, FakeJudge(), max_seconds=0
        )

    assert summary.ranked == 0
    assert summary.incomplete == 1
    assert summary.termination_reason == "budget-exhausted"


def test_missing_representative_is_incomplete(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        summary = rank((make_cluster("c1", "ghost"),), (), store, FakeJudge())

    assert summary.incomplete == 1
    assert summary.decisions[0].termination_reason == "missing-representative"
    assert summary.termination_reason == "complete"


def test_malformed_output_and_transport_become_incomplete(tmp_path: Path) -> None:
    class _Broken:
        @property
        def identity(self) -> str:
            return "broken"

        def score(self, title, article_text, supporting, history_note):
            raise EditorialOutputError("no JSON")

    class _Down:
        @property
        def identity(self) -> str:
            return "down"

        def score(self, title, article_text, supporting, history_note):
            raise EditorialError("timeout")

    article = make_article("a1")
    with make_store(tmp_path) as store:
        malformed = rank((make_cluster("c1", "a1"),), (article,), store, _Broken())
        assert malformed.decisions[0].termination_reason == "malformed-output"
        assert malformed.decisions[0].verdict == "incomplete"
        assert store.get(Score, "c1-editorial") is None

    with make_store(tmp_path) as store:
        down = rank((make_cluster("c1", "a1"),), (article,), store, _Down())
        assert down.decisions[0].termination_reason == "judge-unavailable"


def test_configuration_errors_fail_fast(tmp_path: Path) -> None:
    class _Misconfigured:
        @property
        def identity(self) -> str:
            return "misconfigured"

        def score(self, title, article_text, supporting, history_note):
            raise EditorialConfigurationError("no key")

    article = make_article("a1")
    with make_store(tmp_path) as store, pytest.raises(EditorialConfigurationError):
        rank((make_cluster("c1", "a1"),), (article,), store, _Misconfigured())


def test_build_judge_requires_credentials() -> None:
    with pytest.raises(EditorialConfigurationError, match="OPENROUTER_API_KEY"):
        build_editorial_judge(Settings(_env_file=None))


# OpenRouter transport


class _FakePostResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _editorial_payload(**kwargs) -> dict:
    verdict = {
        "relevance": 4,
        "impact": 4,
        "novelty": 3,
        "evidence": 4,
        "timeliness": 5,
        "evidence_ids": ["e1"],
        "uncertainty": 0.2,
        "rationale": "ok",
        "needs_more_evidence": False,
    }
    verdict.update(kwargs)
    return {"choices": [{"message": {"content": __import__("json").dumps(verdict)}}]}


def _judge() -> OpenRouterEditorialJudge:
    return OpenRouterEditorialJudge(api_key="test-key", model="test-model")


def test_openrouter_judge_posts_structured_request(monkeypatch) -> None:
    seen = {}

    def _post(url, *, headers, json, timeout):
        seen.update({"url": url, "headers": headers, "json": json})
        return _FakePostResponse(payload=_editorial_payload())

    monkeypatch.setattr(httpx, "post", _post)
    verdict = _judge().score("Title", "Body", "Supporting", "new")

    assert verdict.relevance == 4
    assert verdict.evidence_ids == ("e1",)
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["json"]["model"] == "test-model"
    assert seen["json"]["temperature"] == 0
    assert _judge().identity == "test-model"


def test_openrouter_judge_maps_failures(monkeypatch) -> None:
    judge = _judge()
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=401)
    )
    with pytest.raises(EditorialConfigurationError, match="credentials rejected"):
        judge.score("t", "b", "s", "h")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=503)
    )
    with pytest.raises(EditorialError, match="retryable"):
        judge.score("t", "b", "s", "h")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=404)
    )
    with pytest.raises(EditorialError, match="not retryable"):
        judge.score("t", "b", "s", "h")

    def _timeout(*a, **k):
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(httpx, "post", _timeout)
    with pytest.raises(EditorialError, match="timeout"):
        judge.score("t", "b", "s", "h")

    def _broken(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _broken)
    with pytest.raises(EditorialError, match="transport error"):
        judge.score("t", "b", "s", "h")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(payload={"nope": 1})
    )
    with pytest.raises(EditorialOutputError, match="envelope"):
        judge.score("t", "b", "s", "h")

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakePostResponse(
            payload={"choices": [{"message": {"content": "  "}}]}
        ),
    )
    with pytest.raises(EditorialOutputError, match="empty content"):
        judge.score("t", "b", "s", "h")


def test_rank_command_ranks_with_injected_judge(tmp_path: Path, monkeypatch) -> None:
    import json

    from typer.testing import CliRunner

    import ai_news_agent.cli as cli
    from ai_news_agent.cli import app
    from ai_news_agent.schemas import StoryCluster

    runner = CliRunner()
    database = tmp_path / "rank.db"
    with SQLiteStore(database) as store:
        store.migrate()
        article = make_article("a1", "Model release")
        store.save(article)
        store.save(
            StoryCluster(
                id="c1",
                article_ids=("a1",),
                representative_article_id="a1",
                rationale="seed",
                created_at=NOW,
            )
        )

    class _Judge:
        @property
        def identity(self) -> str:
            return "cli-fake"

        def score(self, title, article_text, supporting, history_note):
            return make_verdict()

    monkeypatch.setattr(
        cli, "build_editorial_judge", lambda settings, timeout=30.0: _Judge()
    )
    result = runner.invoke(
        app, ["rank", "--database", str(database), "--digest-run-id", "run-cli-ed"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["attempted"] == 1
    assert payload["ranked"] == 1
    assert payload["ranking"] == ["c1"]
    assert payload["digest_run_id"] == "run-cli-ed"
