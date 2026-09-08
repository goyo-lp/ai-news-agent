from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from ai_news_agent.config import Settings
from ai_news_agent.extraction import FetchedPage
from ai_news_agent.schemas import Article, StoryCluster
from ai_news_agent.screening import (
    JudgeConfigurationError,
    JudgeError,
    JudgeOutputError,
    OpenRouterJudge,
    RelevanceVerdict,
    build_judge,
    cache_key,
    cache_lookup,
    cache_store,
    judge_prompt,
    judge_with_retries,
    parse_verdict,
    screen_clusters,
)
from ai_news_agent.storage import SQLiteStore

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
RELEVANT = RelevanceVerdict(relevant=True, borderline=False, reason="Model release.")
IRRELEVANT = RelevanceVerdict(relevant=False, borderline=False, reason="Not AI.")
BORDERLINE = RelevanceVerdict(relevant=False, borderline=True, reason="Unsure.")


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


def make_cluster(cluster_id: str, *article_ids: str) -> StoryCluster:
    return StoryCluster(
        id=cluster_id,
        article_ids=tuple(article_ids),
        representative_article_id=article_ids[0],
        rationale="test cluster",
        created_at=NOW,
    )


def make_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "screen.db")
    store.migrate()
    return store


class FakeJudge:
    """Route verdicts by title keyword, recording every call."""

    def __init__(
        self,
        default: RelevanceVerdict = RELEVANT,
        error: Exception | None = None,
    ) -> None:
        self.default = default
        self.error = error
        self.calls: list[str] = []

    @property
    def identity(self) -> str:
        return "fake-judge-v1"

    def judge(self, title: str, excerpt: str, source_name: str) -> RelevanceVerdict:
        self.calls.append(title)
        if self.error is not None:
            raise self.error
        if "gadget" in title.lower():
            return IRRELEVANT
        if "rumor" in title.lower():
            return BORDERLINE
        return self.default


def screen(
    clusters: tuple[StoryCluster, ...],
    articles: tuple[Article, ...],
    store: SQLiteStore,
    judge: FakeJudge,
    **kwargs,
):
    return screen_clusters(
        clusters,
        {article.id: article for article in articles},
        store,
        now=NOW,
        judge=judge,
        retrieve_texts=False,
        **kwargs,
    )


# Verdict parsing


def test_parse_verdict_accepts_strict_and_fenced_json() -> None:
    verdict = parse_verdict('{"relevant": true, "borderline": false, "reason": "x"}')

    assert verdict == RelevanceVerdict(relevant=True, borderline=False, reason="x")
    fenced = parse_verdict(
        '```json\n{"relevant": false, "borderline": true, "reason": "y"}\n```'
    )
    assert fenced.borderline is True
    assert fenced.reason == "y"


def test_parse_verdict_rejects_malformed_output() -> None:
    for bad in (
        "no json here",
        '{"relevant": true}',
        '{"relevant": "yes", "borderline": false, "reason": "x"}',
        '{"relevant": false, "borderline": false, "reason": "  "}',
        '{"relevant": tru}',
        "[1, 2, 3]",
        "{oops",
        "```still no json here",
    ):
        with pytest.raises(JudgeOutputError):
            parse_verdict(bad)


def test_judge_prompt_encodes_policy_rules() -> None:
    prompt = judge_prompt("Title", "Excerpt", "Source")

    assert "artificial intelligence" in prompt.lower()
    assert "borderline" in prompt.lower()
    assert "Title" in prompt


# Cache behavior


def test_cache_key_covers_content_policy_prompt_model() -> None:
    base = cache_key("t", "e", model="m")
    assert base == cache_key("t", "e", model="m")
    assert base != cache_key("t!", "e", model="m")
    assert base != cache_key("t", "e", model="other")
    assert base != cache_key("t", "e", model="m", policy_version="other")


def test_cache_roundtrip_and_corrupt_entry(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        assert cache_lookup(store, "missing") is None
        cache_store(store, "key-1", RELEVANT)
        assert cache_lookup(store, "key-1") == RELEVANT
        store.connection.execute(
            "INSERT INTO screening_cache (cache_key, verdict_json) VALUES (?, ?)",
            ("key-bad", '{"wrong": "shape"}'),
        )
        assert cache_lookup(store, "key-bad") is None


def test_cache_lookup_tolerates_missing_table(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "bare.db")

    assert cache_lookup(store, "anything") is None
    store.close()


def test_excerpt_falls_back_to_title_without_feed_evidence(tmp_path: Path) -> None:
    seen: list[str] = []

    class ExcerptSpy(FakeJudge):
        def judge(self, title: str, excerpt: str, source_name: str):
            seen.append(excerpt)
            return super().judge(title, excerpt, source_name)

    article = make_article("a1")
    with make_store(tmp_path) as store:
        screen((make_cluster("c1", "a1"),), (article,), store, ExcerptSpy())

    assert seen == ["Story a1"]


# Date and history rules


def test_stale_clusters_rejected_without_judge_call(tmp_path: Path) -> None:
    article = make_article("old", hours_before=48)
    judge = FakeJudge()
    with make_store(tmp_path) as store:
        summary = screen((make_cluster("c1", "old"),), (article,), store, judge)

    assert summary.rejected == 1
    assert "outside 36h window" in summary.decisions[0].reason
    assert judge.calls == []


def test_late_arrival_records_delay_and_qualifies(tmp_path: Path) -> None:
    article = make_article("late", hours_before=30)
    judge = FakeJudge()
    with make_store(tmp_path) as store:
        summary = screen((make_cluster("c1", "late"),), (article,), store, judge)

    assert summary.shortlisted == 1
    assert summary.decisions[0].late_arrival is True


def test_recycled_clusters_rejected(tmp_path: Path) -> None:
    article = make_article("a1")
    judge = FakeJudge()
    with make_store(tmp_path) as store:
        summary = screen(
            (make_cluster("c1", "a1"),),
            (article,),
            store,
            judge,
            published_article_ids=frozenset({"a1"}),
        )

    assert summary.rejected == 1
    assert "already published" in summary.decisions[0].reason
    assert judge.calls == []


def test_missing_representative_rejected(tmp_path: Path) -> None:
    judge = FakeJudge()
    with make_store(tmp_path) as store:
        summary = screen((make_cluster("c1", "ghost"),), (), store, judge)

    assert summary.rejected == 1
    assert "missing from store" in summary.decisions[0].reason


# Judge verdicts


def test_important_stories_survive_screening(tmp_path: Path) -> None:
    articles = (
        make_article("model"),
        Article(
            **{
                **make_article("gadget").model_dump(),
                "id": "gadget",
                "title": "New phone gadget cases reviewed",
            },
        ),
        Article(
            **{
                **make_article("rumor").model_dump(),
                "id": "rumor",
                "title": "Rumor: secret model maybe soon",
            },
        ),
    )
    clusters = (
        make_cluster("c-model", "model"),
        make_cluster("c-gadget", "gadget"),
        make_cluster("c-rumor", "rumor"),
    )
    judge = FakeJudge()
    with make_store(tmp_path) as store:
        summary = screen(clusters, articles, store, judge)

    assert summary.shortlisted == 1
    assert summary.borderline == 1
    assert summary.rejected == 1
    by_id = {item.cluster_id: item for item in summary.decisions}
    assert by_id["c-model"].verdict == "shortlisted"
    assert by_id["c-gadget"].verdict == "rejected"
    assert "Not AI" in by_id["c-gadget"].reason
    assert by_id["c-rumor"].verdict == "borderline"


def test_malformed_judge_output_becomes_cached_borderline(tmp_path: Path) -> None:
    article = make_article("a1")
    judge = FakeJudge(error=JudgeOutputError("judge returned no JSON object"))
    with make_store(tmp_path) as store:
        first = screen((make_cluster("c1", "a1"),), (article,), store, judge)
        second = screen((make_cluster("c1", "a1"),), (article,), store, judge)

    assert first.borderline == 1
    assert "preserved for review" in first.decisions[0].reason
    assert len(judge.calls) == 1
    assert second.decisions[0].cached is True


def test_transport_errors_become_uncached_borderline(tmp_path: Path) -> None:
    article = make_article("a1")
    judge = FakeJudge(error=JudgeError("judge timeout after 30.0s"))
    with make_store(tmp_path) as store:
        first = screen((make_cluster("c1", "a1"),), (article,), store, judge)
        second = screen((make_cluster("c1", "a1"),), (article,), store, judge)

    assert first.borderline == 1
    assert first.judge_errors == 1
    assert len(judge.calls) == 4
    assert second.decisions[0].cached is False


def test_configuration_errors_fail_fast(tmp_path: Path) -> None:
    article = make_article("a1")
    judge = FakeJudge(error=JudgeConfigurationError("credentials rejected"))
    with make_store(tmp_path) as store, pytest.raises(JudgeConfigurationError):
        screen((make_cluster("c1", "a1"),), (article,), store, judge)


def test_build_judge_requires_credentials() -> None:
    with pytest.raises(JudgeConfigurationError, match="OPENROUTER_API_KEY"):
        build_judge(Settings(_env_file=None))


# Shortlist cap


def test_shortlist_cap_rejects_overflow_with_reasons(tmp_path: Path) -> None:
    articles = tuple(make_article(f"a{i}") for i in range(3))
    clusters = tuple(make_cluster(f"c{i}", f"a{i}") for i in range(3))
    judge = FakeJudge()
    with make_store(tmp_path) as store:
        summary = screen(clusters, articles, store, judge, max_candidates=2)

    assert summary.shortlisted == 2
    assert summary.rejected == 1
    overflow = next(item for item in summary.decisions if item.verdict == "rejected")
    assert "outside top-2" in overflow.reason


def test_relevant_outranks_newer_borderline(tmp_path: Path) -> None:
    old = make_article("old", hours_before=5)
    new_rumor = Article(
        **{**make_article("new", hours_before=1).model_dump(), "title": "Rumor mill"}
    )
    clusters = (make_cluster("c-old", "old"), make_cluster("c-new", "new"))
    judge = FakeJudge(default=RELEVANT)
    with make_store(tmp_path) as store:
        summary = screen(clusters, (old, new_rumor), store, judge, max_candidates=1)

    by_id = {item.cluster_id: item for item in summary.decisions}
    assert by_id["c-old"].verdict == "shortlisted"
    assert by_id["c-new"].verdict == "rejected"
    assert "outside top-1" in by_id["c-new"].reason


def test_second_run_reuses_cache_without_calls(tmp_path: Path) -> None:
    articles = tuple(make_article(f"a{i}") for i in range(2))
    clusters = tuple(make_cluster(f"c{i}", f"a{i}") for i in range(2))
    judge = FakeJudge()
    with make_store(tmp_path) as store:
        first = screen(clusters, articles, store, judge)
        second = screen(clusters, articles, store, judge)

    assert first.judge_calls == 2
    assert second.judge_calls == 0
    assert second.judge_cached == 2


def test_retrieval_merges_identical_shortlist_texts(tmp_path: Path) -> None:
    articles = (
        make_article("a1", hours_before=1),
        Article(
            **{
                **make_article("a2", hours_before=2).model_dump(),
                "title": "Unrelated headline about splendid devices",
            },
        ),
    )
    clusters = (make_cluster("c1", "a1"), make_cluster("c2", "a2"))
    judge = FakeJudge()
    body = (
        "<html><body><article><h1>Shared</h1><p>"
        + "Identical retrieved body text. " * 30
        + "</p></article></body></html>"
    )

    def _fetch(url, **kwargs):
        return FetchedPage(url=url, body=body.encode("utf-8"))

    with make_store(tmp_path) as store:
        summary = screen_clusters(
            clusters,
            {article.id: article for article in articles},
            store,
            now=NOW,
            judge=judge,
            fetcher=_fetch,
        )

    assert summary.shortlisted == 2
    assert summary.merged == 1
    assert len(summary.shortlist) == 1
    assert "merged after retrieval" in summary.shortlist[0].rationale


# Retry helper


def test_judge_with_retries_succeeds_after_transient_failure() -> None:
    calls = {"count": 0}

    class Flaky:
        @property
        def identity(self) -> str:
            return "flaky"

        def judge(self, title: str, excerpt: str, source_name: str):
            calls["count"] += 1
            if calls["count"] == 1:
                raise JudgeError("judge HTTP 503, retryable")
            return RELEVANT

    verdict = judge_with_retries(Flaky(), "t", "e", "s", sleep=lambda _: None)

    assert verdict == RELEVANT
    assert calls["count"] == 2


def test_judge_with_retries_skips_output_errors() -> None:
    calls = {"count": 0}

    class Broken:
        @property
        def identity(self) -> str:
            return "broken"

        def judge(self, title: str, excerpt: str, source_name: str):
            calls["count"] += 1
            raise JudgeOutputError("no JSON object")

    with pytest.raises(JudgeOutputError):
        judge_with_retries(Broken(), "t", "e", "s", sleep=lambda _: None)

    assert calls["count"] == 1


# OpenRouter transport


class _FakePostResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _verdict_payload(**kwargs):
    verdict = {"relevant": True, "borderline": False, "reason": "ok"}
    verdict.update(kwargs)
    return {"choices": [{"message": {"content": __import__("json").dumps(verdict)}}]}


def _judge() -> OpenRouterJudge:
    return OpenRouterJudge(api_key="test-key", model="test-model")


def test_openrouter_judge_posts_structured_request(monkeypatch) -> None:
    seen = {}

    def _post(url, *, headers, json, timeout):
        seen.update({"url": url, "headers": headers, "json": json})
        return _FakePostResponse(payload=_verdict_payload())

    monkeypatch.setattr(httpx, "post", _post)
    verdict = _judge().judge("Title", "Excerpt", "Source")

    assert verdict == RelevanceVerdict(relevant=True, borderline=False, reason="ok")
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
    with pytest.raises(JudgeConfigurationError, match="credentials rejected"):
        judge.judge("t", "e", "s")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=503)
    )
    with pytest.raises(JudgeError, match="retryable"):
        judge.judge("t", "e", "s")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=404)
    )
    with pytest.raises(JudgeError, match="not retryable"):
        judge.judge("t", "e", "s")

    def _timeout(*a, **k):
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(httpx, "post", _timeout)
    with pytest.raises(JudgeError, match="timeout"):
        judge.judge("t", "e", "s")

    def _broken(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", _broken)
    with pytest.raises(JudgeError, match="transport error"):
        judge.judge("t", "e", "s")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(payload={"nope": 1})
    )
    with pytest.raises(JudgeOutputError, match="envelope"):
        judge.judge("t", "e", "s")

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakePostResponse(
            payload={"choices": [{"message": {"content": "  "}}]}
        ),
    )
    with pytest.raises(JudgeOutputError, match="empty content"):
        judge.judge("t", "e", "s")
