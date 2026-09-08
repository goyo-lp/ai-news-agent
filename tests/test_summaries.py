import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ai_news_agent.config import Settings
from ai_news_agent.schemas import (
    Article,
    ArticleText,
    Evidence,
    EvidenceKind,
    StoryCluster,
    Summary,
)
from ai_news_agent.storage import SQLiteStore
from ai_news_agent.summaries import (
    OpenRouterSummaryWriter,
    SummaryConfigurationError,
    SummaryDraft,
    SummaryError,
    SummaryOutputError,
    article_evidence,
    build_digest_items,
    build_summary_writer,
    generate_summaries,
    parse_summary_draft,
    render_archive_html,
    render_email_html,
    render_email_text,
    summary_prompt,
    validate_sentence,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
HASH = "ab" * 32
BODY = "Representative article body with concrete detail. " * 30


def make_article(
    article_id: str, title: str = "Example Lab releases Model A"
) -> Article:
    url = f"https://example.com/ai/{article_id}"
    return Article(
        id=article_id,
        source_id="source-01",
        url=url,
        canonical_url=url,
        title=title,
        published_at=NOW,
        first_seen_at=NOW,
    )


def make_cluster(cluster_id: str, article_id: str) -> StoryCluster:
    return StoryCluster(
        id=cluster_id,
        article_ids=(article_id,),
        representative_article_id=article_id,
        rationale="seed",
        created_at=NOW,
    )


def make_draft(**overrides) -> SummaryDraft:
    payload = {
        "sentences": (
            "Example Lab released Model A with weights available today.",
            "Its report shows a 20% improvement on Benchmark B at v3.14.",
            "The release may lower costs, but the vendor result is provisional.",
        ),
        "evidence_by_sentence": (("a1-text",), ("a1-text",), ("a1-text",)),
    }
    payload.update(overrides)
    return SummaryDraft(**payload)


class FakeWriter:
    def __init__(self, draft: SummaryDraft | None = None) -> None:
        self.draft = draft or make_draft()
        self.calls: list[dict] = []

    @property
    def identity(self) -> str:
        return "fake-summary-v1"

    def write(
        self, title: str, article_text: str, evidence_excerpts: str
    ) -> SummaryDraft:
        self.calls.append({"title": title, "text": article_text})
        return self.draft


def make_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "summaries.db")
    store.migrate()
    return store


def save_body(store: SQLiteStore, article_id: str, text: str = BODY) -> None:
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


def test_prompt_describes_three_sentence_contract() -> None:
    prompt = summary_prompt("Title", BODY, "[e1] excerpt")

    assert "exactly three sentences" in prompt
    assert "Attribute" in prompt or "attribute" in prompt
    assert "Title" in prompt


def test_validate_sentence_accepts_abbreviations_and_decimals() -> None:
    assert validate_sentence(
        "The U.S. regulator published final duties today."
    ).endswith(".")
    assert validate_sentence(
        "Accuracy rose to 3.14 on the eval, e.g. math tasks."
    ).endswith(".")


def test_validate_sentence_rejects_structure_violations() -> None:
    for bad in (
        "",
        "Two sentences. Hidden second.",
        "What happened",
        "- bullet sentence here.",
        "First clause; hidden second clause.",
        "Line one.\nLine two.",
        "Too short.",
    ):
        with pytest.raises(SummaryOutputError):
            validate_sentence(bad)


def test_parse_draft_accepts_strict_and_fenced_json() -> None:
    payload = {
        "sentence1": "Example Lab released Model A today.",
        "sentence2": "The report cites a 20% gain on Benchmark B.",
        "sentence3": "Builders may save costs if replication confirms it.",
        "evidence_by_sentence": [["e1"], ["e1"], ["e1"]],
    }
    raw = json.dumps(payload)
    assert parse_summary_draft(raw).sentences[0].startswith("Example Lab")
    assert parse_summary_draft(f"```json\n{raw}\n```").evidence_by_sentence[0] == (
        "e1",
    )


def test_parse_draft_rejects_bad_shapes() -> None:
    good = {
        "sentence1": "Example Lab released Model A today.",
        "sentence2": "The report cites a 20% gain on Benchmark B.",
        "sentence3": "Builders may save costs if replication confirms it.",
        "evidence_by_sentence": [["e1"], ["e1"], ["e1"]],
    }
    cases = [
        "no json",
        json.dumps({**good, "evidence_by_sentence": [["e1"], ["e1"], []]}),
        json.dumps({**good, "evidence_by_sentence": [["e1"], ["e1"]]}),
        json.dumps({**good, "sentence3": "Two sentences. Hidden."}),
        json.dumps({k: v for k, v in good.items() if k != "sentence3"}),
        "[1,2,3]",
    ]
    for bad in cases:
        with pytest.raises(SummaryOutputError):
            parse_summary_draft(bad)


def test_headline_only_never_gets_a_summary(tmp_path: Path) -> None:
    article = make_article("a1")
    with make_store(tmp_path) as store:
        assert (
            article_evidence(make_cluster("c1", "a1"), {"a1": article}, store) is None
        )
        result = generate_summaries(
            (make_cluster("c1", "a1"),),
            {"a1": article},
            store,
            now=NOW,
            writer=FakeWriter(),
        )

    assert result.generated == 0
    assert result.insufficient == 1
    assert "insufficient article text" in result.decisions[0].reason


def test_thin_text_is_insufficient(tmp_path: Path) -> None:
    article = make_article("a1")
    with make_store(tmp_path) as store:
        save_body(store, "a1", "Too short.")
        assert (
            article_evidence(make_cluster("c1", "a1"), {"a1": article}, store) is None
        )


def test_generation_saves_draft_with_evidence(tmp_path: Path) -> None:
    article = make_article("a1")
    with make_store(tmp_path) as store:
        save_body(store, "a1")
        result = generate_summaries(
            (make_cluster("c1", "a1"),),
            {"a1": article},
            store,
            now=NOW,
            writer=FakeWriter(),
        )

    assert result.generated == 1
    (summary,) = result.summaries
    assert summary.id == "c1-summary"
    assert summary.status == "draft"
    assert len(summary.sentences) == 3
    assert summary.evidence_by_sentence[0] == ("a1-text",)
    assert result.decisions[0].summary_id == "c1-summary"


def test_missing_article_and_writer_failures_are_insufficient(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        missing = generate_summaries(
            (make_cluster("c1", "ghost"),), {}, store, now=NOW, writer=FakeWriter()
        )
        assert "missing from store" in missing.decisions[0].reason

    class _Broken:
        @property
        def identity(self) -> str:
            return "broken"

        def write(self, title, article_text, evidence_excerpts):
            raise SummaryOutputError("no JSON")

    class _Down:
        @property
        def identity(self) -> str:
            return "down"

        def write(self, title, article_text, evidence_excerpts):
            raise SummaryError("timeout")

    article = make_article("a1")
    with make_store(tmp_path) as store:
        save_body(store, "a1")
        broken = generate_summaries(
            (make_cluster("c1", "a1"),),
            {"a1": article},
            store,
            now=NOW,
            writer=_Broken(),
        )
        assert "unusable" in broken.decisions[0].reason
    with make_store(tmp_path) as store:
        save_body(store, "a1")
        down = generate_summaries(
            (make_cluster("c1", "a1"),), {"a1": article}, store, now=NOW, writer=_Down()
        )
        assert "unavailable" in down.decisions[0].reason


def test_configuration_errors_fail_fast(tmp_path: Path) -> None:
    from ai_news_agent.summaries import SummaryConfigurationError as ConfigError

    class _Misconfigured:
        @property
        def identity(self) -> str:
            return "misconfigured"

        def write(self, title, article_text, evidence_excerpts):
            raise ConfigError("no key")

    article = make_article("a1")
    with make_store(tmp_path) as store:
        save_body(store, "a1")
        with pytest.raises(ConfigError):
            generate_summaries(
                (make_cluster("c1", "a1"),),
                {"a1": article},
                store,
                now=NOW,
                writer=_Misconfigured(),
            )


def test_build_writer_requires_credentials() -> None:
    with pytest.raises(SummaryConfigurationError, match="OPENROUTER_API_KEY"):
        build_summary_writer(Settings(_env_file=None))


def test_renderers_share_one_structured_digest_and_escape(tmp_path: Path) -> None:
    tricky = make_article("a1", title='Model <A> & "friends" released')
    draft = make_draft(
        sentences=(
            "Example Lab released Model A today.",
            "The U.S. report cites version 3.14 gains.",
            "Builders may benefit once replication lands.",
        )
    )
    with make_store(tmp_path) as store:
        save_body(store, "a1")
        generate_summaries(
            (make_cluster("c1", "a1"),),
            {"a1": tricky},
            store,
            now=NOW,
            writer=FakeWriter(draft),
        )
        summaries = {
            item.story_cluster_id: item for item in store.iter_records(Summary)
        }
        items = build_digest_items(
            ("c1",), {"c1": make_cluster("c1", "a1")}, {"a1": tricky}, summaries, store
        )
        email_html = render_email_html(items)
        email_text = render_email_text(items)
        archive_html = render_archive_html(items)

    assert len(items) == 1
    for rendered in (email_html, email_text, archive_html):
        assert "Model A today" in rendered
        assert "3.14" in rendered
    assert "Draft" in email_html and "unverified" in email_html
    assert "&lt;A&gt;" in email_html and "&amp;" in email_html
    assert "<A>" not in email_html
    assert "max-width:600px" in email_html
    assert "Draft" in archive_html


def test_summarize_command_writes_local_previews(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from ai_news_agent.cli import app

    runner = CliRunner()
    database = tmp_path / "summaries.db"
    html_path = tmp_path / "email.html"
    text_path = tmp_path / "email.txt"
    archive_path = tmp_path / "archive.html"
    with make_store(tmp_path):
        seed = SQLiteStore(database)
        seed.migrate()
        article = make_article("a1")
        seed.save(article)
        seed.save(make_cluster("c1", "a1"))
        save_body(seed, "a1")
        seed.close()

    import ai_news_agent.cli as cli

    def _writer(settings, timeout=30.0):
        return FakeWriter()

    original = cli.build_summary_writer
    cli.build_summary_writer = _writer
    try:
        result = runner.invoke(
            app,
            [
                "summarize",
                "--database",
                str(database),
                "--html-path",
                str(html_path),
                "--text-path",
                str(text_path),
                "--archive-path",
                str(archive_path),
            ],
        )
    finally:
        cli.build_summary_writer = original

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["generated"] == 1
    assert "Draft" in html_path.read_text(encoding="utf-8")
    assert "Model A" in text_path.read_text(encoding="utf-8")
    assert "Draft" in archive_path.read_text(encoding="utf-8")


class _FakePostResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _writer_payload(**kwargs) -> dict:
    draft = {
        "sentence1": "Example Lab released Model A today.",
        "sentence2": "The report cites a 20% gain on Benchmark B.",
        "sentence3": "Builders may save costs if replication confirms it.",
        "evidence_by_sentence": [["e1"], ["e1"], ["e1"]],
    }
    draft.update(kwargs)
    return {"choices": [{"message": {"content": json.dumps(draft)}}]}


def _writer_client() -> OpenRouterSummaryWriter:
    return OpenRouterSummaryWriter(api_key="test-key", model="test-model")


def test_openrouter_writer_posts_structured_request(monkeypatch) -> None:
    seen = {}

    def _post(url, *, headers, json, timeout):
        seen.update({"url": url, "headers": headers, "json": json})
        return _FakePostResponse(payload=_writer_payload())

    monkeypatch.setattr(httpx, "post", _post)
    draft = _writer_client().write("Title", BODY, "[e1] excerpt")

    assert draft.sentences[0].startswith("Example Lab")
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["json"]["model"] == "test-model"
    assert _writer_client().identity == "test-model"


def test_openrouter_writer_maps_failures(monkeypatch) -> None:
    writer = _writer_client()
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=401)
    )
    with pytest.raises(SummaryConfigurationError, match="credentials rejected"):
        writer.write("t", BODY, "e")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=503)
    )
    with pytest.raises(SummaryError, match="retryable"):
        writer.write("t", BODY, "e")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(status_code=404)
    )
    with pytest.raises(SummaryError, match="not retryable"):
        writer.write("t", BODY, "e")

    def _timeout(*a, **k):
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(httpx, "post", _timeout)
    with pytest.raises(SummaryError, match="timeout"):
        writer.write("t", BODY, "e")

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _FakePostResponse(payload={"nope": 1})
    )
    with pytest.raises(SummaryOutputError, match="envelope"):
        writer.write("t", BODY, "e")

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _FakePostResponse(
            payload={"choices": [{"message": {"content": "  "}}]}
        ),
    )
    with pytest.raises(SummaryOutputError, match="empty content"):
        writer.write("t", BODY, "e")
