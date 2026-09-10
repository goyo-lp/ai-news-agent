import json
import sqlite3
from contextlib import closing
from pathlib import Path

from typer.testing import CliRunner

from ai_news_agent.cli import app

runner = CliRunner()


def test_fixture_command_is_key_free_and_offline(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "fixture.db"
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    result = runner.invoke(app, ["fixture", "--database", str(database)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["record_count"] == 9
    assert payload["tracing"] == "disabled"
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 9


def test_trace_smoke_requires_explicit_configuration(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    result = runner.invoke(app, ["trace-smoke"])

    assert result.exit_code != 0
    assert "LANGSMITH_TRACING=true" in result.output


def test_ingest_command_runs_offline_for_adapter_sources(tmp_path: Path) -> None:
    import yaml

    source_path = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"
    configs = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    adapters_only = [
        c for c in configs if c["route"]["type"] == "official_page_adapter"
    ][:2]
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(yaml.safe_dump(adapters_only), encoding="utf-8")
    database = tmp_path / "ingest.db"

    result = runner.invoke(
        app,
        [
            "ingest",
            "--config",
            str(config_path),
            "--database",
            str(database),
            "--digest-run-id",
            "run-cli-test",
            "--max-workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["attempted"] == 2
    assert payload["unavailable"] == 2
    assert payload["digest_run_id"] == "run-cli-test"
    with closing(sqlite3.connect(database)) as connection:
        run_row = connection.execute(
            "SELECT payload_json FROM records WHERE record_type = 'run'"
        ).fetchone()
        assert run_row is not None


def test_retrieve_command_fetches_stored_articles(tmp_path: Path, monkeypatch) -> None:
    from datetime import UTC, datetime

    import ai_news_agent.extraction as extraction
    from ai_news_agent.extraction import FetchedPage
    from ai_news_agent.schemas import Article
    from ai_news_agent.storage import SQLiteStore

    now = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
    database = tmp_path / "retrieve.db"
    with SQLiteStore(database) as store:
        store.migrate()
        store.save(
            Article(
                id="article-cli",
                source_id="source-01",
                url="https://example.com/ai/cli",
                canonical_url="https://example.com/ai/cli",
                title="CLI story",
                published_at=now,
                first_seen_at=now,
            )
        )
    html = (Path(__file__).parent / "fixtures" / "articles" / "valid.html").read_text(
        encoding="utf-8"
    )

    def _page(url, *, timeout, max_bytes):
        return FetchedPage(url=url, body=html.encode("utf-8"))

    monkeypatch.setattr(extraction, "fetch_article_html", _page)

    result = runner.invoke(app, ["retrieve", "--database", str(database)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["attempted"] == 1
    assert payload["retrieved"] == 1
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT payload_json FROM records WHERE record_type = 'article_text'"
        ).fetchone()
        assert row is not None


def test_cluster_command_groups_stored_articles(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from ai_news_agent.schemas import Article
    from ai_news_agent.storage import SQLiteStore

    now = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
    database = tmp_path / "cluster.db"

    def _article(article_id: str, title: str) -> Article:
        url = f"https://example.com/ai/{article_id}"
        return Article(
            id=article_id,
            source_id="source-01",
            url=url,
            canonical_url=url,
            title=title,
            published_at=now,
            first_seen_at=now,
        )

    with SQLiteStore(database) as store:
        store.migrate()
        store.save(_article("a1", "Fixture Lab releases Model B with open weights"))
        store.save(_article("a2", "Fixture Lab Releases Model B With Open Weights"))
        store.save(_article("b1", "Rival Corp unveils Robot Chef for home kitchens"))

    result = runner.invoke(app, ["cluster", "--database", str(database)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["attempted"] == 3
    assert payload["clusters"] == 2
    with closing(sqlite3.connect(database)) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM records WHERE record_type = 'story_cluster'"
        ).fetchone()[0]
        assert count == 2


def test_screen_command_shortlists_with_injected_judge(
    tmp_path: Path, monkeypatch
) -> None:
    from datetime import UTC, datetime

    import ai_news_agent.cli as cli
    import ai_news_agent.screening as screening
    from ai_news_agent.extraction import RetrievalSummary
    from ai_news_agent.schemas import Article, StoryCluster
    from ai_news_agent.screening import RelevanceVerdict
    from ai_news_agent.storage import SQLiteStore

    now = datetime.now(UTC)
    database = tmp_path / "screen.db"

    def _article(article_id: str, title: str) -> Article:
        url = f"https://example.com/ai/{article_id}"
        return Article(
            id=article_id,
            source_id="source-01",
            url=url,
            canonical_url=url,
            title=title,
            published_at=now,
            first_seen_at=now,
        )

    with SQLiteStore(database) as store:
        store.migrate()
        store.save(_article("a1", "Model release"))
        store.save(_article("a2", "Gadget cases"))
        for cluster_id, article_id in (("c1", "a1"), ("c2", "a2")):
            store.save(
                StoryCluster(
                    id=cluster_id,
                    article_ids=(article_id,),
                    representative_article_id=article_id,
                    rationale="seed",
                    created_at=now,
                )
            )

    class _Judge:
        @property
        def identity(self) -> str:
            return "cli-fake"

        def judge(self, title: str, excerpt: str, source_name: str):
            if "Gadget" in title:
                return RelevanceVerdict(
                    relevant=False, borderline=False, reason="Not AI."
                )
            return RelevanceVerdict(
                relevant=True, borderline=False, reason="Model release."
            )

    monkeypatch.setattr(cli, "build_judge", lambda settings, timeout=30.0: _Judge())
    monkeypatch.setattr(
        screening,
        "retrieve_articles",
        lambda *args, **kwargs: RetrievalSummary(),
    )

    result = runner.invoke(app, ["screen", "--database", str(database)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["attempted"] == 2
    assert payload["shortlisted"] == 1
    assert payload["rejected"] == 1
