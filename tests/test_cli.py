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
