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
