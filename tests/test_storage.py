import sqlite3
from pathlib import Path

import pytest
from pydantic import BaseModel

from ai_news_agent.fixtures import load_sample_records
from ai_news_agent.schemas import Source
from ai_news_agent.storage import SQLiteStore


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path / "records.db") as store:
        assert store.migrate() == (
            "0001_records.sql",
            "0002_feed_state.sql",
            "0003_screening_cache.sql",
        )
        assert store.migrate() == ()
        names = store.connection.execute(
            "SELECT name FROM schema_migrations"
        ).fetchall()

    assert [row["name"] for row in names] == [
        "0001_records.sql",
        "0002_feed_state.sql",
        "0003_screening_cache.sql",
    ]


def test_records_round_trip_without_duplicates(tmp_path: Path) -> None:
    records = load_sample_records()
    with SQLiteStore(tmp_path / "records.db") as store:
        store.migrate()
        for item in records:
            store.save(item)
            store.save(item)

        count = store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        source = store.get(Source, "source-fixture")
        source_ids = [item.id for item in store.iter_records(Source)]

    assert count == len(records)
    assert source == records[0]
    assert source_ids == ["source-fixture"]


def test_missing_record_returns_none() -> None:
    with SQLiteStore(":memory:") as store:
        store.migrate()
        assert store.get(Source, "missing") is None


def test_database_rejects_invalid_json() -> None:
    with SQLiteStore(":memory:") as store:
        store.migrate()
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "INSERT INTO records (record_type, record_id, payload_json) "
                "VALUES ('source', 'broken', 'not-json')"
            )


def test_unknown_model_is_rejected() -> None:
    class Unknown(BaseModel):
        id: str

    with SQLiteStore(":memory:") as store:
        store.migrate()
        with pytest.raises(TypeError, match="unsupported record model"):
            store.get(Unknown, "unknown")
