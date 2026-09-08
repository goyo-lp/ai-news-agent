"""Small SQLite record store with forward-only migrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path

from pydantic import BaseModel

from ai_news_agent.schemas import (
    Article,
    ArticleText,
    Evidence,
    Run,
    Score,
    Source,
    StoredRecord,
    StoryCluster,
    Summary,
    VerificationResult,
)

_MODEL_TO_TYPE: dict[type[BaseModel], str] = {
    Source: "source",
    Article: "article",
    ArticleText: "article_text",
    StoryCluster: "story_cluster",
    Score: "score",
    Evidence: "evidence",
    Summary: "summary",
    VerificationResult: "verification_result",
    Run: "run",
}


class SQLiteStore(AbstractContextManager["SQLiteStore"]):
    """Persist validated records as versionable JSON envelopes in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        path = str(database_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def migrate(self) -> tuple[str, ...]:
        """Apply each bundled migration once and return newly applied names."""

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        existing = {
            row["name"]
            for row in self.connection.execute("SELECT name FROM schema_migrations")
        }
        applied: list[str] = []
        migration_dir = Path(__file__).with_name("migrations")
        for migration in sorted(migration_dir.glob("*.sql")):
            if migration.name in existing:
                continue
            self.connection.executescript(migration.read_text(encoding="utf-8"))
            self.connection.execute(
                "INSERT INTO schema_migrations (name) VALUES (?)", (migration.name,)
            )
            self.connection.commit()
            applied.append(migration.name)
        return tuple(applied)

    def save(self, record: StoredRecord) -> None:
        """Insert or replace one validated record without duplicating its ID."""

        record_type = _record_type(type(record))
        payload = record.model_dump_json(exclude_computed_fields=True)
        self.connection.execute(
            """
            INSERT INTO records (record_type, record_id, payload_json)
            VALUES (?, ?, ?)
            ON CONFLICT(record_type, record_id) DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (record_type, record.id, payload),
        )
        self.connection.commit()

    def get[RecordModel: BaseModel](
        self, model: type[RecordModel], record_id: str
    ) -> RecordModel | None:
        """Load and revalidate a record, returning None when it is absent."""

        row = self.connection.execute(
            "SELECT payload_json FROM records WHERE record_type = ? AND record_id = ?",
            (_record_type(model), record_id),
        ).fetchone()
        if row is None:
            return None
        return model.model_validate_json(row["payload_json"])

    def iter_records[RecordModel: BaseModel](
        self, model: type[RecordModel]
    ) -> Iterator[RecordModel]:
        """Yield one record type in stable ID order."""

        rows = self.connection.execute(
            "SELECT payload_json FROM records WHERE record_type = ? ORDER BY record_id",
            (_record_type(model),),
        )
        for row in rows:
            yield model.model_validate_json(row["payload_json"])


def _record_type[RecordModel: BaseModel](model: type[RecordModel]) -> str:
    try:
        return _MODEL_TO_TYPE[model]
    except KeyError as exc:
        raise TypeError(f"unsupported record model: {model.__name__}") from exc
