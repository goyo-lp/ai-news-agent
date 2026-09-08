CREATE TABLE IF NOT EXISTS feed_state (
    source_id TEXT PRIMARY KEY,
    etag TEXT,
    last_modified TEXT,
    content_hash TEXT NOT NULL DEFAULT '',
    last_checked_at TEXT,
    last_success_at TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0
);
