CREATE TABLE IF NOT EXISTS screening_cache (
    cache_key TEXT PRIMARY KEY,
    verdict_json TEXT NOT NULL CHECK (json_valid(verdict_json)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
