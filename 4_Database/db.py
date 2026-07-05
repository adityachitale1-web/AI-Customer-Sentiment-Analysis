"""Database layer for the Sentiment Analysis project (core requirement 4).

A single SQLite database (feedback.db, created next to this file) stores every
piece of feedback analysed by the API. The dashboard reads from the same file.

Production notes:
- WAL journal mode allows the API (writer) and dashboard (reader) to work
  concurrently without locking each other out.
- Indexes on created_at and sentiment keep trend queries fast as data grows.
- All queries are parameterised; user input never reaches SQL directly.
- The path can be overridden with the SENTIMENT_DB_PATH env var (used by tests
  and by Docker, where the DB lives on a mounted volume).
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.getenv("SENTIMENT_DB_PATH",
                         Path(__file__).resolve().parent / "feedback.db"))

VALID_SENTIMENTS = ("Positive", "Negative", "Neutral")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT    NOT NULL,
                sentiment   TEXT    NOT NULL CHECK (sentiment IN ('Positive', 'Negative', 'Neutral')),
                confidence  REAL    NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                source      TEXT    DEFAULT 'api',
                created_at  TEXT    NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback (created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_sentiment ON feedback (sentiment)")


def insert_feedback(text: str, sentiment: str, confidence: float,
                    source: str = "api", created_at: Optional[str] = None) -> int:
    if sentiment not in VALID_SENTIMENTS:
        raise ValueError(f"sentiment must be one of {VALID_SENTIMENTS}, got {sentiment!r}")
    created_at = created_at or datetime.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO feedback (text, sentiment, confidence, source, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (text, sentiment, round(float(confidence), 4), source, created_at),
        )
        return cur.lastrowid


def fetch_feedback(limit: Optional[int] = None, offset: int = 0) -> list:
    query = "SELECT * FROM feedback ORDER BY created_at DESC"
    params: tuple = ()
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params = (int(limit), int(offset))
    with get_connection() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def total_count() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]


def sentiment_counts() -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT sentiment, COUNT(*) AS n FROM feedback GROUP BY sentiment"
        ).fetchall()
    return {row["sentiment"]: row["n"] for row in rows}


def daily_counts() -> list:
    """Rows of (date, sentiment, count) for the dashboard trend chart."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT substr(created_at, 1, 10) AS date, sentiment, COUNT(*) AS n "
            "FROM feedback GROUP BY date, sentiment ORDER BY date"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_all() -> int:
    """Utility for reseeding demos/tests. Returns number of deleted rows."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM feedback")
        return cur.rowcount


if __name__ == "__main__":
    init_db()
    print(f"Database ready at {DB_PATH}")
    print("Current counts:", sentiment_counts() or "empty")
