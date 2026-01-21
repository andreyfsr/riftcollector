"""Database connection and schema management."""

import sqlite3

from flask import current_app, g


def ensure_collection_schema(conn: sqlite3.Connection) -> None:
    """Create the collection tables if they don't exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            sub TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            picture TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS user_collection (
            user_sub TEXT NOT NULL,
            card_key TEXT NOT NULL,
            name TEXT,
            image_url TEXT,
            count INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_sub, card_key),
            FOREIGN KEY (user_sub) REFERENCES users(sub) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_user_collection_user
            ON user_collection(user_sub);
        """
    )


def get_db() -> sqlite3.Connection:
    """Get or create database connection for the current request."""
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON;")
        ensure_collection_schema(g.db)
    return g.db


def close_db(e=None):
    """Close database connection at the end of request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    """Register database teardown with the app."""
    app.teardown_appcontext(close_db)
