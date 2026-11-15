"""SQLite-backed authentication, conversation ownership, titles, and long-term memory.

Designed for local development. Passwords are stored as salted PBKDF2-HMAC-SHA256
hashes; plaintext passwords are never written to the database.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Generator

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
_db_setting = os.getenv("APP_DB_PATH", "chatbot_app.db")
APP_DB_PATH = Path(_db_setting)
if not APP_DB_PATH.is_absolute():
    APP_DB_PATH = BASE_DIR / APP_DB_PATH
APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

PBKDF2_ITERATIONS = 600_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@contextmanager
def _db() -> Generator[sqlite3.Connection]:
    conn = sqlite3.connect(str(APP_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create local application tables without deleting existing SQLite data."""
    with _db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_user (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated "
            "ON conversations(user_id, updated_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS long_term_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
                memory_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, memory_text)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_updated "
            "ON long_term_memories(user_id, updated_at DESC)"
        )


init_db()


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )


def register_user(
    email: str,
    name: str,
    password: str,
) -> tuple[bool, str, dict[str, str] | None]:
    email = (email or "").strip().lower()
    name = (name or "").strip()
    password = password or ""

    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return False, "Enter a valid email address.", None
    if not 2 <= len(name) <= 80:
        return False, "Name must be between 2 and 80 characters.", None
    if len(password) < 10:
        return False, "Password must be at least 10 characters long.", None
    if len(password) > 1024:
        return False, "Password is too long.", None

    salt = secrets.token_bytes(16)
    digest = _password_hash(password, salt)
    user = {"user_id": str(uuid.uuid4()), "email": email, "name": name}
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO app_user "
                "(user_id, email, name, password_salt, password_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user["user_id"], email, name, salt, digest, _now()),
            )
        return True, "Account created. You can now log in.", user
    except sqlite3.IntegrityError as exc:
        if "email" in str(exc).lower() or "unique" in str(exc).lower():
            return False, "An account with this email already exists.", None
        raise


def authenticate_user(email: str, password: str) -> dict[str, str] | None:
    email = (email or "").strip().lower()
    with _db() as conn:
        row = conn.execute(
            "SELECT user_id, email, name, password_salt, password_hash "
            "FROM app_user WHERE email = ?",
            (email,),
        ).fetchone()

    if row:
        salt = bytes(row["password_salt"])
        expected = bytes(row["password_hash"])
    else:
        salt = b"chatbot-dummy-salt"
        expected = _password_hash("not-the-password", salt)

    actual = _password_hash(password or "", salt)
    if not row or not hmac.compare_digest(actual, expected):
        return None

    return {"user_id": row["user_id"], "email": row["email"], "name": row["name"]}


def create_conversation(user_id: str, thread_id: str, title: str = "New Chat") -> None:
    now = _now()
    with _db() as conn:
        conn.execute(
            "INSERT INTO conversations(thread_id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, user_id, (title or "New Chat")[:80], now, now),
        )


def assert_thread_owner(user_id: str | None, thread_id: str | None) -> None:
    if not user_id or not thread_id:
        raise PermissionError("Missing user or conversation identity.")
    with _db() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE user_id = ? AND thread_id = ?",
            (user_id, thread_id),
        ).fetchone()
    if not row:
        raise PermissionError("This conversation does not belong to the signed-in user.")


def list_conversation(user_id: str) -> list[dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation_title(user_id: str, thread_id: str) -> str | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT title FROM conversations WHERE user_id = ? AND thread_id = ?",
            (user_id, thread_id),
        ).fetchone()
    return row["title"] if row else None


def save_conversation_title(user_id: str, thread_id: str, title: str) -> None:
    title = (title or "New Chat").strip()[:80] or "New Chat"
    with _db() as conn:
        result = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? "
            "WHERE user_id = ? AND thread_id = ?",
            (title, _now(), user_id, thread_id),
        )
        if result.rowcount != 1:
            raise PermissionError("Cannot update a conversation you do not own.")


def touch_conversation(user_id: str, thread_id: str) -> None:
    with _db() as conn:
        result = conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE user_id = ? AND thread_id = ?",
            (_now(), user_id, thread_id),
        )
        if result.rowcount != 1:
            raise PermissionError("Cannot access a conversation you do not own.")


def save_memory(user_id: str, memory_text: str) -> None:
    memory_text = " ".join((memory_text or "").split()).strip()[:1000]
    if not memory_text:
        return
    now = _now()
    with _db() as conn:
        conn.execute(
            "INSERT INTO long_term_memories "
            "(memory_id, user_id, memory_text, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, memory_text) DO UPDATE SET updated_at = excluded.updated_at",
            (str(uuid.uuid4()), user_id, memory_text, now, now),
        )
        conn.execute(
            "DELETE FROM long_term_memories WHERE user_id = ? AND memory_id IN ("
            "SELECT memory_id FROM long_term_memories WHERE user_id = ? "
            "ORDER BY updated_at DESC, created_at DESC LIMIT -1 OFFSET 50)",
            (user_id, user_id),
        )


def get_memories(user_id: str, limit: int = 20) -> list[str]:
    safe_limit = max(1, min(int(limit), 50))
    with _db() as conn:
        rows = conn.execute(
            "SELECT memory_text FROM long_term_memories WHERE user_id = ? "
            "ORDER BY updated_at DESC, created_at DESC LIMIT ?",
            (user_id, safe_limit),
        ).fetchall()
    return [row["memory_text"] for row in rows]


def forget_memories(user_id: str, query: str | None = None) -> int:
    with _db() as conn:
        if not query or query.strip().lower() in {"all", "everything", "all memories"}:
            result = conn.execute(
                "DELETE FROM long_term_memories WHERE user_id = ?",
                (user_id,),
            )
        else:
            result = conn.execute(
                "DELETE FROM long_term_memories "
                "WHERE user_id = ? AND memory_text LIKE ? COLLATE NOCASE",
                (user_id, f"%{query.strip()}%"),
            )
    return result.rowcount
