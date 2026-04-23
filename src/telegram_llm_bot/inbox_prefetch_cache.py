"""Persistent SQLite cache for prefetched Inbox context sources."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersistentInboxPrefetchRecord:
    source_id: int
    original_source_json: str
    hydrated_source_json: str
    enhanced: bool
    initial_reply: str | None
    cached_at: float
    summary_signature: str


class InboxPrefetchPersistentCache:
    def __init__(
        self,
        *,
        raw_path: str | Path | None,
        ttl_seconds: float,
        base_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self._raw_path = raw_path
        self._ttl_seconds = ttl_seconds
        self._base_dir = base_dir
        self._logger = logger or logging.getLogger(__name__)

    def resolve_path(self) -> Path | None:
        raw_path = self._raw_path
        if raw_path is None:
            return None

        if isinstance(raw_path, Path):
            cache_path = raw_path.expanduser()
        else:
            stripped = str(raw_path).strip()
            if not stripped:
                return None
            cache_path = Path(stripped).expanduser()

        if not cache_path.is_absolute():
            cache_path = self._base_dir / cache_path
        return cache_path

    def _connect(self) -> sqlite3.Connection | None:
        cache_path = self.resolve_path()
        if cache_path is None:
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(cache_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inbox_context_prefetch_cache (
                source_id INTEGER PRIMARY KEY,
                original_source_json TEXT NOT NULL,
                hydrated_source_json TEXT NOT NULL,
                enhanced INTEGER NOT NULL,
                initial_reply TEXT,
                cached_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                summary_signature TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS inbox_context_prefetch_cache_expires_at_idx
            ON inbox_context_prefetch_cache (expires_at)
            """
        )
        return connection

    def load(self, source_id: int, *, now: float | None = None) -> PersistentInboxPrefetchRecord | None:
        try:
            connection = self._connect()
        except Exception as exc:
            self._logger.warning("Inbox context prefetch cache open failed source_id=%s: %s", source_id, exc)
            return None

        if connection is None:
            return None

        try:
            row = connection.execute(
                """
                SELECT original_source_json, hydrated_source_json, enhanced, initial_reply, cached_at, summary_signature
                FROM inbox_context_prefetch_cache
                WHERE source_id = ? AND expires_at > ?
                """,
                (source_id, now if now is not None else time.time()),
            ).fetchone()
            if row is None:
                return None

            return PersistentInboxPrefetchRecord(
                source_id=source_id,
                original_source_json=row["original_source_json"],
                hydrated_source_json=row["hydrated_source_json"],
                enhanced=bool(row["enhanced"]),
                initial_reply=row["initial_reply"],
                cached_at=float(row["cached_at"]),
                summary_signature=row["summary_signature"],
            )
        except Exception as exc:
            self._logger.warning("Inbox context prefetch cache load failed source_id=%s: %s", source_id, exc)
            return None
        finally:
            connection.close()

    def save(self, record: PersistentInboxPrefetchRecord) -> None:
        try:
            connection = self._connect()
        except Exception as exc:
            self._logger.warning("Inbox context prefetch cache open failed source_id=%s: %s", record.source_id, exc)
            return

        if connection is None:
            return

        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO inbox_context_prefetch_cache (
                        source_id,
                        original_source_json,
                        hydrated_source_json,
                        enhanced,
                        initial_reply,
                        cached_at,
                        expires_at,
                        summary_signature
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET
                        original_source_json = excluded.original_source_json,
                        hydrated_source_json = excluded.hydrated_source_json,
                        enhanced = excluded.enhanced,
                        initial_reply = excluded.initial_reply,
                        cached_at = excluded.cached_at,
                        expires_at = excluded.expires_at,
                        summary_signature = excluded.summary_signature
                    """,
                    (
                        record.source_id,
                        record.original_source_json,
                        record.hydrated_source_json,
                        int(record.enhanced),
                        record.initial_reply,
                        record.cached_at,
                        record.cached_at + self._ttl_seconds,
                        record.summary_signature,
                    ),
                )
        except Exception as exc:
            self._logger.warning("Inbox context prefetch cache write failed source_id=%s: %s", record.source_id, exc)
        finally:
            connection.close()

    def delete(self, source_id: int) -> None:
        try:
            connection = self._connect()
        except Exception as exc:
            self._logger.warning("Inbox context prefetch cache open failed source_id=%s: %s", source_id, exc)
            return

        if connection is None:
            return

        try:
            with connection:
                connection.execute(
                    "DELETE FROM inbox_context_prefetch_cache WHERE source_id = ?",
                    (source_id,),
                )
        except Exception as exc:
            self._logger.warning("Inbox context prefetch cache delete failed source_id=%s: %s", source_id, exc)
        finally:
            connection.close()

    def purge_stale(self, *, now: float | None = None) -> None:
        try:
            connection = self._connect()
        except Exception as exc:
            self._logger.warning("Inbox context prefetch cache open failed during purge: %s", exc)
            return

        if connection is None:
            return

        try:
            with connection:
                connection.execute(
                    "DELETE FROM inbox_context_prefetch_cache WHERE expires_at <= ?",
                    (now if now is not None else time.time(),),
                )
        except Exception as exc:
            self._logger.warning("Inbox context prefetch cache purge failed: %s", exc)
        finally:
            connection.close()

    def clear(self) -> None:
        try:
            connection = self._connect()
        except Exception:
            return

        if connection is None:
            return

        try:
            with connection:
                connection.execute("DELETE FROM inbox_context_prefetch_cache")
        finally:
            connection.close()
