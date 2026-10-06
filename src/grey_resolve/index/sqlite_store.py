"""SQLite metadata sidecar keyed by integer vector id.

One row per vector id holds the per-vector payload (media id, quality score,
context metadata, ingest time). The vector index stays pure; all payload access
lives here so index and sidecar can be versioned together (see ARCHITECTURE.md).

Single sqlite3 connection per store instance; safe for single-process use.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from grey_resolve.types import ContextMetadata

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    vector_id INTEGER PRIMARY KEY,
    media_id TEXT NOT NULL,
    quality_score REAL NOT NULL,
    timestamp TEXT,
    geo_cluster TEXT,
    source_platform TEXT,
    text_entities TEXT NOT NULL,
    ingest_time TEXT NOT NULL
)
"""

_UPSERT = """
INSERT INTO metadata (
    vector_id, media_id, quality_score, timestamp,
    geo_cluster, source_platform, text_entities, ingest_time
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(vector_id) DO UPDATE SET
    media_id = excluded.media_id,
    quality_score = excluded.quality_score,
    timestamp = excluded.timestamp,
    geo_cluster = excluded.geo_cluster,
    source_platform = excluded.source_platform,
    text_entities = excluded.text_entities,
    ingest_time = excluded.ingest_time
"""

_COLUMNS = (
    "media_id, quality_score, timestamp, "
    "geo_cluster, source_platform, text_entities, ingest_time"
)


class SqliteMetadataStore:
    """Metadata sidecar over sqlite3 (stdlib). Implements ``MetadataStore``.

    ``get()`` / ``get_many()`` return plain dicts with decoded values::

        media_id (str), quality_score (float),
        timestamp (datetime | None), geo_cluster (str | None),
        source_platform (str | None), text_entities (tuple[str, ...]),
        ingest_time (datetime)

    ``ingest_time`` is stamped at ``put()`` time (UTC, ISO 8601 in storage).
    Re-``put()``-ing a vector id replaces its row and restamps ``ingest_time``.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        try:
            with self._conn:
                self._conn.execute(_SCHEMA)
        except Exception:
            self._conn.close()
            raise

    # -- MetadataStore API --------------------------------------------------

    def put(
        self,
        vector_id: int,
        media_id: str,
        context: ContextMetadata,
        quality_score: float,
    ) -> None:
        """Insert or replace the payload row for ``vector_id``."""
        row = (
            int(vector_id),
            str(media_id),
            float(quality_score),
            context.timestamp.isoformat() if context.timestamp is not None else None,
            context.geo_cluster,
            context.source_platform,
            json.dumps(list(context.text_entities)),
            datetime.now(UTC).isoformat(),
        )
        with self._conn:
            self._conn.execute(_UPSERT, row)

    def get(self, vector_id: int) -> dict | None:
        """Return the payload dict for ``vector_id``, or None if absent."""
        cursor = self._conn.execute(
            f"SELECT {_COLUMNS} FROM metadata WHERE vector_id = ?", (int(vector_id),)
        )
        row = cursor.fetchone()
        return _decode_row(row) if row is not None else None

    def get_many(self, vector_ids: list[int]) -> dict[int, dict]:
        """Return {vector_id: payload dict} for the ids present in the store.

        Missing ids are omitted. Input order is preserved for present ids.
        """
        if not vector_ids:
            return {}
        wanted = [int(v) for v in vector_ids]
        placeholders = ", ".join("?" for _ in wanted)
        cursor = self._conn.execute(
            f"SELECT vector_id, {_COLUMNS} FROM metadata WHERE vector_id IN ({placeholders})",
            wanted,
        )
        found = {int(row[0]): _decode_row(row[1:]) for row in cursor.fetchall()}
        return {vid: found[vid] for vid in wanted if vid in found}

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Commit pending work and close the underlying connection."""
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _decode_row(row: tuple) -> dict:
    """Decode a raw ``metadata`` row tuple into a plain payload dict."""
    media_id, quality_score, timestamp, geo_cluster, source_platform, text_entities, ingest_time = row
    return {
        "media_id": media_id,
        "quality_score": float(quality_score),
        "timestamp": datetime.fromisoformat(timestamp) if timestamp is not None else None,
        "geo_cluster": geo_cluster,
        "source_platform": source_platform,
        "text_entities": tuple(json.loads(text_entities)),
        "ingest_time": datetime.fromisoformat(ingest_time),
    }
