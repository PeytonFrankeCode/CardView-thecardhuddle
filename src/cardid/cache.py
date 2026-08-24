"""Result cache keyed by image content.

eBay listings reuse stock and seller photos constantly, and the same card is
relisted over and over. Hashing incoming images means a large share of a 20k/day
feed is answered without running OCR or matching at all.

Two keys are stored. The SHA-256 of the bytes catches byte-identical re-uploads.
The perceptual hash catches the same photo after a resize or re-encode, which is
what eBay's own image pipeline produces.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .models import Identification

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    key         TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    hits        INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_results_created ON results(created_at);
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ResultCache:
    """Stores completed identifications by content hash and perceptual hash."""

    def __init__(self, db_path: str | Path = "data/cache.db", enabled: bool = True) -> None:
        self.enabled = enabled
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _get(self, key: str, kind: str) -> Identification | None:
        row = self._conn.execute(
            "SELECT payload FROM results WHERE key = ? AND kind = ?", (key, kind)
        ).fetchone()
        if not row:
            return None
        with self._conn:
            self._conn.execute(
                "UPDATE results SET hits = hits + 1 WHERE key = ? AND kind = ?",
                (key, kind),
            )
        try:
            return Identification.model_validate(json.loads(row["payload"]))
        except Exception:
            # A stale row from an older schema should miss, not crash the request.
            return None

    def get(self, content_hash: str | None, image_phash: str | None) -> Identification | None:
        if not self.enabled:
            return None
        if content_hash:
            hit = self._get(content_hash, "sha256")
            if hit:
                return hit
        if image_phash:
            return self._get(image_phash, "phash")
        return None

    def put(
        self,
        result: Identification,
        content_hash: str | None,
        image_phash: str | None,
    ) -> None:
        """Cache a result. Only confident results are worth reusing."""
        if not self.enabled or result.decision.value == "reject":
            return
        payload = result.model_dump_json()
        now = time.time()
        rows = []
        if content_hash:
            rows.append((content_hash, "sha256", payload, now))
        if image_phash:
            rows.append((image_phash, "phash", payload, now))
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO results (key, kind, payload, created_at, hits) "
                "VALUES (?, ?, ?, ?, COALESCE((SELECT hits FROM results WHERE key = ?), 0))",
                [(key, kind, data, ts, key) for key, kind, data, ts in rows],
            )

    def stats(self) -> dict[str, int]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits FROM results"
        ).fetchone()
        return {"entries": row["entries"], "hits": row["hits"]}

    def purge_older_than(self, seconds: float) -> int:
        cutoff = time.time() - seconds
        with self._conn:
            cursor = self._conn.execute("DELETE FROM results WHERE created_at < ?", (cutoff,))
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()
