"""Review queue and label store.

Anything the pipeline is not sure about lands here instead of being guessed at.
That is what makes a precision target meaningful: the auto-accepted slice can be
held to a high standard precisely because the uncertain slice has somewhere else
to go.

Resolutions are not thrown away. Every human decision is written to a labels
table, which is both the feedback that seeds the cache for future matches and
the ground truth that scripts/calibrate.py needs to tune thresholds.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .models import Identification

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_queue (
    request_id    TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    confidence    REAL,
    image_phash   TEXT,
    image_url     TEXT,
    title         TEXT,
    predicted_id  TEXT,
    payload       TEXT NOT NULL,
    resolved_at   REAL,
    resolved_by   TEXT,
    resolved_id   TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status, created_at);

CREATE TABLE IF NOT EXISTS labels (
    label_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   REAL NOT NULL,
    image_phash  TEXT,
    image_url    TEXT,
    title        TEXT,
    true_card_id TEXT NOT NULL,
    predicted_id TEXT,
    confidence   REAL,
    was_correct  INTEGER,
    origin       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_labels_phash ON labels(image_phash);
"""


@dataclass
class ReviewItem:
    request_id: str
    created_at: float
    status: str
    confidence: float | None
    image_phash: str | None
    image_url: str | None
    title: str | None
    predicted_id: str | None
    result: Identification


class ReviewStore:
    def __init__(self, db_path: str | Path = "data/review.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def enqueue(
        self,
        result: Identification,
        image_url: str | None = None,
        title: str | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO review_queue "
                "(request_id, created_at, status, confidence, image_phash, image_url, "
                " title, predicted_id, payload) "
                "VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)",
                (
                    result.request_id,
                    time.time(),
                    result.confidence,
                    result.image_phash,
                    image_url,
                    title,
                    result.card.card_id if result.card else None,
                    result.model_dump_json(),
                ),
            )

    def _to_item(self, row: sqlite3.Row) -> ReviewItem:
        return ReviewItem(
            request_id=row["request_id"],
            created_at=row["created_at"],
            status=row["status"],
            confidence=row["confidence"],
            image_phash=row["image_phash"],
            image_url=row["image_url"],
            title=row["title"],
            predicted_id=row["predicted_id"],
            result=Identification.model_validate(json.loads(row["payload"])),
        )

    def pending(self, limit: int = 50, offset: int = 0) -> list[ReviewItem]:
        rows = self._conn.execute(
            "SELECT * FROM review_queue WHERE status = 'pending' "
            "ORDER BY confidence DESC, created_at ASC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._to_item(row) for row in rows]

    def get(self, request_id: str) -> ReviewItem | None:
        row = self._conn.execute(
            "SELECT * FROM review_queue WHERE request_id = ?", (request_id,)
        ).fetchone()
        return self._to_item(row) if row else None

    def resolve(self, request_id: str, true_card_id: str, resolved_by: str = "human") -> bool:
        """Record a human decision and turn it into a training label."""
        item = self.get(request_id)
        if item is None:
            return False
        now = time.time()
        with self._conn:
            self._conn.execute(
                "UPDATE review_queue SET status = 'resolved', resolved_at = ?, "
                "resolved_by = ?, resolved_id = ? WHERE request_id = ?",
                (now, resolved_by, true_card_id, request_id),
            )
            self._conn.execute(
                "INSERT INTO labels (created_at, image_phash, image_url, title, "
                "true_card_id, predicted_id, confidence, was_correct, origin) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'review')",
                (
                    now,
                    item.image_phash,
                    item.image_url,
                    item.title,
                    true_card_id,
                    item.predicted_id,
                    item.confidence,
                    int(item.predicted_id == true_card_id),
                    ),
            )
        return True

    def add_label(
        self,
        true_card_id: str,
        image_phash: str | None = None,
        image_url: str | None = None,
        title: str | None = None,
        predicted_id: str | None = None,
        confidence: float | None = None,
        origin: str = "import",
    ) -> None:
        """Record ground truth from outside the review queue (e.g. a seed set)."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO labels (created_at, image_phash, image_url, title, "
                "true_card_id, predicted_id, confidence, was_correct, origin) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(), image_phash, image_url, title, true_card_id,
                    predicted_id, confidence,
                    None if predicted_id is None else int(predicted_id == true_card_id),
                    origin,
                ),
            )

    def known_by_phash(self, image_phash: str) -> str | None:
        """A card id a human already confirmed for this exact image."""
        row = self._conn.execute(
            "SELECT true_card_id FROM labels WHERE image_phash = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (image_phash,),
        ).fetchone()
        return row["true_card_id"] if row else None

    def labels(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM labels ORDER BY created_at"
        ).fetchall()

    def stats(self) -> dict[str, int]:
        pending = self._conn.execute(
            "SELECT COUNT(*) FROM review_queue WHERE status = 'pending'"
        ).fetchone()[0]
        resolved = self._conn.execute(
            "SELECT COUNT(*) FROM review_queue WHERE status = 'resolved'"
        ).fetchone()[0]
        labels = self._conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        return {"pending": pending, "resolved": resolved, "labels": labels}

    def close(self) -> None:
        self._conn.close()
