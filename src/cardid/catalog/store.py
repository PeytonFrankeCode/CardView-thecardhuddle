"""SQLite-backed reference catalog.

The catalog is the authority on what a card actually is; matching maps a photo's
extracted attributes onto a row here. SQLite with an FTS5 index handles catalogs
into the low millions of rows on a single box, which covers every football card
ever printed with room to spare.

Candidate generation is deliberately layered: a tight structured query first,
then progressively looser full-text queries, stopping as soon as a layer returns
a workable candidate set. That keeps the expensive fuzzy scoring in the matcher
pointed at tens of rows instead of the whole table.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..models import CardAttributes, CatalogCard

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    card_id       TEXT PRIMARY KEY,
    year          INTEGER,
    brand         TEXT,
    set_name      TEXT,
    subset        TEXT,
    player        TEXT,
    team          TEXT,
    card_number   TEXT,
    parallel      TEXT,
    print_run     INTEGER,
    is_rookie     INTEGER DEFAULT 0,
    is_autograph  INTEGER DEFAULT 0,
    is_patch      INTEGER DEFAULT 0,
    display_name  TEXT,
    search_text   TEXT,
    image_phash   TEXT,
    embedding_id  TEXT,
    external_ids  TEXT
);
CREATE INDEX IF NOT EXISTS idx_cards_player_year ON cards(player, year);
CREATE INDEX IF NOT EXISTS idx_cards_number ON cards(card_number);
CREATE INDEX IF NOT EXISTS idx_cards_set ON cards(set_name, year);
CREATE INDEX IF NOT EXISTS idx_cards_phash ON cards(image_phash);

CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
    card_id UNINDEXED,
    search_text,
    tokenize='unicode61'
);
"""

_COLUMNS = (
    "card_id", "year", "brand", "set_name", "subset", "player", "team",
    "card_number", "parallel", "print_run", "is_rookie", "is_autograph",
    "is_patch", "display_name", "search_text", "image_phash", "embedding_id",
    "external_ids",
)

_FTS_UNSAFE = re.compile(r'["\']')


def build_search_text(card: CatalogCard) -> str:
    """The bag of words the FTS index sees for a card."""
    parts = [
        str(card.year or ""), card.brand or "", card.set_name or "",
        card.subset or "", card.player or "", card.team or "",
        card.card_number or "", card.parallel or "",
    ]
    if card.is_rookie:
        parts.append("rookie rc")
    if card.is_autograph:
        parts.append("auto autograph")
    if card.is_patch:
        parts.append("patch relic")
    if card.print_run:
        parts.append(f"/{card.print_run}")
    return " ".join(part for part in parts if part).lower()


_UPPER_WORDS = {"ii", "iii", "iv", "v", "jr", "sr", "rc", "sp", "ssp", "xr", "rpa"}

# Initial-style first names are common in football (CJ Stroud, DK Metcalf,
# TJ Watt) and must not render as "Cj". A short token with no vowel reads as
# initials; "aj" is listed explicitly because it carries one.
_VOWELS = set("aeiou")
_INITIAL_NAMES = {"aj"}


def _is_initials(word: str) -> bool:
    return len(word) in (2, 3) and (not _VOWELS & set(word) or word in _INITIAL_NAMES)


def title_case(value: str) -> str:
    """Title-case a card string while keeping suffixes like II and Jr. right."""
    words = []
    for word in value.split():
        lowered = word.lower()
        # Suffixes are checked first: "jr" has no vowel but is not initials.
        if lowered in _UPPER_WORDS:
            words.append("Jr." if lowered == "jr" else "Sr." if lowered == "sr" else word.upper())
        elif _is_initials(lowered):
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def build_display_name(card: CatalogCard) -> str:
    """Human-facing card name, the string the website ultimately shows."""
    parts: list[str] = []
    if card.year:
        parts.append(str(card.year))
    if card.brand and card.brand != card.set_name:
        parts.append(title_case(card.brand))
    if card.set_name:
        parts.append(title_case(card.set_name))
    if card.parallel and card.parallel != "base":
        parts.append(title_case(card.parallel))
    if card.player:
        parts.append(title_case(card.player))
    if card.card_number:
        parts.append(f"#{card.card_number}")
    flags = []
    if card.is_rookie:
        flags.append("RC")
    if card.is_autograph:
        flags.append("Auto")
    if card.is_patch:
        flags.append("Patch")
    if card.print_run:
        flags.append(f"/{card.print_run}")
    return " ".join(parts) + (f" {' '.join(flags)}" if flags else "")


class CatalogStore:
    """Read/write access to the reference catalog."""

    def __init__(self, db_path: str | Path = "data/catalog.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._player_index_cache: list[tuple[str, str]] | None = None

    # --- writing ---------------------------------------------------------

    def add_cards(self, cards: Iterable[CatalogCard]) -> int:
        """Insert or replace catalog rows, refreshing derived fields."""
        rows, fts_rows = [], []
        for card in cards:
            if not card.search_text:
                card.search_text = build_search_text(card)
            if not card.display_name:
                card.display_name = build_display_name(card)
            rows.append(
                (
                    card.card_id, card.year, card.brand, card.set_name,
                    card.subset, card.player, card.team, card.card_number,
                    card.parallel, card.print_run, int(card.is_rookie),
                    int(card.is_autograph), int(card.is_patch),
                    card.display_name, card.search_text, card.image_phash,
                    card.embedding_id, json.dumps(card.external_ids),
                )
            )
            fts_rows.append((card.card_id, card.search_text))

        placeholders = ", ".join("?" * len(_COLUMNS))
        with self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO cards ({', '.join(_COLUMNS)}) "
                f"VALUES ({placeholders})",
                rows,
            )
            # Keep FTS in sync without triggers: drop then re-add each card_id.
            self._conn.executemany(
                "DELETE FROM cards_fts WHERE card_id = ?",
                [(card_id,) for card_id, _ in fts_rows],
            )
            self._conn.executemany(
                "INSERT INTO cards_fts (card_id, search_text) VALUES (?, ?)",
                fts_rows,
            )
        self._player_index_cache = None
        return len(rows)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]

    # --- reading ---------------------------------------------------------

    def _to_card(self, row: sqlite3.Row) -> CatalogCard:
        return CatalogCard(
            card_id=row["card_id"], year=row["year"], brand=row["brand"],
            set_name=row["set_name"], subset=row["subset"], player=row["player"],
            team=row["team"], card_number=row["card_number"],
            parallel=row["parallel"], print_run=row["print_run"],
            is_rookie=bool(row["is_rookie"]),
            is_autograph=bool(row["is_autograph"]),
            is_patch=bool(row["is_patch"]),
            display_name=row["display_name"] or "",
            search_text=row["search_text"] or "",
            image_phash=row["image_phash"], embedding_id=row["embedding_id"],
            external_ids=json.loads(row["external_ids"] or "{}"),
        )

    def get(self, card_id: str) -> CatalogCard | None:
        row = self._conn.execute(
            "SELECT * FROM cards WHERE card_id = ?", (card_id,)
        ).fetchone()
        return self._to_card(row) if row else None

    def iter_cards(self) -> Iterator[CatalogCard]:
        for row in self._conn.execute("SELECT * FROM cards"):
            yield self._to_card(row)

    def by_phash(self, phash: str) -> list[CatalogCard]:
        rows = self._conn.execute(
            "SELECT * FROM cards WHERE image_phash = ?", (phash,)
        ).fetchall()
        return [self._to_card(row) for row in rows]

    def _fts(self, query: str, limit: int) -> list[CatalogCard]:
        if not query.strip():
            return []
        try:
            rows = self._conn.execute(
                "SELECT c.* FROM cards_fts f JOIN cards c ON c.card_id = f.card_id "
                "WHERE cards_fts MATCH ? ORDER BY bm25(cards_fts) LIMIT ?",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS expression: treat as no candidates rather than 500.
            return []
        return [self._to_card(row) for row in rows]

    def candidates(self, attrs: CardAttributes, limit: int = 200) -> list[CatalogCard]:
        """Generate a candidate set for ``attrs``, widening until something hits."""
        found: dict[str, CatalogCard] = {}

        def absorb(cards: list[CatalogCard]) -> None:
            for card in cards:
                found.setdefault(card.card_id, card)

        player = _fts_term(attrs.player)
        set_term = _fts_term(attrs.set_name)
        year_term = str(attrs.year) if attrs.year else ""

        # Layer 1: structured lookup on the most selective combination we have.
        if attrs.card_number and (attrs.set_name or attrs.year):
            clauses = ["card_number = ?"]
            params: list[object] = [attrs.card_number.upper()]
            if attrs.set_name:
                clauses.append("set_name = ?")
                params.append(attrs.set_name)
            if attrs.year:
                clauses.append("year = ?")
                params.append(attrs.year)
            params.append(limit)
            rows = self._conn.execute(
                f"SELECT * FROM cards WHERE {' AND '.join(clauses)} LIMIT ?",
                params,
            ).fetchall()
            absorb([self._to_card(row) for row in rows])

        # Layer 2..4: full text, tightest first.
        for query in (
            " AND ".join(t for t in (player, set_term, year_term) if t),
            " AND ".join(t for t in (player, set_term) if t),
            player,
        ):
            if len(found) >= limit or not query:
                continue
            absorb(self._fts(query, limit))
            if len(found) >= 3:
                break

        # Layer 5: last resort, OR everything informative we parsed.
        if not found:
            terms = [
                t for t in (player, set_term, year_term, _fts_term(attrs.brand))
                if t
            ]
            if terms:
                absorb(self._fts(" OR ".join(terms), limit))

        return list(found.values())[:limit]

    # --- player index for the parser --------------------------------------

    def player_index(self) -> list[tuple[str, str]]:
        """Build ``(alias, canonical_player)`` pairs from catalog players.

        Full names always map to themselves. A surname is added only when it is
        unambiguous in the catalog, so "mahomes" resolves but "smith" does not.
        """
        if self._player_index_cache is not None:
            return self._player_index_cache

        rows = self._conn.execute(
            "SELECT DISTINCT player FROM cards WHERE player IS NOT NULL AND player != ''"
        ).fetchall()
        players = [row[0] for row in rows]

        by_surname: dict[str, set[str]] = defaultdict(set)
        for player in players:
            parts = player.split()
            if len(parts) >= 2:
                surname = parts[-1]
                if surname in {"jr", "sr", "ii", "iii", "iv", "v"} and len(parts) >= 3:
                    surname = parts[-2]
                by_surname[surname].add(player)

        pairs = [(player, player) for player in players]
        for surname, owners in by_surname.items():
            if len(owners) == 1 and len(surname) > 3:
                pairs.append((surname, next(iter(owners))))

        pairs.sort(key=lambda pair: (-len(pair[0]), pair[0]))
        self._player_index_cache = pairs
        return pairs

    def close(self) -> None:
        self._conn.close()


def _fts_term(value: str | None) -> str:
    """Quote a value as a single FTS5 phrase, or return '' if unusable."""
    if not value:
        return ""
    cleaned = _FTS_UNSAFE.sub(" ", value).strip()
    return f'"{cleaned}"' if cleaned else ""
