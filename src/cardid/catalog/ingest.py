"""Getting cards into the catalog.

Two paths, because most people start without a clean catalog:

``import_rows`` takes a structured export (CSV or JSON) from a card database and
loads it directly. This is the better option when you have one.

``bootstrap_from_titles`` builds a catalog out of your own sold-listing titles.
It parses each title, groups titles that describe the same card, and emits one
catalog row per group. The result is not as clean as a real card database, but
it is derived from exactly the population you need to match against, and it
turns a pile of unsorted sold data into something the matcher can work with on
day one.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..models import CardAttributes, CatalogCard, Source
from ..pipeline.parse import canonical_player, parse_text
from .store import CatalogStore, build_display_name, build_search_text

# Fields that together identify a distinct card.
IDENTITY_FIELDS = ("year", "brand", "set_name", "player", "card_number", "parallel")


def make_card_id(attrs: CardAttributes) -> str:
    """Stable id derived from the identifying fields.

    Deterministic so re-running an import updates rows instead of duplicating
    them, and so two sources describing the same card converge on one id.
    """
    parts = []
    for field in IDENTITY_FIELDS:
        value = getattr(attrs, field)
        if field == "player":
            parts.append(canonical_player(value))
        elif field == "parallel":
            # An unstated parallel means the base card.
            parts.append(str(value or "base").strip().lower())
        else:
            parts.append(str(value or "").strip().lower())
    if attrs.is_autograph:
        parts.append("auto")
    if attrs.is_patch:
        parts.append("patch")
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"cid_{digest}"


def card_from_attributes(attrs: CardAttributes, card_id: str | None = None) -> CatalogCard:
    card = CatalogCard(
        card_id=card_id or make_card_id(attrs),
        year=attrs.year,
        brand=attrs.brand,
        set_name=attrs.set_name,
        subset=attrs.subset,
        player=attrs.player,
        team=attrs.team,
        card_number=attrs.card_number,
        parallel=attrs.parallel or "base",
        print_run=attrs.print_run,
        is_rookie=attrs.is_rookie,
        is_autograph=attrs.is_autograph,
        is_patch=attrs.is_patch,
    )
    card.search_text = build_search_text(card)
    card.display_name = build_display_name(card)
    return card


# --- structured import ------------------------------------------------------

_TRUE = {"1", "true", "yes", "y", "t"}


def _coerce_row(row: dict[str, str]) -> CatalogCard:
    def text(key: str) -> str | None:
        value = (row.get(key) or "").strip()
        return value.lower() or None

    def number(key: str) -> int | None:
        value = (row.get(key) or "").strip()
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def flag(key: str) -> bool:
        return (row.get(key) or "").strip().lower() in _TRUE

    attrs = CardAttributes(
        year=number("year"),
        brand=text("brand"),
        set_name=text("set_name") or text("set"),
        subset=text("subset"),
        player=text("player") or text("player_name"),
        team=text("team"),
        card_number=(row.get("card_number") or row.get("number") or "").strip().upper() or None,
        parallel=text("parallel") or text("variation"),
        print_run=number("print_run"),
        is_rookie=flag("is_rookie") or flag("rookie"),
        is_autograph=flag("is_autograph") or flag("auto"),
        is_patch=flag("is_patch") or flag("patch"),
    )
    card_id = (row.get("card_id") or "").strip() or None
    card = card_from_attributes(attrs, card_id=card_id)
    external = (row.get("external_id") or "").strip()
    if external:
        card.external_ids["source"] = external
    return card


def read_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Read a catalog export from .csv, .json, or .jsonl."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        rows = data if isinstance(data, list) else data.get("cards", [])
        yield from rows
    else:
        raise ValueError(f"unsupported catalog format: {suffix}")


def import_rows(store: CatalogStore, path: str | Path, batch_size: int = 5000) -> int:
    """Load a structured catalog export into the store."""
    total = 0
    batch: list[CatalogCard] = []
    for row in read_rows(path):
        batch.append(_coerce_row({str(k): str(v) if v is not None else "" for k, v in row.items()}))
        if len(batch) >= batch_size:
            total += store.add_cards(batch)
            batch = []
    if batch:
        total += store.add_cards(batch)
    return total


# --- bootstrap from sold titles --------------------------------------------

def bootstrap_from_titles(
    store: CatalogStore,
    titles: Iterable[str],
    min_occurrences: int = 1,
    require_player: bool = True,
) -> dict[str, int]:
    """Derive a catalog from raw sold-listing titles.

    Titles that parse to the same identifying fields collapse into one card.
    ``min_occurrences`` filters out one-off parses, which are usually typos or
    junk listings; raise it when working from a large feed.
    """
    groups: dict[str, list[CardAttributes]] = defaultdict(list)
    stats = Counter()

    for title in titles:
        stats["seen"] += 1
        attrs = parse_text(title, Source.TITLE).attributes
        if require_player and not attrs.player:
            stats["no_player"] += 1
            continue
        if not attrs.year and not attrs.set_name:
            stats["too_vague"] += 1
            continue
        # Grading is a property of the individual slab, not of the card.
        attrs.grader = None
        attrs.grade = None
        attrs.serial = None
        groups[make_card_id(attrs)].append(attrs)

    cards: list[CatalogCard] = []
    for card_id, members in groups.items():
        if len(members) < min_occurrences:
            stats["below_threshold"] += 1
            continue
        merged = members[0]
        for other in members[1:]:
            merged = merged.merge(other)
        # Members agree on identity but may spell the player differently; keep
        # the spelling the feed used most often, longest wins a tie.
        spellings = Counter(m.player for m in members if m.player)
        if spellings:
            merged.player = max(
                spellings.items(), key=lambda kv: (kv[1], len(kv[0]))
            )[0]
        card = card_from_attributes(merged, card_id=card_id)
        card.external_ids["observed"] = str(len(members))
        cards.append(card)

    if cards:
        store.add_cards(cards)
    stats["cards_created"] = len(cards)
    return dict(stats)
