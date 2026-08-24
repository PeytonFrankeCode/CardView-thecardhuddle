"""Catalog import and bootstrapping a catalog from sold-listing titles."""

from __future__ import annotations

import json

from cardid.catalog.ingest import (
    bootstrap_from_titles,
    import_rows,
    make_card_id,
)
from cardid.catalog.store import CatalogStore
from cardid.models import CardAttributes

TITLES = [
    "2017 Panini Prizm Patrick Mahomes II #269 RC Silver Prizm PSA 10",
    "2017 Prizm Patrick Mahomes Silver #269 Rookie BGS 9.5",
    "2017 Panini Prizm Patrick Mahomes II RC #269",
    "2018 Panini Contenders Josh Allen Rookie Ticket Auto #101 Bills",
    "junk lot of football cards read description",
]


def test_bootstrap_groups_titles_describing_the_same_card():
    catalog = CatalogStore(":memory:")
    stats = bootstrap_from_titles(catalog, TITLES)
    # Silver x2 collapse into one row; base and the Allen auto are their own.
    assert stats["cards_created"] == 3
    assert catalog.count() == 3


def test_bootstrap_ignores_titles_with_no_player():
    catalog = CatalogStore(":memory:")
    stats = bootstrap_from_titles(catalog, ["lot of 100 football cards"])
    assert stats["cards_created"] == 0
    assert stats["no_player"] == 1


def test_bootstrap_records_how_often_each_card_was_seen():
    catalog = CatalogStore(":memory:")
    bootstrap_from_titles(catalog, TITLES)
    observed = {card.display_name: card.external_ids.get("observed")
                for card in catalog.iter_cards()}
    assert "2" in observed.values()


def test_min_occurrences_filters_one_off_parses():
    catalog = CatalogStore(":memory:")
    stats = bootstrap_from_titles(catalog, TITLES, min_occurrences=2)
    assert stats["cards_created"] == 1


def test_grading_is_not_part_of_card_identity():
    """A PSA 10 and a BGS 9.5 of the same card are one card, not two."""
    catalog = CatalogStore(":memory:")
    bootstrap_from_titles(catalog, [
        "2017 Prizm Patrick Mahomes II Silver #269 PSA 10",
        "2017 Prizm Patrick Mahomes II Silver #269 BGS 9.5",
    ])
    assert catalog.count() == 1


def test_card_id_is_stable_across_spellings():
    left = make_card_id(CardAttributes(player="patrick mahomes ii", year=2017,
                                       set_name="prizm", card_number="269"))
    right = make_card_id(CardAttributes(player="Patrick Mahomes", year=2017,
                                        set_name="prizm", card_number="269"))
    assert left == right


def test_card_id_separates_parallels():
    base = make_card_id(CardAttributes(player="joe burrow", year=2020, card_number="307"))
    gold = make_card_id(CardAttributes(player="joe burrow", year=2020,
                                       card_number="307", parallel="gold"))
    assert base != gold


def test_import_rows_from_csv(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text(
        "year,brand,set_name,player,card_number,parallel,is_rookie\n"
        "2017,Panini,Prizm,Patrick Mahomes II,269,Silver,1\n"
        "2020,Panini,Prizm,Joe Burrow,307,,1\n",
        encoding="utf-8",
    )
    catalog = CatalogStore(":memory:")
    assert import_rows(catalog, path) == 2
    assert catalog.count() == 2


def test_import_rows_from_json(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps([
        {"card_id": "x1", "year": 2021, "brand": "Panini",
         "set_name": "National Treasures", "player": "Trevor Lawrence",
         "card_number": "115", "is_autograph": "true"},
    ]), encoding="utf-8")
    catalog = CatalogStore(":memory:")
    assert import_rows(catalog, path) == 1
    assert catalog.get("x1").is_autograph is True
