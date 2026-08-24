"""Catalog storage, candidate generation, and the player index."""

from __future__ import annotations

from cardid.catalog.store import CatalogStore, build_display_name, title_case
from cardid.models import CardAttributes, CatalogCard


def test_seed_catalog_loads(store):
    assert store.count() == 6
    assert store.get("mahomes_silver").parallel == "silver"


def test_candidates_found_by_player_and_set(store):
    query = CardAttributes(player="patrick mahomes ii", set_name="prizm", year=2017)
    ids = {card.card_id for card in store.candidates(query)}
    assert {"mahomes_base", "mahomes_silver", "mahomes_gold"} <= ids


def test_candidates_found_by_card_number_and_set(store):
    query = CardAttributes(card_number="101", set_name="contenders", year=2018)
    assert "allen_ticket" in {card.card_id for card in store.candidates(query)}


def test_unknown_player_yields_no_candidates(store):
    assert store.candidates(CardAttributes(player="nobody at all")) == []


def test_player_index_maps_unambiguous_surnames(store):
    index = dict(store.player_index())
    assert index["mahomes"] == "patrick mahomes ii"
    assert index["burrow"] == "joe burrow"


def test_player_index_skips_ambiguous_surnames():
    catalog = CatalogStore(":memory:")
    catalog.add_cards([
        CatalogCard(card_id="a", player="devin smith"),
        CatalogCard(card_id="b", player="jimmy smith"),
    ])
    index = dict(catalog.player_index())
    assert "smith" not in index  # two players share it, so it identifies nobody


def test_add_cards_replaces_rather_than_duplicates(store):
    before = store.count()
    store.add_cards([CatalogCard(card_id="mahomes_base", player="patrick mahomes ii",
                                 year=2017, set_name="prizm", card_number="269")])
    assert store.count() == before


def test_quotes_in_a_name_do_not_break_full_text_search(store):
    # An apostrophe would otherwise be an FTS5 syntax error.
    assert store.candidates(CardAttributes(player="ja'marr chase")) == []


def test_display_name_keeps_suffix_casing():
    name = build_display_name(CatalogCard(card_id="x", year=2017, brand="panini",
                                          set_name="prizm", player="patrick mahomes ii",
                                          card_number="269", is_rookie=True))
    assert "Patrick Mahomes II" in name
    assert "#269" in name


def test_title_case_handles_suffixes():
    assert title_case("odell beckham jr") == "Odell Beckham Jr."
    assert title_case("national treasures") == "National Treasures"


def test_title_case_handles_initial_style_first_names():
    """CJ Stroud and DK Metcalf must not render as "Cj" and "Dk"."""
    assert title_case("cj stroud") == "CJ Stroud"
    assert title_case("dk metcalf") == "DK Metcalf"
    assert title_case("aj brown") == "AJ Brown"


def test_title_case_does_not_shout_short_real_names():
    assert title_case("bo jackson") == "Bo Jackson"
    assert title_case("ed reed") == "Ed Reed"


def test_title_case_keeps_suffixes_over_the_initials_rule():
    # "jr" has no vowel but is a suffix, not initials.
    assert title_case("odell beckham jr") == "Odell Beckham Jr."
