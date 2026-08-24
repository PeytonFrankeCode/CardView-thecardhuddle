"""Parsing eBay titles and card-face text into structured attributes."""

from __future__ import annotations

import pytest

from cardid.models import Source
from cardid.pipeline.parse import canonical_player, parse_text


def attrs(text: str, **kwargs):
    return parse_text(text, **kwargs).attributes


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "2017 Panini Prizm Patrick Mahomes II #269 Rookie RC Silver Prizm PSA 10",
            {"year": 2017, "brand": "panini", "set_name": "prizm",
             "player": "patrick mahomes ii", "card_number": "269",
             "parallel": "silver", "is_rookie": True, "grader": "psa", "grade": 10.0},
        ),
        (
            "Josh Allen 2018 Panini Contenders Rookie Ticket Auto RC #101 BGS 9.5 Bills",
            {"year": 2018, "set_name": "contenders", "player": "josh allen",
             "card_number": "101", "is_autograph": True, "grader": "bgs",
             "grade": 9.5, "team": "buffalo bills"},
        ),
        (
            "2021 National Treasures Trevor Lawrence RPA Patch Auto /99",
            {"year": 2021, "set_name": "national treasures",
             "player": "trevor lawrence", "print_run": 99,
             "is_autograph": True, "is_patch": True},
        ),
        (
            "2020 Donruss Optic Justin Herbert Rated Rookie #158",
            {"year": 2020, "set_name": "optic", "subset": "rated rookie",
             "player": "justin herbert", "card_number": "158", "is_rookie": True},
        ),
    ],
)
def test_titles_parse_to_expected_attributes(title, expected):
    parsed = attrs(title)
    for field, value in expected.items():
        assert getattr(parsed, field) == value, f"{field} on {title!r}"


def test_marketing_noise_does_not_become_a_player_name():
    parsed = attrs("1998 Playoff Contenders Peyton Manning Rookie Ticket #87 L@@K MINT")
    assert parsed.player == "peyton manning"


def test_flag_words_are_stripped_from_player_names():
    # "rookie", "rc", "auto" describe the card, never the person.
    parsed = attrs("2017 Prizm Patrick Mahomes II Rookie RC Auto #269")
    assert parsed.player == "patrick mahomes ii"


def test_decimal_grade_survives_normalization():
    assert attrs("Card BGS 9.5").grade == 9.5
    assert attrs("Card PSA 10").grade == 10.0


def test_grade_is_not_mistaken_for_a_card_number():
    parsed = attrs("2017 Prizm Mahomes PSA 10")
    assert parsed.card_number is None
    assert parsed.grade == 10.0


def test_one_of_one_detected():
    parsed = attrs("2021 National Treasures Lawrence Auto 1/1")
    assert parsed.is_one_of_one is True
    assert parsed.print_run == 1


def test_player_index_resolves_a_surname_only_title():
    index = [("mahomes", "patrick mahomes ii")]
    assert attrs("2017 Prizm Mahomes #269", player_index=index).player == "patrick mahomes ii"


def test_set_beats_shorter_overlapping_alias():
    # "bowman chrome" must not be read as plain "bowman".
    assert attrs("2020 Bowman Chrome Joe Burrow").set_name == "bowman chrome"


def test_brand_inferred_from_set_when_unstated():
    assert attrs("2017 Prizm Mahomes #269").brand == "panini"


def test_ocr_text_uses_the_same_parser():
    parsed = parse_text("PRIZM 2017 PATRICK MAHOMES II 269", Source.OCR).attributes
    assert parsed.set_name == "prizm"
    assert parsed.year == 2017


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Patrick Mahomes II", "patrick mahomes"), ("Odell Beckham Jr.", "odell beckham"),
     ("Josh Allen", "josh allen"), (None, "")],
)
def test_canonical_player_drops_generational_suffixes(raw, expected):
    assert canonical_player(raw) == expected


def test_empty_input_yields_empty_attributes():
    assert attrs("").is_empty()
