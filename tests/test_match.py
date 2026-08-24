"""Scoring catalog candidates — especially telling parallels apart."""

from __future__ import annotations

from cardid.models import CardAttributes, CatalogCard
from cardid.pipeline.match import _norm_number, rank_candidates, score_candidate
from cardid.pipeline.parse import parse_text


def ids(ranked):
    return [candidate.card.card_id for candidate in ranked]


def test_named_parallel_outranks_the_base_card(store):
    query = parse_text("2017 Panini Prizm Patrick Mahomes II #269 Silver Prizm").attributes
    ranked = rank_candidates(query, store.candidates(query))
    assert ids(ranked)[0] == "mahomes_silver"


def test_unstated_parallel_favours_the_base_card(store):
    query = parse_text("2017 Panini Prizm Patrick Mahomes II #269 RC").attributes
    ranked = rank_candidates(query, store.candidates(query))
    assert ids(ranked)[0] == "mahomes_base"


def test_wrong_card_number_is_penalised_below_a_right_one(store):
    query = parse_text("2020 Panini Prizm Joe Burrow #307").attributes
    ranked = rank_candidates(query, store.candidates(query))
    assert ids(ranked)[0] == "burrow_prizm"


def test_print_run_disambiguates_a_numbered_parallel(store):
    query = parse_text("2017 Prizm Patrick Mahomes II #269 Gold /10").attributes
    ranked = rank_candidates(query, store.candidates(query))
    assert ids(ranked)[0] == "mahomes_gold"


def test_contradiction_costs_more_than_agreement_earns():
    query = CardAttributes(player="josh allen", year=2018, set_name="contenders",
                           card_number="101")
    right = CatalogCard(card_id="right", player="josh allen", year=2018,
                        set_name="contenders", card_number="101")
    wrong_number = CatalogCard(card_id="wrong", player="josh allen", year=2018,
                               set_name="contenders", card_number="999")
    assert score_candidate(query, right).score > score_candidate(query, wrong_number).score


def test_broader_evidence_beats_a_thin_lucky_match():
    query = CardAttributes(player="josh allen", year=2018, set_name="contenders",
                           card_number="101", brand="panini")
    broad = CatalogCard(card_id="broad", player="josh allen", year=2018,
                        set_name="contenders", card_number="101", brand="panini")
    thin = CatalogCard(card_id="thin", player="josh allen")
    assert score_candidate(query, broad).score > score_candidate(query, thin).score


def test_score_breakdown_is_reported_for_review():
    query = CardAttributes(player="josh allen", year=2018, card_number="101")
    card = CatalogCard(card_id="c", player="josh allen", year=2018, card_number="101")
    scored = score_candidate(query, card)
    assert scored.field_scores["player"] > 0
    assert scored.field_scores["card_number"] > 0


def test_no_comparable_fields_scores_zero():
    assert score_candidate(CardAttributes(), CatalogCard(card_id="x")).score == 0.0


def test_card_numbers_compare_past_formatting():
    assert _norm_number("#0269") == _norm_number("269")
    assert _norm_number("rc-15") == _norm_number("RC-15")
    assert _norm_number("269") != _norm_number("270")
