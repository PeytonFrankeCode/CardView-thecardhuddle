"""Confidence scoring and the auto-accept / review / reject gate."""

from __future__ import annotations

import pytest

from cardid.config import settings
from cardid.models import (
    CardAttributes,
    CatalogCard,
    Decision,
    Extraction,
    ScoredCandidate,
    Source,
)
from cardid.pipeline.confidence import (
    NEAR_TIE_CEILING,
    compute_confidence,
    decide,
    source_agreement,
)

FULL = CardAttributes(player="josh allen", card_number="101", year=2018,
                      set_name="contenders")


def ranked(*scores):
    return [
        ScoredCandidate(card=CatalogCard(card_id=f"c{i}"), score=score)
        for i, score in enumerate(scores)
    ]


def agreeing_sources():
    return [
        Extraction(source=Source.TITLE, attributes=FULL),
        Extraction(source=Source.OCR, attributes=FULL),
    ]


def test_near_tie_never_auto_accepts():
    """The dangerous failure is two rows fitting equally well, not a low score."""
    confidence, _ = compute_confidence(ranked(1.0, 0.99), agreeing_sources(), FULL,
                                       settings.min_margin)
    assert confidence <= NEAR_TIE_CEILING
    assert decide(confidence, settings.auto_accept_threshold,
                  settings.review_threshold) is Decision.REVIEW


def test_decisive_margin_auto_accepts():
    confidence, _ = compute_confidence(ranked(1.0, 0.6), agreeing_sources(), FULL,
                                       settings.min_margin)
    assert decide(confidence, settings.auto_accept_threshold,
                  settings.review_threshold) is Decision.AUTO_ACCEPT


def test_confidence_rises_monotonically_with_margin():
    scores = [
        compute_confidence(ranked(1.0, 1.0 - gap), agreeing_sources(), FULL,
                           settings.min_margin)[0]
        for gap in (0.02, 0.10, 0.20, 0.40)
    ]
    assert scores == sorted(scores)


def test_no_candidates_scores_zero():
    confidence, reasons = compute_confidence([], [], FULL, settings.min_margin)
    assert confidence == 0.0
    assert decide(confidence, 0.9, 0.45) is Decision.REJECT
    assert reasons


def test_conflicting_sources_reduce_confidence():
    conflicting = [
        Extraction(source=Source.TITLE, attributes=CardAttributes(player="josh allen")),
        Extraction(source=Source.OCR, attributes=CardAttributes(player="joe burrow")),
    ]
    agreed, _ = compute_confidence(ranked(1.0, 0.5), agreeing_sources(), FULL,
                                   settings.min_margin)
    conflicted, _ = compute_confidence(ranked(1.0, 0.5), conflicting, FULL,
                                       settings.min_margin)
    assert conflicted < agreed


def test_missing_number_and_year_lowers_confidence():
    """A player alone is weak: one player has hundreds of cards."""
    thin = CardAttributes(player="josh allen")
    strong, _ = compute_confidence(ranked(1.0, 0.5), agreeing_sources(), FULL,
                                   settings.min_margin)
    weak, _ = compute_confidence(ranked(1.0, 0.5), agreeing_sources(), thin,
                                 settings.min_margin)
    assert weak < strong


def test_source_agreement_needs_two_sources():
    multiplier, _ = source_agreement([Extraction(source=Source.TITLE, attributes=FULL)])
    assert multiplier == 1.0


def test_source_agreement_flags_the_conflicting_field():
    _, notes = source_agreement([
        Extraction(source=Source.TITLE, attributes=CardAttributes(player="josh allen")),
        Extraction(source=Source.OCR, attributes=CardAttributes(player="joe burrow")),
    ])
    assert any("disagree on player" in note for note in notes)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.99, Decision.AUTO_ACCEPT), (0.90, Decision.AUTO_ACCEPT),
     (0.70, Decision.REVIEW), (0.45, Decision.REVIEW), (0.10, Decision.REJECT)],
)
def test_thresholds_map_to_decisions(confidence, expected):
    assert decide(confidence, 0.90, 0.45) is expected
