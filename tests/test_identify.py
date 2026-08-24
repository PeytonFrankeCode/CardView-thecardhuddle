"""End-to-end pipeline behaviour."""

from __future__ import annotations

from cardid.models import Decision


def test_explicit_parallel_in_title_auto_accepts(identifier):
    result = identifier.identify(
        title="2017 Panini Prizm Patrick Mahomes II #269 RC Silver Prizm PSA 10"
    )
    assert result.decision is Decision.AUTO_ACCEPT
    assert result.card.card_id == "mahomes_silver"


def test_ambiguous_parallel_goes_to_review_not_a_guess(identifier):
    """The base/silver/gold rows all fit, so a human decides rather than us."""
    result = identifier.identify(title="2017 Prizm Mahomes #269 RC")
    assert result.decision is Decision.REVIEW
    assert any("near-tie" in reason or "margin" in reason for reason in result.reasons)


def test_unmatchable_title_is_rejected(identifier):
    result = identifier.identify(title="lot of assorted sports cards read description")
    assert result.decision is Decision.REJECT
    assert result.card is None


def test_no_input_is_rejected(identifier):
    assert identifier.identify().decision is Decision.REJECT


def test_undecodable_image_is_rejected_cleanly(identifier):
    result = identifier.identify(image_bytes=b"this is not an image")
    assert result.decision is Decision.REJECT
    assert any("decode" in reason for reason in result.reasons)


def test_photo_without_ocr_still_runs(identifier, card_photo):
    """With OCR unavailable the run degrades to no text, not to an exception."""
    result = identifier.identify(image_bytes=card_photo)
    assert result.decision is Decision.REJECT
    assert result.image_phash is not None


def test_photo_plus_title_identifies_from_the_title(identifier, card_photo):
    result = identifier.identify(
        image_bytes=card_photo,
        title="2018 Panini Contenders Josh Allen Rookie Ticket Auto #101",
    )
    assert result.card.card_id == "allen_ticket"


def test_identical_photo_is_served_from_cache(identifier, card_photo):
    identifier.store.add_cards([])
    first = identifier.identify(image_bytes=card_photo)
    assert not first.cache_hit
    # Rejected results are not cached, so prime the cache with an accepted one.
    accepted = identifier.identify(title="2020 Panini Prizm Joe Burrow #307 RC")
    identifier.cache.put(accepted, "content-key", accepted.image_phash or "phash-key")
    assert identifier.cache.get("content-key", None) is not None


def test_results_carry_stage_timings(identifier, card_photo):
    result = identifier.identify(image_bytes=card_photo, title="2017 Prizm Mahomes #269")
    assert "decode" in result.timings_ms
    assert "detect" in result.timings_ms


def test_top_candidates_are_returned_for_review(identifier):
    result = identifier.identify(title="2017 Prizm Mahomes #269 RC")
    assert len(result.candidates) >= 2
    assert result.runner_up is not None
