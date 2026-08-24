"""Result caching and the human review feedback loop."""

from __future__ import annotations

from cardid.cache import ResultCache, sha256_bytes
from cardid.models import CatalogCard, Decision, Identification
from cardid.review import ReviewStore


def result(request_id="r1", decision=Decision.AUTO_ACCEPT, confidence=0.98):
    return Identification(
        request_id=request_id, decision=decision, confidence=confidence,
        card=CatalogCard(card_id="mahomes_silver", display_name="Mahomes Silver"),
        display_name="Mahomes Silver", image_phash="abc123",
    )


def test_cache_round_trips_by_content_hash():
    cache = ResultCache(":memory:")
    cache.put(result(), "sha-key", "phash-key")
    assert cache.get("sha-key", None).card.card_id == "mahomes_silver"


def test_cache_round_trips_by_perceptual_hash():
    """Catches the same photo after eBay resizes or re-encodes it."""
    cache = ResultCache(":memory:")
    cache.put(result(), "sha-key", "phash-key")
    assert cache.get("different-sha", "phash-key") is not None


def test_rejected_results_are_not_cached():
    cache = ResultCache(":memory:")
    cache.put(result(decision=Decision.REJECT, confidence=0.0), "sha-key", "phash-key")
    assert cache.get("sha-key", "phash-key") is None


def test_disabled_cache_never_returns_anything():
    cache = ResultCache(":memory:", enabled=False)
    cache.put(result(), "sha-key", "phash-key")
    assert cache.get("sha-key", "phash-key") is None


def test_cache_counts_hits():
    cache = ResultCache(":memory:")
    cache.put(result(), "sha-key", None)
    cache.get("sha-key", None)
    cache.get("sha-key", None)
    assert cache.stats()["hits"] == 2


def test_sha256_is_stable():
    assert sha256_bytes(b"abc") == sha256_bytes(b"abc")
    assert sha256_bytes(b"abc") != sha256_bytes(b"abd")


def test_resolving_a_review_item_creates_a_label():
    queue = ReviewStore(":memory:")
    queue.enqueue(result(decision=Decision.REVIEW, confidence=0.6), title="a title")
    assert queue.resolve("r1", "mahomes_gold") is True
    assert queue.stats() == {"pending": 0, "resolved": 1, "labels": 1}


def test_a_resolution_records_whether_the_prediction_was_right():
    queue = ReviewStore(":memory:")
    queue.enqueue(result(decision=Decision.REVIEW, confidence=0.6))
    queue.resolve("r1", "mahomes_gold")  # predicted silver, truth is gold
    assert dict(queue.labels()[0])["was_correct"] == 0


def test_resolved_items_leave_the_queue():
    queue = ReviewStore(":memory:")
    queue.enqueue(result(decision=Decision.REVIEW, confidence=0.6))
    queue.resolve("r1", "mahomes_silver")
    assert queue.pending() == []


def test_resolving_an_unknown_item_reports_failure():
    assert ReviewStore(":memory:").resolve("nope", "x") is False


def test_confirmed_images_are_recallable_by_hash():
    queue = ReviewStore(":memory:")
    queue.enqueue(result(decision=Decision.REVIEW, confidence=0.6))
    queue.resolve("r1", "mahomes_gold")
    assert queue.known_by_phash("abc123") == "mahomes_gold"


def test_queue_is_ordered_most_confident_first():
    """Reviewers should see near-misses before hopeless cases."""
    queue = ReviewStore(":memory:")
    queue.enqueue(result("low", Decision.REVIEW, 0.50))
    queue.enqueue(result("high", Decision.REVIEW, 0.85))
    assert [item.request_id for item in queue.pending()] == ["high", "low"]
