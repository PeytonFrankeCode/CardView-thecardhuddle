"""The visual reranker slot."""

from __future__ import annotations

import numpy as np

from cardid.models import CatalogCard, ScoredCandidate
from cardid.pipeline.embed import NullEmbeddingBackend, cosine_similarity, rerank


def candidates():
    return [
        ScoredCandidate(card=CatalogCard(card_id="base"), score=0.90),
        ScoredCandidate(card=CatalogCard(card_id="silver"), score=0.88),
    ]


def test_visual_evidence_can_overturn_a_near_tie():
    """Exactly the case text cannot resolve: same card, different finish."""
    reranked = rerank(candidates(), {"base": 0.10, "silver": 0.95})
    assert reranked[0].card.card_id == "silver"


def test_candidates_without_a_visual_score_are_left_alone():
    reranked = rerank(candidates(), {"base": 0.5})
    assert {c.card.card_id for c in reranked} == {"base", "silver"}
    assert next(c for c in reranked if c.card.card_id == "silver").score == 0.88


def test_no_visual_scores_is_a_no_op():
    original = candidates()
    assert rerank(original, {}) is original


def test_visual_score_is_recorded_for_review():
    reranked = rerank(candidates(), {"base": 0.42, "silver": 0.9})
    assert reranked[0].field_scores["visual"] == 0.9


def test_cosine_similarity_bounds():
    vector = np.array([1.0, 0.0, 0.0], dtype="float32")
    assert cosine_similarity(vector, vector) == 1.0
    assert cosine_similarity(vector, -vector) == 0.0
    assert cosine_similarity(np.zeros(0), vector) == 0.0


def test_null_backend_disables_reranking():
    backend = NullEmbeddingBackend()
    assert backend.embed(np.zeros((10, 10, 3), np.uint8)).size == 0
