"""Visual embedding interface — the reranker slot.

Text can only take identification so far. Two parallels of the same card share a
year, set, player and card number; they differ in *finish* — a silver shimmer, a
gold refractor pattern, a coloured border. When neither the listing title nor
the card face text names the parallel, no amount of text matching can separate
them, and the pipeline correctly routes those to review.

Closing that gap needs pixels: embed the rectified card crop, compare it against
reference images for each candidate, and rerank. That is the highest-value
remaining improvement, and this module is where it plugs in.

To implement one:

1. Embed a reference image per catalog card (CLIP or DINOv2 both work well) and
   store the vector, keyed by ``CatalogCard.embedding_id``.
2. Implement :class:`EmbeddingBackend` below.
3. Call :func:`rerank` in ``CardIdentifier.identify`` after ``rank_candidates``,
   blending the visual score into the text score.

Nothing else in the pipeline needs to change — candidates already carry a score
breakdown, and the confidence gate already keys off the margin, which is exactly
the quantity a good reranker widens.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ...models import ScoredCandidate

# How much weight a visual score carries relative to the text score. Text stays
# dominant because it is far more reliable when present; vision breaks ties.
VISUAL_WEIGHT = 0.35


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str
    dimensions: int

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Return a unit-norm embedding for a rectified card crop."""
        ...

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        """Cosine similarity in 0..1."""
        ...


class NullEmbeddingBackend:
    """No-op backend. Reranking is skipped and text scores stand unchanged."""

    name = "null"
    dimensions = 0

    def embed(self, image: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros(0, dtype="float32")

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:  # noqa: ARG002
        return 0.0


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity mapped from [-1, 1] onto [0, 1]."""
    if left.size == 0 or right.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float((np.dot(left, right) / denominator + 1.0) / 2.0)


def rerank(
    candidates: list[ScoredCandidate],
    visual_scores: dict[str, float],
    visual_weight: float = VISUAL_WEIGHT,
) -> list[ScoredCandidate]:
    """Blend visual similarity into text scores and re-sort.

    ``visual_scores`` maps ``card_id`` to a 0..1 similarity. Candidates with no
    visual score keep their text score unchanged, so a partially-embedded
    catalog degrades gracefully instead of penalising unembedded cards.
    """
    if not visual_scores:
        return candidates

    reranked: list[ScoredCandidate] = []
    for candidate in candidates:
        visual = visual_scores.get(candidate.card.card_id)
        if visual is None:
            reranked.append(candidate)
            continue
        updated = candidate.model_copy(deep=True)
        updated.score = round(
            candidate.score * (1.0 - visual_weight) + visual * visual_weight, 6
        )
        updated.field_scores["visual"] = round(visual, 4)
        reranked.append(updated)

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked
