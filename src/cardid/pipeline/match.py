"""Score catalog candidates against the attributes extracted from a photo.

Scoring is field-weighted with explicit contradiction penalties. The weights say
how much each field contributes when it agrees; the penalties say how badly a
direct disagreement should hurt. Both halves matter: without penalties, a card
that agrees on player and year but has the wrong card number still scores high.

The output keeps a per-field breakdown so a low-confidence result can be
explained to a human reviewer instead of being a bare number.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from ..models import CardAttributes, CatalogCard, ScoredCandidate

# Contribution of each field when the query and candidate agree.
WEIGHTS: dict[str, float] = {
    "player": 0.34,
    "card_number": 0.22,
    "set_name": 0.16,
    "year": 0.12,
    "parallel": 0.10,
    "brand": 0.06,
}

# Multiplier applied to a field's weight when the two sides directly disagree.
# Above 1.0 means a contradiction costs more than agreement earns, which is what
# keeps near-miss parallels and wrong-year reprints out of the top slot.
PENALTIES: dict[str, float] = {
    "player": 2.0,
    "card_number": 2.0,
    "set_name": 1.5,
    "year": 1.5,
    "parallel": 1.6,
    "brand": 0.8,
}

FLAG_FIELDS = ("is_rookie", "is_autograph", "is_patch")
FLAG_WEIGHT = 0.04


def _norm_number(value: str | None) -> str | None:
    """Card numbers compare after case and leading-zero normalization."""
    if not value:
        return None
    cleaned = value.strip().upper().lstrip("#")
    prefix = "".join(ch for ch in cleaned if ch.isalpha() or ch == "-")
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return f"{prefix}{digits.lstrip('0') or '0'}" if digits else cleaned


def _string_similarity(left: str, right: str) -> float:
    """0..1 similarity tolerant of token order and OCR-grade noise."""
    if left == right:
        return 1.0
    return max(fuzz.token_set_ratio(left, right), fuzz.ratio(left, right)) / 100.0


def _score_field(
    field: str, query_value: object, candidate_value: object
) -> tuple[float, float]:
    """Return ``(credit, penalty)`` for one field, both already weighted."""
    weight = WEIGHTS[field]
    if query_value in (None, "") or candidate_value in (None, ""):
        # Nothing to compare. Neither side is rewarded or punished; the missing
        # signal simply lowers the achievable total, which the caller normalizes.
        return 0.0, 0.0

    if field == "year":
        similarity = 1.0 if query_value == candidate_value else (
            0.55 if abs(int(query_value) - int(candidate_value)) == 1 else 0.0
        )
    elif field == "card_number":
        similarity = 1.0 if _norm_number(str(query_value)) == _norm_number(
            str(candidate_value)
        ) else 0.0
    elif field == "parallel":
        similarity = _string_similarity(str(query_value), str(candidate_value))
        similarity = 1.0 if similarity > 0.9 else 0.0
    else:
        similarity = _string_similarity(str(query_value), str(candidate_value))
        if similarity < 0.72:
            similarity = 0.0

    credit = weight * similarity
    penalty = weight * PENALTIES[field] * (1.0 - similarity) if similarity < 0.5 else 0.0
    return credit, penalty


def _score_parallel_absence(
    query: CardAttributes, candidate: CatalogCard
) -> tuple[float, float]:
    """Handle the common case where the query names no parallel at all.

    A listing that says nothing about a parallel is usually a base card, so an
    unnamed parallel gently favours base over a coloured variant rather than
    treating every parallel as equally likely.
    """
    if query.parallel:
        return 0.0, 0.0
    candidate_parallel = (candidate.parallel or "base").lower()
    if candidate_parallel in ("", "base"):
        return WEIGHTS["parallel"] * 0.5, 0.0
    return 0.0, WEIGHTS["parallel"] * 0.4


def score_candidate(query: CardAttributes, candidate: CatalogCard) -> ScoredCandidate:
    """Score one catalog card against the fused query attributes."""
    field_scores: dict[str, float] = {}
    penalties: dict[str, float] = {}
    achievable = 0.0
    earned = 0.0
    total_penalty = 0.0

    candidate_attrs = candidate.to_attributes()
    for field in WEIGHTS:
        query_value = getattr(query, field)
        candidate_value = getattr(candidate_attrs, field)
        credit, penalty = _score_field(field, query_value, candidate_value)
        if query_value not in (None, "") and candidate_value not in (None, ""):
            achievable += WEIGHTS[field]
        if credit:
            field_scores[field] = round(credit, 4)
        if penalty:
            penalties[field] = round(penalty, 4)
        earned += credit
        total_penalty += penalty

    # Only meaningful once something else actually matched: preferring the base
    # card is a tie-breaker between real candidates, not evidence on its own.
    credit, penalty = _score_parallel_absence(query, candidate) if achievable > 0 else (0.0, 0.0)
    if credit:
        field_scores["parallel_absent"] = round(credit, 4)
        achievable += WEIGHTS["parallel"] * 0.5
        earned += credit
    if penalty:
        penalties["parallel_absent"] = round(penalty, 4)
        total_penalty += penalty

    # Boolean flags act as tie-breakers between otherwise identical rows.
    for flag in FLAG_FIELDS:
        query_flag = getattr(query, flag)
        candidate_flag = getattr(candidate_attrs, flag)
        if query_flag and candidate_flag:
            earned += FLAG_WEIGHT
            achievable += FLAG_WEIGHT
            field_scores[flag] = FLAG_WEIGHT
        elif query_flag != candidate_flag and query_flag:
            total_penalty += FLAG_WEIGHT
            penalties[flag] = FLAG_WEIGHT

    if query.print_run and candidate.print_run:
        if query.print_run == candidate.print_run:
            earned += 0.05
            achievable += 0.05
            field_scores["print_run"] = 0.05
        else:
            total_penalty += 0.08
            penalties["print_run"] = 0.08

    if achievable <= 0:
        return ScoredCandidate(card=candidate, score=0.0)

    # Normalize by what was actually comparable, then subtract penalties. A card
    # matched on two fields cannot outscore one matched on six purely because it
    # had less to disagree about, so scale by evidence breadth.
    raw = (earned - total_penalty) / achievable
    breadth = min(1.0, achievable / 0.75)
    score = max(0.0, min(1.0, raw)) * breadth

    return ScoredCandidate(
        card=candidate,
        score=round(score, 6),
        field_scores=field_scores,
        penalties=penalties,
    )


def rank_candidates(
    query: CardAttributes, candidates: list[CatalogCard], top_k: int = 5
) -> list[ScoredCandidate]:
    """Score every candidate and return the best ``top_k``, highest first."""
    scored = [score_candidate(query, candidate) for candidate in candidates]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]
