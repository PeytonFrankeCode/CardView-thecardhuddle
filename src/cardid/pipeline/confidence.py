"""Turn a ranked candidate list into a confidence score and a decision.

The raw match score says how well the best catalog row fits the evidence. That
alone is not enough to act on, because the dangerous failure is not a low score
— it is a high score shared by two rows that differ only by parallel. So
confidence folds in three things:

* how well the top candidate fits,
* how much better it fits than the runner-up (the margin),
* whether independent sources (slab label, listing title, card face) agree.

The decision thresholds are configuration, not constants: the whole point is to
calibrate them against labeled data so the auto-accepted slice hits whatever
precision the business actually needs.
"""

from __future__ import annotations

from ..models import CardAttributes, Decision, Extraction, ScoredCandidate, Source

# Fields where two sources agreeing is strong evidence, and disagreeing is a
# genuine red flag rather than a difference in how much each source saw.
CORROBORATION_FIELDS = ("player", "card_number", "year", "set_name")

# Margin at which the top candidate is considered decisively better.
DECISIVE_MARGIN = 0.25

# Hardest confidence a near-tied result may report. Sits below any sane
# auto-accept threshold and above the review floor, so near-ties become review
# items rather than silent wrong answers.
NEAR_TIE_CEILING = 0.60


def source_agreement(extractions: list[Extraction]) -> tuple[float, list[str]]:
    """Compare independent extractions field by field.

    Returns a multiplier and human-readable notes. Sources that never saw a
    field are ignored; only genuine contradictions are penalized.
    """
    notes: list[str] = []
    usable = [e for e in extractions if e.source != Source.USER and not e.attributes.is_empty()]
    if len(usable) < 2:
        return 1.0, notes

    agreements = 0
    conflicts = 0
    for field in CORROBORATION_FIELDS:
        values = {
            str(getattr(e.attributes, field)).lower()
            for e in usable
            if getattr(e.attributes, field) not in (None, "")
        }
        if len(values) > 1:
            conflicts += 1
            notes.append(f"sources disagree on {field}: {sorted(values)}")
        elif len(values) == 1 and sum(
            1 for e in usable if getattr(e.attributes, field) not in (None, "")
        ) > 1:
            agreements += 1
            notes.append(f"sources agree on {field}")

    if conflicts:
        # Each contradiction cuts confidence; two independent sources naming
        # different players is close to fatal.
        return max(0.55, 1.0 - 0.22 * conflicts), notes
    if agreements >= 2:
        return 1.06, notes
    if agreements == 1:
        return 1.02, notes
    return 1.0, notes


def compute_confidence(
    ranked: list[ScoredCandidate],
    extractions: list[Extraction],
    fused: CardAttributes,
    min_margin: float = 0.04,
) -> tuple[float, list[str]]:
    """Score how much to trust the top candidate. Returns (confidence, reasons)."""
    reasons: list[str] = []
    if not ranked:
        return 0.0, ["no catalog candidates"]

    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    if runner_up is None:
        margin = 1.0
        reasons.append("only one candidate considered")
    else:
        margin = top.score - runner_up.score
        reasons.append(f"margin over runner-up: {margin:.3f}")

    # The margin ramps from min_margin (no credit) to DECISIVE_MARGIN (full
    # credit). Saturating right at min_margin would hand full confidence to a
    # near-tie, which is the single most dangerous failure mode here: two rows
    # differing only by parallel routinely land within a few points.
    span = max(1e-6, DECISIVE_MARGIN - min_margin)
    margin_factor = max(0.0, min(1.0, (margin - min_margin) / span))

    multiplier, agreement_notes = source_agreement(extractions)
    reasons.extend(agreement_notes)

    # Identification with no card number and no year is thin evidence even when
    # the player matches perfectly, since a player has hundreds of cards.
    specificity = 1.0
    if not fused.card_number and not fused.year:
        specificity = 0.72
        reasons.append("no card number or year recovered — weak specificity")
    elif not fused.card_number:
        specificity = 0.90
        reasons.append("no card number recovered")

    confidence = top.score * (0.62 + 0.38 * margin_factor) * multiplier * specificity
    confidence = max(0.0, min(1.0, confidence))

    # A margin below the floor is a hard cap, not a discount. However well the
    # top row scores, if a second row fits nearly as well we cannot claim to
    # know which one it is, so the result is forced out of auto-accept and in
    # front of a human.
    if margin < min_margin:
        reasons.append(
            f"near-tie: margin {margin:.3f} below floor {min_margin:.3f} — capped for review"
        )
        confidence = min(confidence, NEAR_TIE_CEILING)

    return round(confidence, 4), reasons


def decide(
    confidence: float, auto_accept_threshold: float, review_threshold: float
) -> Decision:
    """Map a confidence score onto an action."""
    if confidence >= auto_accept_threshold:
        return Decision.AUTO_ACCEPT
    if confidence >= review_threshold:
        return Decision.REVIEW
    return Decision.REJECT
