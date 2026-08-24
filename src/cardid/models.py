"""Core domain models shared by every stage of the pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Source(str, Enum):
    """Where a piece of evidence came from."""

    OCR = "ocr"
    TITLE = "title"
    VISUAL = "visual"
    USER = "user"


class Decision(str, Enum):
    """What the caller should do with a result."""

    AUTO_ACCEPT = "auto_accept"
    REVIEW = "review"
    REJECT = "reject"


class CardAttributes(BaseModel):
    """Normalized attributes of a football card.

    Every field is optional: a listing title may carry a year and player while
    the card face carries the number, and the matcher fuses whatever it gets.
    Strings are stored in canonical (lowercase, de-abbreviated) form so that
    values from OCR and from eBay titles compare directly.
    """

    year: int | None = None
    brand: str | None = None
    set_name: str | None = None
    subset: str | None = None
    player: str | None = None
    team: str | None = None
    card_number: str | None = None
    parallel: str | None = None

    print_run: int | None = Field(default=None, description="Denominator of /99")
    serial: str | None = Field(default=None, description="Full serial, e.g. 12/99")

    is_rookie: bool = False
    is_autograph: bool = False
    is_patch: bool = False
    is_one_of_one: bool = False

    grader: str | None = None
    grade: float | None = None

    def is_empty(self) -> bool:
        return not any(
            (self.year, self.brand, self.set_name, self.player, self.card_number)
        )

    def merge(self, other: CardAttributes) -> CardAttributes:
        """Fill this object's empty fields from ``other``.

        Self wins on conflict, so callers merge in order of trust: the highest
        trust source is the receiver.
        """
        merged = self.model_copy(deep=True)
        for name in self.model_fields:
            mine = getattr(merged, name)
            theirs = getattr(other, name)
            if theirs in (None, False, ""):
                continue
            if mine in (None, False, ""):
                setattr(merged, name, theirs)
        return merged


class Extraction(BaseModel):
    """Attributes parsed from one source, with the raw text kept for audit."""

    source: Source
    attributes: CardAttributes
    raw_text: str = ""
    tokens_used: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class CatalogCard(BaseModel):
    """One canonical card in the reference catalog: the thing we match to."""

    card_id: str
    year: int | None = None
    brand: str | None = None
    set_name: str | None = None
    subset: str | None = None
    player: str | None = None
    team: str | None = None
    card_number: str | None = None
    parallel: str | None = None
    print_run: int | None = None
    is_rookie: bool = False
    is_autograph: bool = False
    is_patch: bool = False

    display_name: str = ""
    search_text: str = ""
    image_phash: str | None = None
    embedding_id: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)

    def to_attributes(self) -> CardAttributes:
        return CardAttributes(
            year=self.year,
            brand=self.brand,
            set_name=self.set_name,
            subset=self.subset,
            player=self.player,
            team=self.team,
            card_number=self.card_number,
            parallel=self.parallel,
            print_run=self.print_run,
            is_rookie=self.is_rookie,
            is_autograph=self.is_autograph,
            is_patch=self.is_patch,
        )


class ScoredCandidate(BaseModel):
    """A catalog card the matcher considered, with its score breakdown."""

    card: CatalogCard
    score: float
    field_scores: dict[str, float] = Field(default_factory=dict)
    penalties: dict[str, float] = Field(default_factory=dict)

    @property
    def card_id(self) -> str:
        return self.card.card_id


class Identification(BaseModel):
    """The pipeline's answer for one photo."""

    request_id: str
    decision: Decision
    confidence: float
    card: CatalogCard | None = None
    display_name: str | None = None

    runner_up: ScoredCandidate | None = None
    candidates: list[ScoredCandidate] = Field(default_factory=list)
    extractions: list[Extraction] = Field(default_factory=list)
    fused: CardAttributes = Field(default_factory=CardAttributes)

    reasons: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    cache_hit: bool = False
    image_phash: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "decision": self.decision.value,
            "confidence": round(self.confidence, 4),
            "card_id": self.card.card_id if self.card else None,
            "display_name": self.display_name,
        }
