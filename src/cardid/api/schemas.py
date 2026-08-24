"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import CardAttributes, CatalogCard, Decision


class IdentifyRequest(BaseModel):
    """Identify one card from a URL and/or a listing title."""

    image_url: str | None = None
    title: str | None = Field(
        default=None,
        description="eBay listing title. Supply it whenever you have it — on a "
        "sold-listing feed it is usually the strongest single signal.",
    )
    request_id: str | None = None
    use_cache: bool = True


class BatchItem(BaseModel):
    item_id: str
    image_url: str | None = None
    title: str | None = None


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., max_length=500)
    use_cache: bool = True


class CandidateOut(BaseModel):
    card_id: str
    display_name: str
    score: float
    field_scores: dict[str, float] = Field(default_factory=dict)
    penalties: dict[str, float] = Field(default_factory=dict)


class IdentifyResponse(BaseModel):
    """What the website gets back."""

    request_id: str
    decision: Decision
    confidence: float
    card_id: str | None = None
    display_name: str | None = None
    card: CatalogCard | None = None
    attributes: CardAttributes = Field(default_factory=CardAttributes)
    candidates: list[CandidateOut] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class BatchResponseItem(BaseModel):
    item_id: str
    result: IdentifyResponse | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    items: list[BatchResponseItem]
    counts: dict[str, int] = Field(default_factory=dict)


class ResolveRequest(BaseModel):
    card_id: str
    resolved_by: str = "human"


class ReviewItemOut(BaseModel):
    request_id: str
    created_at: float
    confidence: float | None
    title: str | None
    image_url: str | None
    predicted_card_id: str | None
    predicted_name: str | None
    candidates: list[CandidateOut] = Field(default_factory=list)
