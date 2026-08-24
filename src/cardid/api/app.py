"""FastAPI application: the interface the website talks to.

Design notes:

* Identification is CPU-bound (OpenCV plus OCR), so it runs in a worker thread
  and never blocks the event loop. That is what lets one process serve image
  downloads concurrently while a match is in flight.
* Anything not auto-accepted is queued for review automatically, so the caller
  never has to remember to do it.
* Auth is a shared API key header, on by default whenever CARDID_API_KEY is set.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ..cache import ResultCache
from ..catalog.store import CatalogStore
from ..config import settings
from ..fetch import FetchError, ImageFetcher
from ..models import Decision, Identification
from ..pipeline.identify import CardIdentifier
from ..review import ReviewStore
from .schemas import (
    BatchRequest,
    BatchResponse,
    BatchResponseItem,
    CandidateOut,
    IdentifyRequest,
    IdentifyResponse,
    ResolveRequest,
    ReviewItemOut,
)

log = logging.getLogger(__name__)

# Populated on startup; module-level so tests can swap them out.
state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.setdefault("store", CatalogStore(settings.catalog_path))
    state.setdefault("review", ReviewStore("data/review.db"))
    state.setdefault(
        "identifier",
        CardIdentifier(
            store=state["store"],  # type: ignore[arg-type]
            cache=ResultCache(settings.cache_path, enabled=settings.cache_enabled),
        ),
    )
    state.setdefault(
        "fetcher",
        ImageFetcher(
            max_bytes=settings.max_image_bytes,
            timeout=settings.fetch_timeout_seconds,
            max_concurrent=settings.max_concurrent_fetches,
        ),
    )
    log.info("catalog holds %s cards", state["store"].count())  # type: ignore[union-attr]
    yield
    await state["fetcher"].aclose()  # type: ignore[union-attr]


app = FastAPI(
    title="Football Card Identification",
    version="1.0.0",
    summary="Identify a football card from a photo and/or an eBay listing title.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Shared-key auth. Disabled when no key is configured, for local dev."""
    import os

    expected = os.environ.get("CARDID_API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _identifier() -> CardIdentifier:
    identifier = state.get("identifier")
    if identifier is None:
        raise HTTPException(status_code=503, detail="service not ready")
    return identifier  # type: ignore[return-value]


def _review() -> ReviewStore:
    return state["review"]  # type: ignore[return-value]


def _to_response(result: Identification) -> IdentifyResponse:
    return IdentifyResponse(
        request_id=result.request_id,
        decision=result.decision,
        confidence=result.confidence,
        card_id=result.card.card_id if result.card else None,
        display_name=result.display_name,
        card=result.card,
        attributes=result.fused,
        candidates=[
            CandidateOut(
                card_id=candidate.card.card_id,
                display_name=candidate.card.display_name,
                score=candidate.score,
                field_scores=candidate.field_scores,
                penalties=candidate.penalties,
            )
            for candidate in result.candidates
        ],
        reasons=result.reasons,
        cache_hit=result.cache_hit,
        timings_ms=result.timings_ms,
    )


async def _run_identify(
    image_bytes: bytes | None,
    title: str | None,
    request_id: str | None,
    use_cache: bool,
    image_url: str | None = None,
) -> IdentifyResponse:
    """Run the pipeline off the event loop and queue anything uncertain."""
    identifier = _identifier()
    result = await asyncio.to_thread(
        identifier.identify,
        image_bytes=image_bytes,
        title=title,
        request_id=request_id,
        use_cache=use_cache,
    )
    if result.decision is not Decision.AUTO_ACCEPT:
        _review().enqueue(result, image_url=image_url, title=title)
    return _to_response(result)


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, object]:
    store: CatalogStore = state["store"]  # type: ignore[assignment]
    identifier = _identifier()
    return {
        "status": "ok",
        "catalog_cards": store.count(),
        "ocr_backend": identifier.ocr.name,
        "auto_accept_threshold": settings.auto_accept_threshold,
    }


@app.get("/v1/stats", tags=["ops"], dependencies=[Depends(require_api_key)])
async def stats() -> dict[str, object]:
    identifier = _identifier()
    return {
        "catalog": {"cards": state["store"].count()},  # type: ignore[union-attr]
        "cache": identifier.cache.stats(),
        "review": _review().stats(),
    }


@app.post(
    "/v1/identify",
    response_model=IdentifyResponse,
    tags=["identify"],
    dependencies=[Depends(require_api_key)],
)
async def identify(payload: IdentifyRequest) -> IdentifyResponse:
    """Identify a card from an image URL and/or a listing title."""
    if not payload.image_url and not payload.title:
        raise HTTPException(status_code=422, detail="supply image_url, title, or both")

    image_bytes: bytes | None = None
    if payload.image_url:
        fetcher: ImageFetcher = state["fetcher"]  # type: ignore[assignment]
        try:
            image_bytes = (await fetcher.fetch(payload.image_url)).data
        except FetchError as exc:
            if not payload.title:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            # With a title in hand, a dead image URL is a degraded run, not a
            # failure: identify from text and say so in the reasons.
            log.warning("image fetch failed, continuing title-only: %s", exc)

    response = await _run_identify(
        image_bytes, payload.title, payload.request_id, payload.use_cache,
        image_url=payload.image_url,
    )
    if payload.image_url and image_bytes is None:
        response.reasons.append("image fetch failed — identified from title only")
    return response


@app.post(
    "/v1/identify/upload",
    response_model=IdentifyResponse,
    tags=["identify"],
    dependencies=[Depends(require_api_key)],
)
async def identify_upload(
    file: Annotated[UploadFile, File(description="Card photo")],
    title: Annotated[str | None, Form()] = None,
    request_id: Annotated[str | None, Form()] = None,
) -> IdentifyResponse:
    """Identify a card from a directly uploaded photo."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty upload")
    if len(data) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="image exceeds size limit")
    return await _run_identify(data, title, request_id, use_cache=True)


@app.post(
    "/v1/identify/batch",
    response_model=BatchResponse,
    tags=["identify"],
    dependencies=[Depends(require_api_key)],
)
async def identify_batch(payload: BatchRequest) -> BatchResponse:
    """Identify up to 500 cards in one call.

    Images are fetched concurrently, then matched. This is the endpoint to use
    for backfilling a sold-listing table.
    """
    fetcher: ImageFetcher = state["fetcher"]  # type: ignore[assignment]

    async def one(item) -> BatchResponseItem:
        image_bytes = None
        if item.image_url:
            try:
                image_bytes = (await fetcher.fetch(item.image_url)).data
            except FetchError as exc:
                if not item.title:
                    return BatchResponseItem(item_id=item.item_id, error=str(exc))
        if not image_bytes and not item.title:
            return BatchResponseItem(item_id=item.item_id, error="no usable input")
        try:
            result = await _run_identify(
                image_bytes, item.title, None, payload.use_cache,
                image_url=item.image_url,
            )
            return BatchResponseItem(item_id=item.item_id, result=result)
        except Exception as exc:  # one bad item must not fail the batch
            log.exception("batch item %s failed", item.item_id)
            return BatchResponseItem(item_id=item.item_id, error=str(exc))

    results = await asyncio.gather(*(one(item) for item in payload.items))

    counts: dict[str, int] = {"auto_accept": 0, "review": 0, "reject": 0, "error": 0}
    for entry in results:
        if entry.error or entry.result is None:
            counts["error"] += 1
        else:
            counts[entry.result.decision.value] += 1
    return BatchResponse(items=list(results), counts=counts)


@app.get(
    "/v1/review",
    response_model=list[ReviewItemOut],
    tags=["review"],
    dependencies=[Depends(require_api_key)],
)
async def list_review(limit: int = 50, offset: int = 0) -> list[ReviewItemOut]:
    """Items the pipeline was not confident enough to accept, best-first."""
    items = _review().pending(limit=min(limit, 200), offset=offset)
    return [
        ReviewItemOut(
            request_id=item.request_id,
            created_at=item.created_at,
            confidence=item.confidence,
            title=item.title,
            image_url=item.image_url,
            predicted_card_id=item.predicted_id,
            predicted_name=item.result.display_name,
            candidates=[
                CandidateOut(
                    card_id=candidate.card.card_id,
                    display_name=candidate.card.display_name,
                    score=candidate.score,
                    field_scores=candidate.field_scores,
                    penalties=candidate.penalties,
                )
                for candidate in item.result.candidates
            ],
        )
        for item in items
    ]


@app.post(
    "/v1/review/{request_id}/resolve",
    tags=["review"],
    dependencies=[Depends(require_api_key)],
)
async def resolve_review(request_id: str, payload: ResolveRequest) -> dict[str, object]:
    """Record the correct card for a reviewed item, feeding it back as a label."""
    store: CatalogStore = state["store"]  # type: ignore[assignment]
    if store.get(payload.card_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown card_id {payload.card_id}")
    if not _review().resolve(request_id, payload.card_id, payload.resolved_by):
        raise HTTPException(status_code=404, detail="unknown request_id")
    return {"status": "resolved", "request_id": request_id, "card_id": payload.card_id}
