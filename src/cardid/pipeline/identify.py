"""The identification pipeline: photo (and optional listing title) in, card out.

Order of operations matters. Cheap, high-signal steps run first so the expensive
ones are skipped whenever possible:

1. Hash the bytes and check the cache.
2. Locate and straighten the card; split off a slab label if present.
3. OCR the slab label and the card face as separate passes.
4. Parse every text source, including the eBay listing title.
5. Fuse the extractions in trust order.
6. Generate catalog candidates, score them, and gate on confidence.

The listing title is treated as a first-class input rather than a hint. On a
sold-listing feed it is usually the strongest single signal available, and
fusing it with the image is what lifts accuracy past what either reaches alone.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager

from ..cache import ResultCache, sha256_bytes
from ..catalog.store import CatalogStore
from ..config import Settings, settings as default_settings
from ..models import (
    CardAttributes,
    Decision,
    Extraction,
    Identification,
    Source,
)
from . import image as image_utils
from .confidence import compute_confidence, decide
from .match import rank_candidates
from .ocr import OcrBackend, get_ocr_backend
from .parse import parse_text


class CardIdentifier:
    """Stateful pipeline. Build one per process and reuse it across requests."""

    def __init__(
        self,
        store: CatalogStore,
        ocr_backend: OcrBackend | None = None,
        cache: ResultCache | None = None,
        config: Settings | None = None,
    ) -> None:
        self.settings = config or default_settings
        self.store = store
        self.ocr = ocr_backend or get_ocr_backend(self.settings.ocr_backend)
        self.cache = cache or ResultCache(
            self.settings.cache_path, enabled=self.settings.cache_enabled
        )

    # --- helpers ---------------------------------------------------------

    @contextmanager
    def _timed(self, timings: dict[str, float], label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            timings[label] = round((time.perf_counter() - start) * 1000, 2)

    def _player_index(self) -> list[tuple[str, str]]:
        return self.store.player_index()

    # --- main entry point -------------------------------------------------

    def identify(
        self,
        image_bytes: bytes | None = None,
        title: str | None = None,
        request_id: str | None = None,
        use_cache: bool = True,
    ) -> Identification:
        """Identify one card from a photo, a listing title, or both."""
        request_id = request_id or uuid.uuid4().hex
        timings: dict[str, float] = {}
        extractions: list[Extraction] = []
        player_index = self._player_index()

        if not image_bytes and not title:
            return Identification(
                request_id=request_id,
                decision=Decision.REJECT,
                confidence=0.0,
                reasons=["no image and no title supplied"],
            )

        content_hash = sha256_bytes(image_bytes) if image_bytes else None
        image_phash: str | None = None
        regions: image_utils.CardRegions | None = None
        notes: list[str] = []

        if image_bytes:
            with self._timed(timings, "decode"):
                try:
                    frame = image_utils.load_image(image_bytes)
                except Exception as exc:
                    return Identification(
                        request_id=request_id,
                        decision=Decision.REJECT,
                        confidence=0.0,
                        reasons=[f"could not decode image: {exc}"],
                    )

            with self._timed(timings, "phash"):
                image_phash = image_utils.phash(frame)

            # A cached answer is only reusable when the title matches too: the
            # same photo can be relisted with a corrected title.
            if use_cache and not title:
                cached = self.cache.get(content_hash, image_phash)
                if cached:
                    cached.request_id = request_id
                    cached.cache_hit = True
                    return cached

            with self._timed(timings, "detect"):
                regions = image_utils.extract_regions(frame)
                notes.extend(regions.notes)

        # --- text extraction ---
        if title:
            extractions.append(
                parse_text(title, Source.TITLE, player_index=player_index)
            )

        if regions is not None:
            with self._timed(timings, "ocr"):
                if regions.slab_label is not None:
                    label_result = self.ocr.read(
                        image_utils.upscale_for_ocr(regions.slab_label)
                    )
                    if label_result.text:
                        label_extraction = parse_text(
                            label_result.text, Source.OCR, player_index=player_index
                        )
                        # A slab label is printed, authoritative text; trust it
                        # above a seller-written title.
                        label_extraction.confidence = min(
                            1.0, label_extraction.confidence + 0.10
                        )
                        label_extraction.tokens_used.append("slab_label")
                        extractions.insert(0, label_extraction)

                face_result = self.ocr.read(image_utils.upscale_for_ocr(regions.card))
                if face_result.text:
                    extractions.append(
                        parse_text(face_result.text, Source.OCR, player_index=player_index)
                    )
                elif self.ocr.name == "null":
                    notes.append("ocr_disabled")
                else:
                    notes.append("no_text_read_from_card")

        # --- fuse in trust order: slab label, then title, then card face ---
        fused = CardAttributes()
        for extraction in sorted(extractions, key=lambda e: -e.confidence):
            fused = fused.merge(extraction.attributes)

        if fused.is_empty():
            return Identification(
                request_id=request_id,
                decision=Decision.REJECT,
                confidence=0.0,
                extractions=extractions,
                fused=fused,
                reasons=notes + ["no identifying attributes extracted"],
                timings_ms=timings,
                image_phash=image_phash,
            )

        # --- catalog match ---
        with self._timed(timings, "candidates"):
            candidates = self.store.candidates(fused, limit=self.settings.max_candidates)
        with self._timed(timings, "rank"):
            ranked = rank_candidates(fused, candidates, top_k=self.settings.top_k)

        confidence, reasons = compute_confidence(
            ranked, extractions, fused, self.settings.min_margin
        )
        decision = decide(
            confidence,
            self.settings.auto_accept_threshold,
            self.settings.review_threshold,
        )

        top = ranked[0] if ranked else None
        result = Identification(
            request_id=request_id,
            decision=decision,
            confidence=confidence,
            card=top.card if top else None,
            display_name=top.card.display_name if top else None,
            runner_up=ranked[1] if len(ranked) > 1 else None,
            candidates=ranked,
            extractions=extractions,
            fused=fused,
            reasons=notes + reasons,
            timings_ms=timings,
            image_phash=image_phash,
        )

        if use_cache and not title:
            self.cache.put(result, content_hash, image_phash)
        return result
