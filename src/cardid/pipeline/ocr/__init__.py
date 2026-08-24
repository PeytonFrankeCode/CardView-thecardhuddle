"""OCR backend selection."""

from __future__ import annotations

import logging

from .base import NullOcrBackend, OcrBackend, OcrLine, OcrResult

log = logging.getLogger(__name__)

__all__ = ["NullOcrBackend", "OcrBackend", "OcrLine", "OcrResult", "get_ocr_backend"]


def get_ocr_backend(name: str = "auto", **kwargs: object) -> OcrBackend:
    """Return an OCR backend by name.

    ``auto`` prefers PaddleOCR, falls back to Tesseract, and finally to the null
    backend so a missing optional dependency degrades the service instead of
    breaking it.
    """
    name = (name or "auto").lower()

    if name in ("paddle", "paddleocr", "auto"):
        try:
            from .paddle import PaddleOcrBackend

            return PaddleOcrBackend(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            if name != "auto":
                raise
            log.warning("PaddleOCR unavailable (%s); trying Tesseract", exc)

    if name in ("tesseract", "auto"):
        try:
            from .tesseract import TesseractBackend

            return TesseractBackend(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            if name != "auto":
                raise
            log.warning("Tesseract unavailable (%s); falling back to null OCR", exc)

    if name == "null":
        return NullOcrBackend()

    log.warning("No OCR backend available; running title-only")
    return NullOcrBackend()
