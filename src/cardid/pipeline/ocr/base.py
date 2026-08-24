"""OCR backend interface.

Backends are pluggable because the right engine depends on where this runs: a
GPU box should use PaddleOCR, a small VPS can use Tesseract, and CI needs
neither. Everything downstream consumes :class:`OcrResult`, so swapping engines
never touches the matcher.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field


class OcrLine(BaseModel):
    text: str
    confidence: float = 0.0
    box: list[list[float]] = Field(default_factory=list)


class OcrResult(BaseModel):
    """Everything one OCR pass produced."""

    text: str = ""
    lines: list[OcrLine] = Field(default_factory=list)
    confidence: float = 0.0
    backend: str = "none"

    @classmethod
    def from_lines(cls, lines: list[OcrLine], backend: str) -> OcrResult:
        kept = [line for line in lines if line.text.strip()]
        text = " ".join(line.text.strip() for line in kept)
        confidence = (
            sum(line.confidence for line in kept) / len(kept) if kept else 0.0
        )
        return cls(text=text, lines=kept, confidence=confidence, backend=backend)


@runtime_checkable
class OcrBackend(Protocol):
    name: str

    def read(self, image: np.ndarray) -> OcrResult:
        """Extract text from a BGR image array."""
        ...


class NullOcrBackend:
    """Returns nothing. Lets the service run title-only, and keeps tests fast."""

    name = "null"

    def read(self, image: np.ndarray) -> OcrResult:  # noqa: ARG002
        return OcrResult(backend=self.name)
