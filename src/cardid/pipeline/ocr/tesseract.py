"""Tesseract OCR backend.

Needs the `tesseract-ocr` system package plus `pytesseract`. Cheap to run and
adequate on slab labels, which are clean printed text; weaker than PaddleOCR on
stylized card-face fonts.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import OcrLine, OcrResult


class TesseractBackend:
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 11) -> None:
        import pytesseract  # imported lazily so the dep stays optional

        self._pytesseract = pytesseract
        self.lang = lang
        # PSM 11 = sparse text, which suits card faces where text is scattered.
        self.config = f"--oem 3 --psm {psm}"

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.bilateralFilter(gray, 5, 50, 50)

    def read(self, image: np.ndarray) -> OcrResult:
        data = self._pytesseract.image_to_data(
            self._preprocess(image),
            lang=self.lang,
            config=self.config,
            output_type=self._pytesseract.Output.DICT,
        )
        lines: list[OcrLine] = []
        for text, conf in zip(data["text"], data["conf"], strict=False):
            text = (text or "").strip()
            try:
                confidence = float(conf)
            except (TypeError, ValueError):
                confidence = -1.0
            if text and confidence >= 0:
                lines.append(OcrLine(text=text, confidence=confidence / 100.0))
        return OcrResult.from_lines(lines, self.name)
