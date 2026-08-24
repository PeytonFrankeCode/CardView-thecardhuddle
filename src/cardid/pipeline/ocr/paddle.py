"""PaddleOCR backend — the recommended engine for production.

PaddleOCR handles the stylized, low-contrast, foil-over-foil text on modern
card faces markedly better than Tesseract, which is where most identification
failures come from. It is heavier: install `paddlepaddle` (GPU build if you
have one) and `paddleocr`, and expect a few hundred MB of model weights.
"""

from __future__ import annotations

import numpy as np

from .base import OcrLine, OcrResult


class PaddleOcrBackend:
    name = "paddleocr"

    def __init__(self, lang: str = "en", use_gpu: bool = False) -> None:
        from paddleocr import PaddleOCR  # imported lazily; heavy optional dep

        self._engine = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )

    def read(self, image: np.ndarray) -> OcrResult:
        raw = self._engine.ocr(image, cls=True)
        lines: list[OcrLine] = []
        # PaddleOCR returns [[ [box, (text, score)], ... ]]; empty pages give [None].
        for page in raw or []:
            for entry in page or []:
                box, (text, score) = entry[0], entry[1]
                lines.append(
                    OcrLine(
                        text=text,
                        confidence=float(score),
                        box=[[float(x), float(y)] for x, y in box],
                    )
                )
        return OcrResult.from_lines(lines, self.name)
