"""Image preparation: find the card, straighten it, and hash it.

eBay photos are handheld, angled, and often show a graded slab rather than a
raw card. Two things follow from that:

* The card must be located and perspective-corrected before OCR, or stylized
  set names read as garbage.
* A graded slab carries a printed label with the year, set, player and card
  number already spelled out. That label is the single most reliable text on
  the whole image, so it is detected and returned as its own region.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# A raw trading card is 2.5" x 3.5"; a PSA/BGS slab is roughly 3.25" x 5.25".
CARD_ASPECT = 2.5 / 3.5
SLAB_ASPECT = 3.25 / 5.25
# Midpoint between the two aspect ratios: narrower than this reads as a slab.
SLAB_ASPECT_CUTOFF = (CARD_ASPECT + SLAB_ASPECT) / 2
# Fraction of a slab's height taken up by the grading label.
SLAB_LABEL_FRACTION = 0.19

OUTPUT_WIDTH = 700


@dataclass
class CardRegions:
    """The crops pulled out of one photo."""

    card: np.ndarray
    slab_label: np.ndarray | None = None
    is_slab: bool = False
    detected: bool = False
    quad: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def load_image(data: bytes) -> np.ndarray:
    """Decode image bytes to a BGR array, honouring EXIF rotation."""
    with Image.open(io.BytesIO(data)) as pil_image:
        pil_image = pil_image.convert("RGB")
        try:
            from PIL import ImageOps

            pil_image = ImageOps.exif_transpose(pil_image)
        except Exception:  # pragma: no cover - EXIF is best effort
            pass
        rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    points = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered


def find_card_quad(image: np.ndarray) -> np.ndarray | None:
    """Locate the card's four corners, or None if no clean quad is found."""
    height, width = image.shape[:2]
    scale = 900.0 / max(height, width)
    working = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = working.shape[0] * working.shape[1]
    best: np.ndarray | None = None
    best_area = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        # Ignore specks and near-full-frame contours (usually the photo border).
        if area < frame_area * 0.06 or area > frame_area * 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best, best_area = approx, area

    if best is None:
        return None
    quad = _order_corners(best)
    if scale < 1:
        quad /= scale
    return quad


def warp(image: np.ndarray, quad: np.ndarray, aspect: float) -> np.ndarray:
    """Perspective-correct the quad to a upright rectangle of given aspect."""
    width = OUTPUT_WIDTH
    height = int(round(width / aspect))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(quad, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def _quad_aspect(quad: np.ndarray) -> float:
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    width = (top + bottom) / 2
    height = (left + right) / 2
    return float(width / height) if height else CARD_ASPECT


def extract_regions(image: np.ndarray) -> CardRegions:
    """Find, straighten and split the card in a photo."""
    quad = find_card_quad(image)
    if quad is None:
        # No clean quad: fall back to the whole frame. OCR still often works on
        # a tight product shot, it just loses the deskew.
        return CardRegions(
            card=image,
            detected=False,
            notes=["card_not_detected", "using_full_frame"],
        )

    aspect = _quad_aspect(quad)
    is_slab = aspect < SLAB_ASPECT_CUTOFF
    target_aspect = SLAB_ASPECT if is_slab else CARD_ASPECT
    straightened = warp(image, quad, target_aspect)

    if not is_slab:
        return CardRegions(card=straightened, detected=True, quad=quad, notes=["raw_card"])

    split = int(straightened.shape[0] * SLAB_LABEL_FRACTION)
    return CardRegions(
        card=straightened[split:],
        slab_label=straightened[:split],
        is_slab=True,
        detected=True,
        quad=quad,
        notes=["slab_detected"],
    )


def phash(image: np.ndarray, hash_size: int = 16) -> str:
    """DCT perceptual hash, used to dedupe repeated eBay photos.

    eBay reuses the same stock and seller photos constantly, so hashing lets the
    pipeline answer from cache instead of re-running OCR on an image it has
    already identified.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_freq = dct[:hash_size, :hash_size]
    # Exclude the DC term from the median so flat images still produce spread.
    median = np.median(low_freq[1:, 1:])
    bits = (low_freq > median).flatten()
    return "".join(
        f"{int(''.join('1' if bit else '0' for bit in bits[i : i + 4]), 2):x}"
        for i in range(0, len(bits), 4)
    )


def hamming(left: str, right: str) -> int:
    """Bit distance between two hex phashes; 255 if they are incomparable."""
    if not left or not right or len(left) != len(right):
        return 255
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def upscale_for_ocr(image: np.ndarray, min_width: int = 1000) -> np.ndarray:
    """Enlarge small crops so OCR has enough pixels on stylized set names."""
    if image.shape[1] >= min_width:
        return image
    scale = min_width / image.shape[1]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
