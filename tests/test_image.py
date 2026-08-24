"""Card detection, slab splitting, and perceptual hashing."""

from __future__ import annotations

import cv2
import numpy as np

from cardid.pipeline.image import (
    extract_regions,
    hamming,
    load_image,
    phash,
    upscale_for_ocr,
)


def test_card_is_detected_and_straightened(card_photo):
    regions = extract_regions(load_image(card_photo))
    assert regions.detected
    assert not regions.is_slab
    height, width = regions.card.shape[:2]
    assert 0.6 < width / height < 0.8  # roughly a 2.5x3.5 card


def test_slab_is_detected_and_label_split_off(slab_photo):
    regions = extract_regions(load_image(slab_photo))
    assert regions.is_slab
    assert regions.slab_label is not None
    assert regions.slab_label.shape[0] > 0


def test_undetectable_image_falls_back_to_the_full_frame():
    noise = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    regions = extract_regions(noise)
    assert regions.card.shape == noise.shape
    assert "card_not_detected" in regions.notes


def test_phash_is_stable_and_discriminating(card_photo, slab_photo):
    left = phash(load_image(card_photo))
    assert hamming(left, left) == 0
    assert hamming(left, phash(load_image(slab_photo))) > 5


def test_phash_survives_a_resize(card_photo):
    """Cache hits depend on this: eBay re-encodes and resizes images."""
    frame = load_image(card_photo)
    smaller = cv2.resize(frame, None, fx=0.5, fy=0.5)
    assert hamming(phash(frame), phash(smaller)) <= 6


def test_hamming_rejects_incomparable_hashes():
    assert hamming("abc", "") == 255
    assert hamming("abc", "abcdef") == 255


def test_upscale_only_enlarges_small_crops():
    small = np.zeros((50, 100, 3), np.uint8)
    assert upscale_for_ocr(small, min_width=400).shape[1] >= 400
    large = np.zeros((500, 1200, 3), np.uint8)
    assert upscale_for_ocr(large, min_width=400).shape[1] == 1200
