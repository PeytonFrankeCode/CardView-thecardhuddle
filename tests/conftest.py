"""Shared fixtures: a small in-memory catalog and a ready-to-use identifier."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cardid.cache import ResultCache
from cardid.catalog.store import CatalogStore
from cardid.models import CatalogCard
from cardid.pipeline.identify import CardIdentifier
from cardid.pipeline.ocr.base import NullOcrBackend
from cardid.review import ReviewStore

SEED_CARDS = [
    CatalogCard(card_id="mahomes_base", year=2017, brand="panini", set_name="prizm",
                player="patrick mahomes ii", card_number="269", parallel="base",
                is_rookie=True, team="kansas city chiefs"),
    CatalogCard(card_id="mahomes_silver", year=2017, brand="panini", set_name="prizm",
                player="patrick mahomes ii", card_number="269", parallel="silver",
                is_rookie=True, team="kansas city chiefs"),
    CatalogCard(card_id="mahomes_gold", year=2017, brand="panini", set_name="prizm",
                player="patrick mahomes ii", card_number="269", parallel="gold",
                print_run=10, is_rookie=True, team="kansas city chiefs"),
    CatalogCard(card_id="allen_ticket", year=2018, brand="panini", set_name="contenders",
                subset="rookie ticket", player="josh allen", card_number="101",
                parallel="base", is_rookie=True, is_autograph=True,
                team="buffalo bills"),
    CatalogCard(card_id="herbert_optic", year=2020, brand="panini", set_name="optic",
                subset="rated rookie", player="justin herbert", card_number="158",
                parallel="base", is_rookie=True, team="los angeles chargers"),
    CatalogCard(card_id="burrow_prizm", year=2020, brand="panini", set_name="prizm",
                player="joe burrow", card_number="307", parallel="base",
                is_rookie=True, team="cincinnati bengals"),
]


@pytest.fixture
def store() -> CatalogStore:
    catalog = CatalogStore(":memory:")
    catalog.add_cards([card.model_copy(deep=True) for card in SEED_CARDS])
    return catalog


@pytest.fixture
def identifier(store: CatalogStore) -> CardIdentifier:
    return CardIdentifier(
        store=store,
        ocr_backend=NullOcrBackend(),
        cache=ResultCache(":memory:"),
    )


@pytest.fixture
def review() -> ReviewStore:
    return ReviewStore(":memory:")


@pytest.fixture
def card_photo() -> bytes:
    """A synthetic photo of a card-shaped rectangle on a dark background."""
    frame = np.full((900, 900, 3), 25, np.uint8)
    card = np.full((490, 350, 3), 235, np.uint8)
    cv2.putText(card, "MAHOMES", (18, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    frame[200:690, 275:625] = card
    return cv2.imencode(".jpg", frame)[1].tobytes()


@pytest.fixture
def slab_photo() -> bytes:
    """A synthetic photo shaped like a graded slab (taller and narrower)."""
    frame = np.full((1000, 900, 3), 25, np.uint8)
    slab = np.full((630, 390, 3), 240, np.uint8)
    slab[:120] = 250  # label band
    frame[150:780, 255:645] = slab
    return cv2.imencode(".jpg", frame)[1].tobytes()
