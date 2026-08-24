"""HTTP surface: the contract the website depends on."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cardid.api import app as app_module
from cardid.cache import ResultCache
from cardid.fetch import FetchError, ImageFetcher
from cardid.pipeline.identify import CardIdentifier
from cardid.pipeline.ocr.base import NullOcrBackend
from cardid.review import ReviewStore


class StubFetcher(ImageFetcher):
    """Never touches the network; every fetch fails as if the URL were dead."""

    async def fetch(self, url: str):
        raise FetchError("stubbed fetch")

    async def aclose(self) -> None:
        return None


@pytest.fixture
def client(store):
    review = ReviewStore(":memory:")
    app_module.state.clear()
    app_module.state.update({
        "store": store,
        "review": review,
        "identifier": CardIdentifier(store=store, ocr_backend=NullOcrBackend(),
                                     cache=ResultCache(":memory:")),
        "fetcher": StubFetcher(),
    })
    with TestClient(app_module.app) as test_client:
        test_client.review = review
        yield test_client
    app_module.state.clear()


def test_healthz_reports_catalog_size(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["catalog_cards"] == 6


def test_identify_by_title_returns_the_card(client):
    response = client.post("/v1/identify", json={
        "title": "2017 Panini Prizm Patrick Mahomes II #269 RC Silver Prizm PSA 10"
    })
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "auto_accept"
    assert body["card_id"] == "mahomes_silver"
    assert "Silver" in body["display_name"]


def test_identify_requires_some_input(client):
    assert client.post("/v1/identify", json={}).status_code == 422


def test_dead_image_url_falls_back_to_the_title(client):
    response = client.post("/v1/identify", json={
        "image_url": "https://example.com/gone.jpg",
        "title": "2020 Panini Prizm Joe Burrow #307 RC",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["card_id"] == "burrow_prizm"
    assert any("fetch failed" in reason for reason in body["reasons"])


def test_dead_image_url_without_a_title_is_an_error(client):
    response = client.post("/v1/identify", json={"image_url": "https://example.com/gone.jpg"})
    assert response.status_code == 400


def test_upload_endpoint_accepts_a_photo(client, card_photo):
    response = client.post(
        "/v1/identify/upload",
        files={"file": ("card.jpg", card_photo, "image/jpeg")},
        data={"title": "2018 Panini Contenders Josh Allen Rookie Ticket Auto #101"},
    )
    assert response.status_code == 200
    assert response.json()["card_id"] == "allen_ticket"


def test_empty_upload_is_rejected(client):
    response = client.post("/v1/identify/upload",
                           files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 422


def test_batch_identifies_many_and_counts_outcomes(client):
    response = client.post("/v1/identify/batch", json={"items": [
        {"item_id": "1", "title": "2017 Panini Prizm Patrick Mahomes II #269 Silver Prizm"},
        {"item_id": "2", "title": "2017 Prizm Mahomes #269 RC"},
        {"item_id": "3", "title": "lot of assorted cards"},
        {"item_id": "4"},
    ]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 4
    assert body["counts"]["auto_accept"] == 1
    assert body["counts"]["review"] == 1
    by_id = {item["item_id"]: item for item in body["items"]}
    assert by_id["4"]["error"] == "no usable input"


def test_uncertain_results_are_queued_for_review_automatically(client):
    client.post("/v1/identify", json={"title": "2017 Prizm Mahomes #269 RC"})
    queue = client.get("/v1/review").json()
    assert len(queue) == 1
    assert queue[0]["candidates"]


def test_confident_results_are_not_queued(client):
    client.post("/v1/identify", json={
        "title": "2017 Panini Prizm Patrick Mahomes II #269 Silver Prizm"})
    assert client.get("/v1/review").json() == []


def test_resolving_a_review_item_records_the_decision(client):
    client.post("/v1/identify", json={"title": "2017 Prizm Mahomes #269 RC"})
    request_id = client.get("/v1/review").json()[0]["request_id"]
    response = client.post(f"/v1/review/{request_id}/resolve",
                           json={"card_id": "mahomes_gold"})
    assert response.status_code == 200
    assert client.get("/v1/review").json() == []
    assert client.review.stats()["labels"] == 1


def test_resolving_with_an_unknown_card_id_is_rejected(client):
    client.post("/v1/identify", json={"title": "2017 Prizm Mahomes #269 RC"})
    request_id = client.get("/v1/review").json()[0]["request_id"]
    response = client.post(f"/v1/review/{request_id}/resolve",
                           json={"card_id": "does_not_exist"})
    assert response.status_code == 404


def test_resolving_an_unknown_request_is_rejected(client):
    response = client.post("/v1/review/nope/resolve", json={"card_id": "mahomes_base"})
    assert response.status_code == 404


def test_stats_endpoint_reports_all_three_stores(client):
    body = client.get("/v1/stats").json()
    assert set(body) == {"catalog", "cache", "review"}


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("CARDID_API_KEY", "secret")
    assert client.post("/v1/identify", json={"title": "x"}).status_code == 401
    response = client.post("/v1/identify", json={
        "title": "2020 Panini Prizm Joe Burrow #307 RC"},
        headers={"X-API-Key": "secret"})
    assert response.status_code == 200


def test_healthz_stays_open_without_a_key(client, monkeypatch):
    """Load balancers must still be able to probe the service."""
    monkeypatch.setenv("CARDID_API_KEY", "secret")
    assert client.get("/healthz").status_code == 200
