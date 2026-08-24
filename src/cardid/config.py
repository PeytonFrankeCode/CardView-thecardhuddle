"""Runtime configuration, overridable by environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CARDID_", env_file=".env", extra="ignore")

    catalog_path: str = "data/catalog.db"
    cache_path: str = "data/cache.db"

    ocr_backend: str = "auto"
    ocr_use_gpu: bool = False

    # Decision thresholds. These are the dial that trades coverage for
    # precision: raise auto_accept_threshold to accept fewer cards but be more
    # often right. Tune them on labeled data with scripts/calibrate.py rather
    # than guessing — the defaults are a starting point, not a promise.
    auto_accept_threshold: float = 0.90
    review_threshold: float = 0.45

    # Below this margin between the top two candidates, a result is sent to
    # review even if its raw score is high: it means two catalog rows fit the
    # evidence almost equally well.
    min_margin: float = 0.08

    max_candidates: int = 200
    top_k: int = 5

    max_image_bytes: int = 12 * 1024 * 1024
    fetch_timeout_seconds: float = 12.0
    max_concurrent_fetches: int = 16

    cache_enabled: bool = True
    phash_match_distance: int = 6


settings = Settings()
