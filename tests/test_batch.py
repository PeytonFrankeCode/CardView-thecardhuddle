"""Bulk processing: the path a 20k/day backfill actually runs through."""

from __future__ import annotations

import asyncio
import csv

import pytest

from cardid.fetch import FetchError, ImageFetcher
from cardid.review import ReviewStore
from cardid.work.batch import completed_ids, read_input, run_batch


class DeadFetcher(ImageFetcher):
    async def fetch(self, url: str):
        raise FetchError("unreachable")

    async def aclose(self) -> None:
        return None


def write_input(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "title", "image_url"])
        writer.writerows(rows)
    return path


def read_output(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["item_id"]: row for row in csv.DictReader(handle)}


@pytest.fixture
def job(tmp_path):
    return tmp_path / "in.csv", tmp_path / "out.csv"


def test_batch_writes_one_row_per_input(identifier, job):
    source, target = job
    write_input(source, [
        ["a1", "2017 Panini Prizm Patrick Mahomes II #269 Silver Prizm", ""],
        ["a2", "2020 Panini Prizm Joe Burrow #307 RC", ""],
    ])
    stats = asyncio.run(run_batch(identifier, source, target, fetcher=DeadFetcher()))
    assert stats.processed == 2
    rows = read_output(target)
    assert rows["a1"]["card_id"] == "mahomes_silver"
    assert rows["a2"]["card_id"] == "burrow_prizm"


def test_one_bad_row_does_not_stop_the_run(identifier, job):
    """A 20k-row job must not lose 19,999 results to a single bad row."""
    source, target = job
    write_input(source, [
        ["good", "2020 Panini Prizm Joe Burrow #307 RC", ""],
        ["bad", "", ""],
        ["also_good", "2017 Panini Prizm Patrick Mahomes II #269 Silver Prizm", ""],
    ])
    stats = asyncio.run(run_batch(identifier, source, target, fetcher=DeadFetcher()))
    assert stats.processed == 3
    assert stats.errors == 1
    rows = read_output(target)
    assert rows["good"]["decision"] == "auto_accept"
    assert rows["also_good"]["decision"] == "auto_accept"
    assert rows["bad"]["decision"] == "error"


def test_rerunning_skips_completed_items(identifier, job):
    source, target = job
    write_input(source, [["a1", "2020 Panini Prizm Joe Burrow #307 RC", ""]])
    asyncio.run(run_batch(identifier, source, target, fetcher=DeadFetcher()))
    again = asyncio.run(run_batch(identifier, source, target, fetcher=DeadFetcher()))
    assert again.processed == 0
    assert again.skipped == 1


def test_no_resume_reprocesses_everything(identifier, job):
    source, target = job
    write_input(source, [["a1", "2020 Panini Prizm Joe Burrow #307 RC", ""]])
    asyncio.run(run_batch(identifier, source, target, fetcher=DeadFetcher()))
    again = asyncio.run(
        run_batch(identifier, source, target, fetcher=DeadFetcher(), resume=False)
    )
    assert again.processed == 1


def test_dead_image_url_falls_back_to_the_title(identifier, job):
    source, target = job
    write_input(source, [
        ["a1", "2020 Panini Prizm Joe Burrow #307 RC", "https://example.com/gone.jpg"],
    ])
    asyncio.run(run_batch(identifier, source, target, fetcher=DeadFetcher()))
    assert read_output(target)["a1"]["card_id"] == "burrow_prizm"


def test_uncertain_items_are_queued_for_review(identifier, job):
    source, target = job
    queue = ReviewStore(":memory:")
    write_input(source, [["a1", "2017 Prizm Mahomes #269 RC", ""]])
    asyncio.run(run_batch(identifier, source, target, fetcher=DeadFetcher(), review=queue))
    assert queue.stats()["pending"] == 1


def test_decision_counts_are_tallied(identifier, job):
    source, target = job
    write_input(source, [
        ["a1", "2017 Panini Prizm Patrick Mahomes II #269 Silver Prizm", ""],
        ["a2", "2017 Prizm Mahomes #269 RC", ""],
        ["a3", "lot of assorted cards", ""],
    ])
    stats = asyncio.run(run_batch(identifier, source, target, fetcher=DeadFetcher()))
    assert (stats.auto_accept, stats.review, stats.reject) == (1, 1, 1)


def test_read_input_accepts_alternate_column_names(tmp_path):
    path = tmp_path / "alt.csv"
    path.write_text("id,title,image\nx1,some title,http://example.com/a.jpg\n",
                    encoding="utf-8")
    rows = list(read_input(path))
    assert rows[0].item_id == "x1"
    assert rows[0].image_url == "http://example.com/a.jpg"


def test_completed_ids_of_a_missing_file_is_empty(tmp_path):
    assert completed_ids(tmp_path / "nope.csv") == set()


def test_a_caller_supplied_fetcher_is_not_closed(identifier, job):
    """It may be shared with other jobs, so the batch must not close it."""
    source, target = job
    write_input(source, [["a1", "2020 Panini Prizm Joe Burrow #307 RC", ""]])

    class TrackingFetcher(DeadFetcher):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    fetcher = TrackingFetcher()
    asyncio.run(run_batch(identifier, source, target, fetcher=fetcher))
    assert fetcher.closed is False
