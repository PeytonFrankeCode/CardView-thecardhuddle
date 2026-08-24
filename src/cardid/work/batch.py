"""Bulk identification for backfilling a sold-listing table.

Sized for the real workload: 20,000 photos a day is only ~0.25/second averaged
out, so throughput is never the hard part — robustness is. A run over tens of
thousands of rows will hit dead image URLs, oversized files and malformed rows,
and it must not lose 19,000 good results because of them.

So this worker:

* streams input and appends output row by row, never holding the whole job in
  memory;
* is resumable — re-running skips item_ids already present in the output, which
  matters when a run is interrupted three hours in;
* isolates failures per item, recording the error and continuing;
* bounds concurrency so a batch cannot exhaust sockets or memory.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..fetch import FetchError, ImageFetcher
from ..models import Decision
from ..pipeline.identify import CardIdentifier
from ..review import ReviewStore

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "item_id", "decision", "confidence", "card_id", "display_name",
    "year", "brand", "set_name", "player", "card_number", "parallel", "error",
]


@dataclass
class BatchStats:
    processed: int = 0
    auto_accept: int = 0
    review: int = 0
    reject: int = 0
    errors: int = 0
    skipped: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "auto_accept": self.auto_accept,
            "review": self.review,
            "reject": self.reject,
            "errors": self.errors,
            "skipped_already_done": self.skipped,
        }


@dataclass
class BatchInput:
    item_id: str
    image_url: str | None = None
    title: str | None = None


def read_input(path: str | Path) -> Iterator[BatchInput]:
    """Stream input rows. Requires an ``item_id`` plus a title and/or image_url."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            item_id = (row.get("item_id") or row.get("id") or "").strip()
            yield BatchInput(
                item_id=item_id or f"row_{index}",
                image_url=(row.get("image_url") or row.get("image") or "").strip() or None,
                title=(row.get("title") or "").strip() or None,
            )


def completed_ids(path: str | Path) -> set[str]:
    """item_ids already written, so an interrupted run can resume."""
    path = Path(path)
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row.get("item_id") or "").strip()
            for row in csv.DictReader(handle)
            if (row.get("item_id") or "").strip()
        }


async def run_batch(
    identifier: CardIdentifier,
    input_path: str | Path,
    output_path: str | Path,
    fetcher: ImageFetcher | None = None,
    review: ReviewStore | None = None,
    concurrency: int = 8,
    resume: bool = True,
    progress_every: int = 500,
) -> BatchStats:
    """Identify every row in ``input_path``, appending results to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = completed_ids(output_path) if resume else set()
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    # Only close a fetcher we created; a caller-supplied one may be shared with
    # other jobs and outlive this run.
    owns_fetcher = fetcher is None
    fetcher = fetcher or ImageFetcher()
    stats = BatchStats()
    semaphore = asyncio.Semaphore(concurrency)
    # One writer, one lock: CSV rows must not interleave mid-line.
    write_lock = asyncio.Lock()

    handle = output_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
    if write_header:
        writer.writeheader()
        handle.flush()

    async def process(item: BatchInput) -> None:
        async with semaphore:
            row = {"item_id": item.item_id}
            try:
                image_bytes = None
                if item.image_url:
                    try:
                        image_bytes = (await fetcher.fetch(item.image_url)).data
                    except FetchError as exc:
                        if not item.title:
                            raise
                        log.debug("fetch failed for %s, using title: %s", item.item_id, exc)

                if image_bytes is None and not item.title:
                    raise ValueError("no usable input")

                result = await asyncio.to_thread(
                    identifier.identify, image_bytes=image_bytes, title=item.title
                )
                if review is not None and result.decision is not Decision.AUTO_ACCEPT:
                    review.enqueue(result, image_url=item.image_url, title=item.title)

                attrs = result.fused
                row.update({
                    "decision": result.decision.value,
                    "confidence": round(result.confidence, 4),
                    "card_id": result.card.card_id if result.card else "",
                    "display_name": result.display_name or "",
                    "year": attrs.year or "",
                    "brand": attrs.brand or "",
                    "set_name": attrs.set_name or "",
                    "player": attrs.player or "",
                    "card_number": attrs.card_number or "",
                    "parallel": attrs.parallel or "",
                    "error": "",
                })
                setattr(stats, result.decision.value, getattr(stats, result.decision.value) + 1)
            except Exception as exc:
                # One bad row must never end the run.
                log.warning("item %s failed: %s", item.item_id, exc)
                row.update({"decision": "error", "confidence": 0.0, "error": str(exc)})
                stats.errors += 1

            async with write_lock:
                writer.writerow(row)
                stats.processed += 1
                if stats.processed % progress_every == 0:
                    handle.flush()
                    log.info("processed %s items", stats.processed)

    try:
        pending: list[asyncio.Task] = []
        for item in read_input(input_path):
            if item.item_id in done:
                stats.skipped += 1
                continue
            pending.append(asyncio.create_task(process(item)))
            # Keep the task list bounded so a huge input file does not create
            # a million task objects up front.
            if len(pending) >= concurrency * 20:
                await asyncio.gather(*pending)
                pending = []
        if pending:
            await asyncio.gather(*pending)
    finally:
        handle.flush()
        handle.close()
        if owns_fetcher:
            await fetcher.aclose()

    return stats
