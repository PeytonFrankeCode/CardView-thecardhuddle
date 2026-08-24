"""Command line interface.

Everything you need to run the system day to day: load a catalog, identify a
card, backfill a table, calibrate thresholds, and serve the API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .cache import ResultCache
from .calibrate import format_report, observe, read_labels, sweep
from .catalog.ingest import bootstrap_from_titles, import_rows
from .catalog.store import CatalogStore
from .config import settings
from .pipeline.identify import CardIdentifier
from .review import ReviewStore
from .work.batch import run_batch


def _identifier(catalog_path: str) -> CardIdentifier:
    store = CatalogStore(catalog_path)
    if store.count() == 0:
        print(
            f"warning: catalog at {catalog_path} is empty — "
            "load one with `cardid import-catalog` or `cardid bootstrap`",
            file=sys.stderr,
        )
    return CardIdentifier(
        store=store,
        cache=ResultCache(settings.cache_path, enabled=settings.cache_enabled),
    )


def cmd_import_catalog(args: argparse.Namespace) -> int:
    store = CatalogStore(args.catalog)
    count = import_rows(store, args.path)
    print(f"imported {count} cards into {args.catalog} (total {store.count()})")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Build a catalog from a newline-delimited file of sold-listing titles."""
    titles = [
        line.strip()
        for line in Path(args.titles).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    store = CatalogStore(args.catalog)
    stats = bootstrap_from_titles(
        store, titles, min_occurrences=args.min_occurrences
    )
    print(json.dumps(stats, indent=2))
    print(f"catalog now holds {store.count()} cards")
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    identifier = _identifier(args.catalog)
    image_bytes = Path(args.file).read_bytes() if args.file else None
    result = identifier.identify(image_bytes=image_bytes, title=args.title)

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"decision   : {result.decision.value}")
    print(f"confidence : {result.confidence}")
    print(f"card       : {result.display_name or '(none)'}")
    if result.card:
        print(f"card_id    : {result.card.card_id}")
    if result.candidates:
        print("\ncandidates:")
        for candidate in result.candidates:
            print(f"  {candidate.score:>6.3f}  {candidate.card.display_name}")
    if result.reasons:
        print("\nreasons:")
        for reason in result.reasons:
            print(f"  - {reason}")
    return 0 if result.card else 1


def cmd_batch(args: argparse.Namespace) -> int:
    identifier = _identifier(args.catalog)
    review = ReviewStore(args.review) if args.review else None
    stats = asyncio.run(
        run_batch(
            identifier,
            args.input,
            args.output,
            review=review,
            concurrency=args.concurrency,
            resume=not args.no_resume,
        )
    )
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    identifier = _identifier(args.catalog)
    observations = observe(identifier, read_labels(args.labels))
    report = sweep(
        observations,
        target_precision=args.target,
        min_accepted=args.min_accepted,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0 if report.recommended_threshold is not None else 2


def cmd_stats(args: argparse.Namespace) -> int:
    store = CatalogStore(args.catalog)
    print(json.dumps({
        "catalog_cards": store.count(),
        "cache": ResultCache(settings.cache_path).stats(),
        "review": ReviewStore(args.review or "data/review.db").stats(),
    }, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "cardid.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cardid", description="Identify football cards from photos and listing titles."
    )
    parser.add_argument("--catalog", default=settings.catalog_path, help="catalog database path")
    parser.add_argument("--log-level", default="warning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("import-catalog", help="load a structured catalog export")
    p.add_argument("path", help=".csv, .json, or .jsonl catalog file")
    p.set_defaults(func=cmd_import_catalog)

    p = subparsers.add_parser("bootstrap", help="build a catalog from sold-listing titles")
    p.add_argument("titles", help="file with one listing title per line")
    p.add_argument("--min-occurrences", type=int, default=1,
                   help="drop cards seen fewer than this many times")
    p.set_defaults(func=cmd_bootstrap)

    p = subparsers.add_parser("identify", help="identify a single card")
    p.add_argument("--title", help="listing title")
    p.add_argument("--file", help="path to a card photo")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_identify)

    p = subparsers.add_parser("batch", help="identify a CSV of listings")
    p.add_argument("input", help="CSV with item_id,title,image_url")
    p.add_argument("output", help="CSV to append results to")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--review", default="data/review.db", help="queue uncertain items here")
    p.add_argument("--no-resume", action="store_true", help="reprocess already-done items")
    p.set_defaults(func=cmd_batch)

    p = subparsers.add_parser("calibrate", help="tune the auto-accept threshold")
    p.add_argument("labels", help="CSV/JSONL with true_card_id plus title/image_path")
    p.add_argument("--target", type=float, default=0.99, help="precision target")
    p.add_argument("--min-accepted", type=int, default=20,
                   help="ignore thresholds that accept fewer than this many examples")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    p = subparsers.add_parser("stats", help="show catalog, cache and review counts")
    p.add_argument("--review", default="data/review.db")
    p.set_defaults(func=cmd_stats)

    p = subparsers.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
