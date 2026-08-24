"""Threshold calibration: turning a precision target into a setting.

A single "accuracy" number hides the decision that actually matters. What the
pipeline controls is a threshold, and the threshold trades two quantities:

* **precision** — of the cards auto-accepted, how many were right;
* **coverage** — what share of the feed got auto-accepted at all.

You cannot pick both. This module measures the curve on labeled data and finds
the lowest threshold that still meets a precision target, because the lowest
such threshold is the one that sends the fewest cards to human review.

Run it against ground truth you trust, re-run it whenever the catalog or the
OCR backend changes, and put the number it prints into CARDID_AUTO_ACCEPT_THRESHOLD.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .models import Decision
from .pipeline.identify import CardIdentifier


@dataclass
class LabeledExample:
    true_card_id: str
    title: str | None = None
    image_path: str | None = None


@dataclass
class Observation:
    """One scored prediction against its label."""

    confidence: float
    predicted_id: str | None
    true_card_id: str
    correct: bool


@dataclass
class ThresholdPoint:
    threshold: float
    precision: float
    coverage: float
    accepted: int
    correct: int
    total: int


@dataclass
class CalibrationReport:
    points: list[ThresholdPoint] = field(default_factory=list)
    recommended_threshold: float | None = None
    target_precision: float = 0.99
    achieved_precision: float | None = None
    achieved_coverage: float | None = None
    total_examples: int = 0
    # Precision if every prediction were accepted, i.e. no gating at all.
    ungated_precision: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "target_precision": self.target_precision,
            "recommended_threshold": self.recommended_threshold,
            "achieved_precision": self.achieved_precision,
            "achieved_coverage": self.achieved_coverage,
            "ungated_precision": self.ungated_precision,
            "total_examples": self.total_examples,
            "curve": [
                {
                    "threshold": round(point.threshold, 3),
                    "precision": round(point.precision, 4),
                    "coverage": round(point.coverage, 4),
                    "accepted": point.accepted,
                    "correct": point.correct,
                }
                for point in self.points
            ],
        }


def read_labels(path: str | Path) -> Iterator[LabeledExample]:
    """Read labeled examples from CSV or JSONL.

    Expected columns: ``true_card_id`` plus ``title`` and/or ``image_path``.
    """
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows: Iterable[dict] = (json.loads(line) for line in handle if line.strip())
            for row in rows:
                yield LabeledExample(
                    true_card_id=str(row["true_card_id"]),
                    title=row.get("title"),
                    image_path=row.get("image_path"),
                )
        return
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield LabeledExample(
                true_card_id=str(row["true_card_id"]),
                title=(row.get("title") or "").strip() or None,
                image_path=(row.get("image_path") or "").strip() or None,
            )


def observe(
    identifier: CardIdentifier, examples: Iterable[LabeledExample]
) -> list[Observation]:
    """Run the pipeline over labeled examples and record each outcome."""
    observations: list[Observation] = []
    for example in examples:
        image_bytes = None
        if example.image_path:
            path = Path(example.image_path)
            if path.exists():
                image_bytes = path.read_bytes()

        result = identifier.identify(
            image_bytes=image_bytes, title=example.title, use_cache=False
        )
        predicted = result.card.card_id if result.card else None
        observations.append(
            Observation(
                confidence=result.confidence,
                predicted_id=predicted,
                true_card_id=example.true_card_id,
                correct=predicted == example.true_card_id,
            )
        )
    return observations


def sweep(
    observations: list[Observation],
    target_precision: float = 0.99,
    steps: int = 101,
    min_accepted: int = 1,
) -> CalibrationReport:
    """Measure precision and coverage across the threshold range.

    ``min_accepted`` guards against a threshold that looks perfect only because
    it accepted a handful of examples; raise it on larger label sets.
    """
    report = CalibrationReport(
        target_precision=target_precision, total_examples=len(observations)
    )
    if not observations:
        return report

    total = len(observations)
    report.ungated_precision = round(
        sum(1 for o in observations if o.correct) / total, 4
    )

    best: ThresholdPoint | None = None
    for step in range(steps):
        threshold = step / (steps - 1)
        accepted = [o for o in observations if o.confidence >= threshold]
        correct = sum(1 for o in accepted if o.correct)
        precision = correct / len(accepted) if accepted else 1.0
        point = ThresholdPoint(
            threshold=threshold,
            precision=precision,
            coverage=len(accepted) / total,
            accepted=len(accepted),
            correct=correct,
            total=total,
        )
        report.points.append(point)

        # Lowest qualifying threshold wins: it keeps the most volume automated
        # while still clearing the precision bar.
        if (
            precision >= target_precision
            and len(accepted) >= min_accepted
            and best is None
        ):
            best = point

    if best is not None:
        report.recommended_threshold = round(best.threshold, 3)
        report.achieved_precision = round(best.precision, 4)
        report.achieved_coverage = round(best.coverage, 4)
    return report


def format_report(report: CalibrationReport) -> str:
    """Render a report for a terminal."""
    lines = [
        f"examples evaluated : {report.total_examples}",
        f"precision if ungated: {report.ungated_precision}",
        f"target precision   : {report.target_precision}",
    ]
    if report.recommended_threshold is None:
        lines += [
            "",
            "No threshold reached the target precision on this data.",
            "That means the pipeline cannot hit the target by gating alone —",
            "improve OCR, the catalog, or the parser, then re-calibrate.",
        ]
    else:
        lines += [
            f"recommended threshold: {report.recommended_threshold}",
            f"  precision at that threshold: {report.achieved_precision}",
            f"  coverage  at that threshold: {report.achieved_coverage} "
            f"({round((report.achieved_coverage or 0) * 100, 1)}% auto-accepted, "
            f"the rest go to review)",
            "",
            "Set it with: export CARDID_AUTO_ACCEPT_THRESHOLD="
            f"{report.recommended_threshold}",
        ]
    lines += ["", "threshold  precision  coverage  accepted"]
    for point in report.points[::10]:
        lines.append(
            f"{point.threshold:>9.2f}  {point.precision:>9.3f}  "
            f"{point.coverage:>8.3f}  {point.accepted:>8d}"
        )
    return "\n".join(lines)
