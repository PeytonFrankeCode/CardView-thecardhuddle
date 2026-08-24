"""Threshold calibration — how a precision target becomes a setting."""

from __future__ import annotations

from cardid.calibrate import (
    LabeledExample,
    Observation,
    format_report,
    observe,
    read_labels,
    sweep,
)


def observations(pairs):
    """pairs: (confidence, correct)"""
    return [
        Observation(confidence=confidence, predicted_id="a" if correct else "b",
                    true_card_id="a", correct=correct)
        for confidence, correct in pairs
    ]


def test_recommends_the_lowest_threshold_meeting_the_target():
    """Lowest qualifying threshold keeps the most volume automated."""
    data = observations([(0.95, True)] * 10 + [(0.5, False)] * 10)
    report = sweep(data, target_precision=0.99, min_accepted=5)
    assert report.recommended_threshold is not None
    assert 0.5 < report.recommended_threshold <= 0.95
    assert report.achieved_precision == 1.0
    assert report.achieved_coverage == 0.5


def test_reports_no_threshold_when_the_target_is_unreachable():
    # Wrong answers arrive with the same confidence as right ones, so no
    # threshold can separate them.
    data = observations([(0.99, True), (0.99, False)] * 10)
    report = sweep(data, target_precision=0.99, min_accepted=5)
    assert report.recommended_threshold is None
    assert "cannot hit the target" in format_report(report)


def test_min_accepted_rejects_a_threshold_that_accepts_almost_nothing():
    data = observations([(1.0, True)] + [(0.2, False)] * 50)
    lenient = sweep(data, target_precision=0.99, min_accepted=1)
    strict = sweep(data, target_precision=0.99, min_accepted=10)
    assert lenient.recommended_threshold is not None
    assert strict.recommended_threshold is None


def test_coverage_falls_as_the_threshold_rises():
    data = observations([(c / 10, True) for c in range(11)])
    coverages = [point.coverage for point in sweep(data).points]
    assert coverages == sorted(coverages, reverse=True)


def test_ungated_precision_is_reported():
    report = sweep(observations([(0.9, True), (0.9, True), (0.9, False), (0.9, True)]))
    assert report.ungated_precision == 0.75


def test_empty_label_set_is_handled():
    report = sweep([])
    assert report.total_examples == 0
    assert report.recommended_threshold is None


def test_observe_scores_predictions_against_labels(identifier):
    results = observe(identifier, [
        LabeledExample(true_card_id="mahomes_silver",
                       title="2017 Panini Prizm Patrick Mahomes II #269 Silver Prizm"),
        LabeledExample(true_card_id="burrow_prizm",
                       title="2020 Panini Prizm Joe Burrow #307 RC"),
        LabeledExample(true_card_id="mahomes_gold",
                       title="2017 Prizm Mahomes #269 RC"),
    ])
    assert [o.correct for o in results] == [True, True, False]


def test_read_labels_from_csv(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("true_card_id,title\nc1,a title\n", encoding="utf-8")
    labels = list(read_labels(path))
    assert labels[0].true_card_id == "c1"
    assert labels[0].title == "a title"


def test_read_labels_from_jsonl(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"true_card_id": "c1", "title": "a title"}\n', encoding="utf-8")
    assert list(read_labels(path))[0].true_card_id == "c1"


def test_report_serialises_to_json():
    payload = sweep(observations([(0.9, True)] * 5)).to_dict()
    assert "curve" in payload and "recommended_threshold" in payload
