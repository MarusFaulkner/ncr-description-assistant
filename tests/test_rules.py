"""Unit tests for the rule-based checker and keyword classifier.

Run:  python -m pytest -q
"""

import csv
from pathlib import Path

import pytest

from ncr_assistant import CATEGORIES, check_description, keyword_classify

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --- check_description -------------------------------------------------------

def test_vague_description_needs_rewrite():
    report = check_description("bushing bad")
    assert report.grade == "Rewrite"
    assert "bad" in report.vague_terms


def test_specific_description_is_actionable():
    report = check_description(
        "Oil weep at HV bushing #2 flange gasket, approx 1 drop per 10 min"
    )
    assert report.grade == "Actionable"
    assert report.has_condition and report.has_location


def test_cause_speculation_is_flagged_and_penalized():
    observed = check_description("Porosity in tank seam weld, 3 pores")
    guessed = check_description(
        "Porosity in tank seam weld, 3 pores, probably due to moisture"
    )
    assert guessed.cause_speculation
    assert guessed.score < observed.score


def test_measurement_with_units_detected():
    assert check_description("Hole diameter measured 41.2 mm").has_measurement
    assert check_description("4 of 12 bolts loose").has_measurement
    assert not check_description("Hole too big").has_measurement


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_input_does_not_crash(text):
    report = check_description(text)
    assert report.score == 0
    assert report.grade == "Rewrite"


def test_score_is_clamped():
    report = check_description("bad broken issue probably maybe because")
    assert 0 <= report.score <= 100


# --- keyword_classify --------------------------------------------------------

def test_leak_classified():
    assert next(iter(keyword_classify("Oil weep at drain valve"))) == "Leak"


def test_units_do_not_force_dimensional():
    # Regression: "mm" used to vote Dimensional for any measured defect.
    top = next(iter(keyword_classify("Scratch on cover, 120 mm long")))
    assert top == "Surface / Coating"


def test_qty_short_is_not_electrical():
    # Regression: bare "short" used to match "qty 1 short".
    scores = keyword_classify("Valve not installed, qty 1 short")
    assert "Electrical" not in scores


def test_no_match_returns_empty():
    assert keyword_classify("lorem ipsum") == {}


def test_confidences_sum_to_one_or_less():
    scores = keyword_classify("Rust and paint blister on weld seam")
    assert 0.99 <= sum(scores.values()) <= 1.0 or len(scores) == 3


# --- dataset integrity -------------------------------------------------------

@pytest.mark.parametrize("name", ["synthetic_ncr_descriptions.csv",
                                  "holdout_ncr_descriptions.csv"])
def test_dataset_labels_are_valid(name):
    with (DATA_DIR / name).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for row in rows:
        assert row["category"] in CATEGORIES, row
        assert row["quality"] in {"good", "poor"}, row
        assert row["description"].strip(), row
