"""Measure the assistant against the labeled synthetic dataset.

Reports:
  * category accuracy of the keyword classifier (top-1)
  * agreement between the rule-based grade and the good/poor label
  * every miss, so you can see where the rules or keywords need work

Usage:
    python evaluate.py                # keyword classifier
    python evaluate.py --model        # zero-shot model (needs transformers)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ncr_assistant import check_description, classify

DATA = Path(__file__).parent / "data" / "synthetic_ncr_descriptions.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="store_true", help="use zero-shot model")
    parser.add_argument(
        "--data", type=Path, default=DATA,
        help="CSV to score (default: tuning set; use data/holdout_*.csv for "
             "an honest number)",
    )
    args = parser.parse_args()

    with args.data.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cat_hits, grade_hits, misses = 0, 0, []
    method = "keyword"
    for row in rows:
        text = row["description"]
        scores, method = classify(text, use_model=args.model)
        predicted = next(iter(scores), "(none)")
        if predicted == row["category"]:
            cat_hits += 1
        else:
            misses.append(f"CATEGORY  {text!r}: expected {row['category']}, got {predicted}")

        report = check_description(text)
        # "Actionable" should line up with good; anything else with poor.
        predicted_quality = "good" if report.grade == "Actionable" else "poor"
        if predicted_quality == row["quality"]:
            grade_hits += 1
        else:
            misses.append(
                f"QUALITY   {text!r}: labeled {row['quality']}, "
                f"scored {report.score} ({report.grade})"
            )

    n = len(rows)
    print(f"Rows: {n}   Classifier: {method}")
    print(f"Category accuracy (top-1): {cat_hits}/{n} = {cat_hits / n:.0%}")
    print(f"Quality agreement:         {grade_hits}/{n} = {grade_hits / n:.0%}")
    if misses:
        print("\nMisses:")
        print("\n".join(f"  {m}" for m in misses))


if __name__ == "__main__":
    main()
