"""Suggest a defect category for a free-text description.

Two methods:
  * keyword  -- transparent, offline, no dependencies (default)
  * zero-shot -- Hugging Face `facebook/bart-large-mnli`, no training data
                 needed; loaded lazily and only if `transformers` is installed

The category scheme is generic and illustrative. Replace it with your own
organization's defect codes when adapting this tool.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Category -> keywords. Keywords are matched as word prefixes.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "Leak": ("leak", "weep", "seep", "drip", "oil on", "wet", "stain"),
    # Units (mm, inch) are deliberately NOT keywords: a measurement can
    # describe any defect type, so units say nothing about the category.
    "Dimensional": (
        "dimension", "out of tolerance", "oot", "undersize", "oversize",
        "gap", "spacing", "misalign", "warp", "flatness", "diameter",
        "size", "length", "width", "height", "depth",
    ),
    "Surface / Coating": (
        "scratch", "paint", "coating", "blister", "peel", "rust", "corrod",
        "discolor", "chip", "run", "overspray", "dft",
    ),
    "Weld": ("weld", "porosity", "undercut", "spatter", "crack", "fusion"),
    "Electrical": (
        # "short circuit"/"shorted", not bare "short" ("qty 1 short").
        "electrical", "continuity", "open circuit", "short circuit",
        "shorted", "insulation", "megger",
        "resistance", "ohm", "ratio", "polarity", "kv", "wiring", "terminal",
    ),
    "Fastener / Torque": (
        "bolt", "nut", "torque", "loose", "stripped", "thread", "washer",
        "ft-lb", "in-lb",
    ),
    "Missing / Wrong Part": (
        "missing", "wrong part", "incorrect part", "not installed",
        "shortage", "extra", "substitut",
    ),
    "Handling / Shipping Damage": (
        "dent", "crate", "pallet", "forklift", "dropped", "packaging",
        "impact", "bent", "shipping", "transit",
    ),
    "Labeling / Documentation": (
        "label", "nameplate", "tag", "illegible", "mislabel", "drawing",
        "revision", "serial", "paperwork", "record",
    ),
    "Contamination": (
        "contaminat", "debris", "moisture", "dirt", "foreign", "particle",
        "fod", "water",
    ),
}

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"


def keyword_classify(text: str, top_k: int = 3) -> dict[str, float]:
    """Rank categories by keyword hits. Returns {category: confidence}.

    Confidence is each category's share of total hits, so the values sum
    to 1.0 when anything matched. Returns {} when nothing matched.
    """
    lower = (text or "").lower()
    hits = {
        cat: sum(1 for kw in kws if re.search(r"\b" + re.escape(kw), lower))
        for cat, kws in CATEGORIES.items()
    }
    total = sum(hits.values())
    if total == 0:
        return {}
    ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
    return {cat: n / total for cat, n in ranked[:top_k] if n > 0}


@lru_cache(maxsize=1)
def _load_zero_shot():
    """Load the zero-shot pipeline once. Returns None if unavailable."""
    try:
        from transformers import pipeline  # imported lazily on purpose
    except ImportError:
        return None
    try:
        return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    except Exception:  # model download or runtime failure
        return None


def zero_shot_classify(text: str, top_k: int = 3) -> dict[str, float] | None:
    """Classify with an NLI model. Returns None if the model is unavailable."""
    clf = _load_zero_shot()
    if clf is None or not (text or "").strip():
        return None
    result = clf(
        text,
        candidate_labels=list(CATEGORIES),
        hypothesis_template="This manufacturing defect is a {} issue.",
    )
    return dict(list(zip(result["labels"], result["scores"]))[:top_k])


def classify(text: str, use_model: bool = False) -> tuple[dict[str, float], str]:
    """Return (scores, method_used). Falls back to keywords if needed."""
    if use_model:
        scores = zero_shot_classify(text)
        if scores is not None:
            return scores, f"zero-shot ({ZERO_SHOT_MODEL})"
    return keyword_classify(text), "keyword"
