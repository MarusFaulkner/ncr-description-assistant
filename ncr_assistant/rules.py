"""Rule-based quality checks for nonconformance (NCR) defect descriptions.

Guiding principle: the person who finds a defect should record what they
OBSERVED -- what, where, and how much -- and leave root cause to the
investigation. These checks score a free-text description against that
principle and return plain-language feedback.

The rules are deliberately transparent (no ML) so an inspector can see
exactly why a description scored the way it did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words that say "something is wrong" without saying what.
VAGUE_TERMS = (
    "bad", "broken", "issue", "problem", "failed", "failure", "wrong",
    "defective", "not good", "no good", "messed up", "junk", "not right",
    "doesn't work", "does not work", "not working",
)

# Phrases that speculate about cause instead of stating an observation.
CAUSE_PATTERNS = (
    r"\bbecause\b", r"\bdue to\b", r"\bcaused by\b", r"\bprobably\b",
    r"\blikely\b", r"\bmaybe\b", r"\bseems?\b", r"\bi think\b",
    r"\bmust have\b", r"\bmight\b", r"\bguess\b", r"\bfault of\b",
)

# Observable conditions -- what the inspector actually saw or measured.
CONDITION_TERMS = (
    "leak", "weep", "seep", "drip", "crack", "dent", "scratch", "gouge",
    "missing", "loose", "corrod", "rust", "chip", "bent", "misalign",
    "out of tolerance", "oot", "exceed", "below", "above", "measured",
    "reads", "reading", "contaminat", "burr", "porosity", "discolor",
    "torn", "mislabel", "illegible", "wrong part", "incorrect", "short",
    "open circuit", "no continuity", "undersize", "oversize", "warp",
    "blister", "peel", "spatter", "undercut", "damaged packaging",
    "stripped", "debris", "moisture", "particle", "flatness",
    "does not match", "mismatch", "not installed", "worn",
)

# Physical locations / features typical of transformer and general
# manufacturing. Extend this list for your own product line.
LOCATION_TERMS = (
    "top", "bottom", "left", "right", "front", "rear", "side", "upper",
    "lower", "hv", "lv", "phase", "tank", "cover", "flange", "bushing",
    "radiator", "tap changer", "core", "coil", "gasket", "weld", "valve",
    "nameplate", "terminal", "conservator", "pallet", "crate", "lead",
    "bracket", "lifting lug", "cabinet", "door", "hole", "seam", "bolt",
    "surface", "edge", "corner", "panel", "pipe", "fitting", "wiring",
)

LOCATION_PATTERNS = (
    r"\bpos(ition)?\.?\s*#?\d+",  # position 3, pos. #3
    r"\bitem\s*#?\d+",            # item 12
    r"#\d+",                      # #4
    r"\b[a-z]\d\b",               # grid refs like h1, x2
)

# Numbers with units, or explicit counts.
MEASUREMENT_PATTERN = re.compile(
    r"(\d+(\.\d+)?\s?(mm|cm|m|in|inch|inches|\"|ft|psi|kpa|bar|kv|v|a|ma|"
    r"ohms?|mohm|°c|°f|deg|ft-?lbs?|in-?lbs?|nm|%|mils?|kg|lbs?)\b)"
    r"|(\bqty\.?\s*\d+)|(\bx\s?\d+\b)|(\b\d+\s+(of|pcs|pieces|places|plcs)\b)",
    re.IGNORECASE,
)

MIN_WORDS = 4
MAX_WORDS = 30


@dataclass
class DescriptionReport:
    """Result of checking one defect description."""

    text: str
    score: int
    grade: str
    has_condition: bool
    has_location: bool
    has_measurement: bool
    vague_terms: list[str] = field(default_factory=list)
    cause_speculation: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)


def _find_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    """Return the terms that appear in text as whole words/phrases."""
    found = []
    for term in terms:
        # Prefix match for stems like "corrod" -> corroded/corrosion.
        if re.search(r"\b" + re.escape(term), text):
            found.append(term)
    return found


def _grade(score: int) -> str:
    if score >= 75:
        return "Actionable"
    if score >= 50:
        return "Needs detail"
    return "Rewrite"


def check_description(text: str) -> DescriptionReport:
    """Score a defect description from 0-100 and explain the score.

    Scoring (additive, clamped to 0-100):
        +30  states an observable condition (leak, crack, missing, ...)
        +25  states where (feature, position, side, item number)
        +25  states how much (measurement with unit, or a count)
        +20  reasonable length (MIN_WORDS..MAX_WORDS words)
        -20  relies on vague words with no observable condition
        -25  speculates about cause
    """
    clean = (text or "").strip()
    lower = clean.lower()
    words = re.findall(r"\S+", clean)

    conditions = _find_terms(lower, CONDITION_TERMS)
    locations = _find_terms(lower, LOCATION_TERMS) + [
        m.group(0) for p in LOCATION_PATTERNS for m in re.finditer(p, lower)
    ]
    has_measurement = bool(MEASUREMENT_PATTERN.search(lower))
    vague = _find_terms(lower, VAGUE_TERMS)
    cause = [m.group(0) for p in CAUSE_PATTERNS for m in re.finditer(p, lower)]

    score = 0
    feedback: list[str] = []

    if conditions:
        score += 30
    else:
        feedback.append(
            "Name the observable condition (e.g., leak, crack, missing, "
            "loose, out of tolerance)."
        )

    if locations:
        score += 25
    else:
        feedback.append(
            "Say where: the feature, side, phase, or position number."
        )

    if has_measurement:
        score += 25
    else:
        feedback.append(
            "Add how much: a measured value with units, or a count "
            "(e.g., 2.5 mm, qty 3, 4 of 12 bolts)."
        )

    if MIN_WORDS <= len(words) <= MAX_WORDS:
        score += 20
    elif len(words) < MIN_WORDS:
        feedback.append(
            f"Too short to act on ({len(words)} words). Aim for "
            f"{MIN_WORDS}-{MAX_WORDS}."
        )
    else:
        feedback.append(
            f"Long ({len(words)} words). Keep the description to the "
            "observation; put history and context in the body of the report."
        )

    if vague and not conditions:
        score -= 20
        feedback.append(
            f"Vague wording ({', '.join(vague)}) does not tell the reader "
            "what was seen. Replace it with the actual condition."
        )

    if cause:
        score -= 25
        feedback.append(
            f"Speculates about cause ({', '.join(cause)}). Record only what "
            "you observed; root cause belongs to the investigation."
        )

    score = max(0, min(100, score))
    if not feedback:
        feedback.append("Clear, specific, observation-only. Good to submit.")

    return DescriptionReport(
        text=clean,
        score=score,
        grade=_grade(score),
        has_condition=bool(conditions),
        has_location=bool(locations),
        has_measurement=has_measurement,
        vague_terms=vague,
        cause_speculation=cause,
        feedback=feedback,
    )
