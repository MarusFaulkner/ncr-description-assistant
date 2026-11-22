"""NCR Description Assistant: check defect descriptions and suggest categories."""

from .classifier import CATEGORIES, classify, keyword_classify
from .rules import DescriptionReport, check_description

__all__ = [
    "CATEGORIES",
    "DescriptionReport",
    "check_description",
    "classify",
    "keyword_classify",
]
