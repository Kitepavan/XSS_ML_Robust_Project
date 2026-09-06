"""Detector-attack (evasion) measurement subpackage.

Scores already-validated variant records with an already-fitted detector
(``predict_proba``/``predict`` only — never fit or calibrate) and reports
aggregate evasion metrics with explicit valid-only versus raw denominators.
"""

from xssharden.attack.evasion import (
    BLOCKED_SPLITS,
    EvasionEvaluation,
    EvasionMetrics,
    compute_metrics,
    evaluate_variants,
)

__all__ = [
    "BLOCKED_SPLITS",
    "EvasionEvaluation",
    "EvasionMetrics",
    "compute_metrics",
    "evaluate_variants",
]
