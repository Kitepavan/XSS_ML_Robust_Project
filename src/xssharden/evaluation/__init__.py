"""Independent four-arm hardening evaluation.

Scores each arm detector read-only on held-out clean-test records and on
a named adversarial split, reusing the shared attack scorer. Tuning
records are never touched and detectors are never trained or tuned here;
payloads stay inert strings throughout and every input is left unchanged.
"""

from xssharden.evaluation.evaluate import (
    ARM_NAMES,
    ADV_SPLIT_DEFAULT,
    CLEAN_SPLIT_DEFAULT,
    ArmEvaluation,
    CleanTestMetrics,
    HardeningEvaluation,
    evaluate_arms,
    evaluate_hardening_result,
)

__all__ = [
    "ARM_NAMES",
    "ADV_SPLIT_DEFAULT",
    "CLEAN_SPLIT_DEFAULT",
    "ArmEvaluation",
    "CleanTestMetrics",
    "HardeningEvaluation",
    "evaluate_arms",
    "evaluate_hardening_result",
]
