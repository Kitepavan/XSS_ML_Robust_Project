"""Four-arm detector-hardening subpackage.

Fits the required baseline/naive/random-valid/selective comparison arms
from one shared training corpus and one pool of validated variants. The
random-valid arm is the mandatory budget-matched control for the
selective arm; validity (``valid is True``) is a hard gate and payloads
stay inert throughout.
"""

from xssharden.hardening.arms import (
    ARM_NAMES,
    ArmResult,
    HardeningResult,
    run_hardening_arms,
)

__all__ = [
    "ARM_NAMES",
    "ArmResult",
    "HardeningResult",
    "run_hardening_arms",
]
