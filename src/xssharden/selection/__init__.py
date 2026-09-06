"""Validity-gated variant selection subpackage.

Selects a budgeted subset of already-scored variant records for
hardening, behind a validity hard gate (only ``valid is True``
malicious records are eligible), with split safety, exact-payload
deduplication, detector-impact ranking, and a seeded random-valid
control arm.
"""

from xssharden.selection.selection import (
    BLOCKED_SPLITS,
    DEFAULT_ALLOWED_SPLITS,
    SelectionResult,
    select_variants,
)

__all__ = [
    "BLOCKED_SPLITS",
    "DEFAULT_ALLOWED_SPLITS",
    "SelectionResult",
    "select_variants",
]
