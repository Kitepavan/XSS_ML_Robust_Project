"""Four-arm detector hardening for validated XSS variants.

Builds the required comparison arms from one shared training corpus and
one pool of already-validated variant records:

- ``baseline`` — original training records only.
- ``naive`` — original training records plus every valid malicious
  variant (``valid is True`` and ``label == 1``), deduplicated.
- ``random_valid`` — original training records plus a budget-matched
  random-valid control from :func:`xssharden.selection.select_variants`.
- ``selective`` — original training records plus a budgeted
  validity-gated detector-impact selection from
  :func:`xssharden.selection.select_variants`.

Pipeline per call::

    train records
        ↓ fit fresh baseline detector via ``detector_factory``
    score validated variants read-only via
    :func:`xssharden.attack.evaluate_variants` (never fits the detector)
        ↓
    select budgeted subsets via
    :func:`xssharden.selection.select_variants`
    (``random_valid`` and ``impact`` strategies, same budget)
        ↓
    assemble per-arm training lists (augmented copies carry
    ``split="train"`` and ``hardening_arm="<arm>"``)
        ↓
    fit one fresh detector per arm via ``detector_factory`` and,
    only when ``validation_records`` is given, calibrate each arm via
    ``calibrate_threshold(validation_records)``

Payloads are inert strings throughout: this module never imports or
invokes the browser validator, never renders, executes, fetches, or
sends anything, and never executes payload text. Inputs are never
mutated. No CLI is provided here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

from xssharden.attack import evaluate_variants
from xssharden.selection import SelectionResult, select_variants

#: Required hardening comparison arms, in canonical order.
ARM_NAMES: tuple[str, ...] = ("baseline", "naive", "random_valid", "selective")

Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Typed result structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmResult:
    """One hardening arm: training records plus its fitted detector."""

    name: str = ""
    records: list[Record] = field(default_factory=list)
    detector: Any = None
    added_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return arm metadata as plain JSON-serializable dicts.

        The detector object itself is not serializable, so only its
        class name is reported; records are returned as plain copies.
        """
        return {
            "name": self.name,
            "records": [dict(record) for record in self.records],
            "detector": type(self.detector).__name__,
            "added_count": self.added_count,
            "train_size": len(self.records),
        }


@dataclass(frozen=True)
class HardeningResult:
    """Four fitted hardening arms plus scored variants and audits."""

    arm_records: dict[str, list[Record]] = field(default_factory=dict)
    arm_detectors: dict[str, Any] = field(default_factory=dict)
    added_counts: dict[str, int] = field(default_factory=dict)
    arm_sizes: dict[str, int] = field(default_factory=dict)
    scored_variants: list[Record] = field(default_factory=list)
    selection_results: dict[str, SelectionResult] = field(default_factory=dict)
    budget: int = 0
    seed: int = 42
    max_per_category: int | None = None
    train_size: int = 0

    # -- compatibility aliases -------------------------------------------
    @property
    def records(self) -> dict[str, list[Record]]:
        """Alias for :attr:`arm_records`."""
        return self.arm_records

    @property
    def detectors(self) -> dict[str, Any]:
        """Alias for :attr:`arm_detectors`."""
        return self.arm_detectors

    @property
    def counts(self) -> dict[str, int]:
        """Alias for :attr:`added_counts`."""
        return self.added_counts

    @property
    def selection(self) -> dict[str, SelectionResult]:
        """Alias for :attr:`selection_results`."""
        return self.selection_results

    @property
    def scored(self) -> list[Record]:
        """Alias for :attr:`scored_variants` (copies)."""
        return [dict(record) for record in self.scored_variants]

    @property
    def arms(self) -> dict[str, ArmResult]:
        """Per-arm :class:`ArmResult` views keyed by arm name."""
        views: dict[str, ArmResult] = {}
        for name in ARM_NAMES:
            views[name] = ArmResult(
                name=name,
                records=[dict(record) for record in self.arm_records.get(name, [])],
                detector=self.arm_detectors.get(name),
                added_count=int(self.added_counts.get(name, 0)),
            )
        return views

    def to_dict(self) -> dict[str, Any]:
        """Return counts plus scored variants and selection audits.

        Detector objects are reported by class name only; per-arm
        records and scored variants are included as plain copies.
        """
        return {
            "arm_records": {
                name: [dict(record) for record in self.arm_records.get(name, [])]
                for name in ARM_NAMES
            },
            "arm_detectors": {
                name: type(self.arm_detectors.get(name)).__name__
                for name in ARM_NAMES
            },
            "added_counts": dict(self.added_counts),
            "arm_sizes": dict(self.arm_sizes),
            "scored_variants": [dict(record) for record in self.scored_variants],
            "selection_results": {
                name: result.to_dict()
                for name, result in self.selection_results.items()
            },
            "budget": self.budget,
            "seed": self.seed,
            "max_per_category": self.max_per_category,
            "train_size": self.train_size,
        }


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _check_budget(budget: Any) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"budget must be a positive integer, got {budget!r}")
    if budget <= 0:
        raise ValueError(f"budget must be a positive integer, got {budget!r}")
    return budget


def _check_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    return seed


def _check_max_per_category(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "max_per_category must be a positive integer or None, "
            f"got {value!r}"
        )
    if value <= 0:
        raise ValueError(
            "max_per_category must be a positive integer or None, "
            f"got {value!r}"
        )
    return value


def _check_factory(factory: Any) -> Callable[[], Any]:
    if not callable(factory):
        raise TypeError(
            "detector_factory must be a callable returning a fresh "
            f"detector, got {type(factory).__name__}"
        )
    return factory


def _materialize(name: str, records: Any) -> list[Record]:
    if records is None:
        raise TypeError(f"{name} must be an iterable of dicts, got None")
    if isinstance(records, (dict, str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be an iterable of dicts (e.g. list), "
            f"got {type(records).__name__}"
        )
    try:
        items = list(records)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable of dicts, "
            f"got {type(records).__name__}"
        ) from exc
    if len(items) == 0:
        raise ValueError(f"{name} must be a non-empty sequence")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"{name} record at index {index} must be dict, "
                f"got {type(item).__name__}"
            )
    return items


def _check_train_shape(items: Sequence[Record]) -> None:
    for index, item in enumerate(items):
        payload = item.get("payload", None)
        if "payload" not in item:
            raise ValueError(
                f"train record at index {index}: missing required field 'payload'"
            )
        if not isinstance(payload, str):
            raise TypeError(
                f"train record at index {index}: 'payload' must be str, "
                f"got {type(payload).__name__}"
            )
        if not payload.strip():
            raise ValueError(
                f"train record at index {index}: "
                "'payload' must be a non-empty string"
            )
        if "label" not in item:
            raise ValueError(
                f"train record at index {index}: missing required field 'label'"
            )
        label = item["label"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(
                f"train record at index {index}: 'label' must be 0 or 1, "
                f"got {label!r}"
            )
        if label not in (0, 1):
            raise ValueError(
                f"train record at index {index}: 'label' must be 0 or 1, "
                f"got {label!r}"
            )


def _check_validation_shape(items: Sequence[Record]) -> None:
    """Require calibration records to be explicitly validation-split."""
    for index, item in enumerate(items):
        split = item.get("split")
        if split != "validation":
            raise ValueError(
                "validation_records must contain only split='validation' "
                f"records; record at index {index} has split={split!r}"
            )
def _require_fit(detector: Any, *, arm: str) -> None:
    if not callable(getattr(detector, "fit", None)):
        raise TypeError(
            f"detector_factory produced an object without a callable "
            f"fit() for arm {arm!r} (got {type(detector).__name__})"
        )


def _augmented_copy(scored: Record, *, arm: str) -> Record:
    """Copy a scored variant as an augmented training record.

    Every input key/provenance (including ``detector_score``) is
    preserved; only ``split`` is reassigned to ``"train"`` and the
    ``hardening_arm`` provenance marker is set.
    """
    merged = dict(scored)
    merged["split"] = "train"
    merged["hardening_arm"] = arm
    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_hardening_arms(
    train_records: Iterable[Record],
    validated_variants: Iterable[Record],
    detector_factory: Callable[[], Any],
    budget: int,
    validation_records: Iterable[Record] | None = None,
    seed: int = 42,
    max_per_category: int | None = None,
) -> HardeningResult:
    """Fit the four required hardening arms and return them.

    Parameters
    ----------
    train_records:
        Non-empty iterable of dicts, each with a non-empty ``payload``
        string and a binary ``label`` (0/1). Copied verbatim per arm;
        never mutated.
    validated_variants:
        Non-empty iterable of already-validated variant dicts (each
        with ``payload``/``label`` plus ``valid`` and provenance). Only
        records with ``valid is True`` and ``label == 1`` can ever be
        added to an augmented arm. Scored read-only with
        :func:`xssharden.attack.evaluate_variants` using the freshly
        fitted baseline detector; payloads stay inert.
    detector_factory:
        Zero-argument callable returning a fresh, unfitted detector
        exposing ``fit(records)`` (and optionally
        ``calibrate_threshold(validation_records)``). Called once per
        arm (four times total) so no fitted state is shared.
    budget:
        Positive integer augmentation budget shared by the
        ``random_valid`` and ``selective`` arms (forwarded to
        :func:`xssharden.selection.select_variants`). Shortfall returns
        whatever is eligible; nothing is fabricated.
    validation_records:
        Optional non-empty iterable of ``split == "validation"`` record
        dicts. When given, each arm detector exposing
        ``calibrate_threshold`` is calibrated on an independent copy of
        these records; detectors without that method are left
        uncalibrated. ``None`` (the default) skips calibration.
        Validation records are never used for fitting.
    seed:
        Deterministic seed forwarded to both
        :func:`xssharden.selection.select_variants` calls.
    max_per_category:
        Optional positive-integer per-category cap forwarded to both
        selection calls; ``None`` disables the cap. The ``naive`` arm
        always takes every eligible valid variant and is never capped.

    Returns
    -------
    HardeningResult:
        ``arm_records``/``arm_detectors``/``added_counts``/``arm_sizes``
        keyed by ``("baseline", "naive", "random_valid", "selective")``,
        plus ``scored_variants`` holding the baseline-scored variant
        records and ``selection_results`` holding the
        ``SelectionResult`` audits for ``random_valid`` and
        ``selective``.
    """
    budget = _check_budget(budget)
    seed = _check_seed(seed)
    cap = _check_max_per_category(max_per_category)
    factory = _check_factory(detector_factory)

    train_items = _materialize("train_records", train_records)
    _check_train_shape(train_items)
    variant_items = _materialize("validated_variants", validated_variants)

    validation_items: list[Record] | None = None
    if validation_records is not None:
        validation_items = _materialize("validation_records", validation_records)
        _check_validation_shape(validation_items)

    train_payloads = {item["payload"] for item in train_items}

    # -- Baseline arm: fit on original training records only ---------------
    baseline_detector = factory()
    _require_fit(baseline_detector, arm="baseline")
    baseline_detector.fit([dict(item) for item in train_items])

    # -- Score variants read-only with the fitted baseline detector --------
    # Pass copies so scoring can never mutate the caller's variant dicts.
    scoring_inputs = [dict(item) for item in variant_items]
    scored = evaluate_variants(scoring_inputs, baseline_detector).records

    # -- Naive arm: every valid malicious variant, deduplicated ------------
    naive_added: list[Record] = []
    seen_naive: set[str] = set()
    for row in scored:
        if row.get("valid") is not True or int(row["label"]) != 1:
            continue
        payload = row["payload"]
        if payload in train_payloads or payload in seen_naive:
            continue
        seen_naive.add(payload)
        naive_added.append(_augmented_copy(row, arm="naive"))

    # -- Budgeted arms via the shared selection engine ---------------------
    random_selection: SelectionResult = select_variants(
        scored,
        budget,
        strategy="random_valid",
        seed=seed,
        max_per_category=cap,
        training_records=train_items,
        allowed_splits=("adv_dev",),
    )
    selective_selection: SelectionResult = select_variants(
        scored,
        budget,
        strategy="impact",
        seed=seed,
        max_per_category=cap,
        training_records=train_items,
        allowed_splits=("adv_dev",),
    )
    random_added = [
        _augmented_copy(dict(row), arm="random_valid")
        for row in random_selection.records
    ]
    selective_added = [
        _augmented_copy(dict(row), arm="selective")
        for row in selective_selection.records
    ]

    # -- Assemble per-arm training lists (caller inputs never reused) ------
    arm_added: dict[str, list[Record]] = {
        "baseline": [],
        "naive": naive_added,
        "random_valid": random_added,
        "selective": selective_added,
    }
    arm_records: dict[str, list[Record]] = {}
    for name in ARM_NAMES:
        arm_records[name] = [dict(item) for item in train_items] + [
            dict(item) for item in arm_added[name]
        ]

    # -- Fit one fresh detector per remaining arm --------------------------
    arm_detectors: dict[str, Any] = {"baseline": baseline_detector}
    for name in ("naive", "random_valid", "selective"):
        detector = factory()
        _require_fit(detector, arm=name)
        detector.fit([dict(item) for item in arm_records[name]])
        arm_detectors[name] = detector

    # -- Optional threshold calibration on validation records only ---------
    if validation_items is not None:
        for name in ARM_NAMES:
            detector = arm_detectors[name]
            calibrate = getattr(detector, "calibrate_threshold", None)
            if callable(calibrate):
                calibrate([dict(item) for item in validation_items])

    added_counts = {name: len(arm_added[name]) for name in ARM_NAMES}
    arm_sizes = {name: len(arm_records[name]) for name in ARM_NAMES}

    return HardeningResult(
        arm_records=arm_records,
        arm_detectors=arm_detectors,
        added_counts=added_counts,
        arm_sizes=arm_sizes,
        scored_variants=[dict(row) for row in scored],
        selection_results={
            "random_valid": random_selection,
            "selective": selective_selection,
        },
        budget=budget,
        seed=seed,
        max_per_category=cap,
        train_size=len(train_items),
    )
