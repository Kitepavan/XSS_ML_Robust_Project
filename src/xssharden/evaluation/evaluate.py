"""Independent evaluation of the four hardening arms.

For every arm (``baseline``/``naive``/``random_valid``/``selective``) this
module reports clean-test classification quality (confusion counts plus
accuracy/precision/recall/F1/FPR with safe zero denominators) and
adversarial robustness (raw evasion rate and valid-malicious evasion rate
with explicit denominators).

All scoring goes through :func:`xssharden.attack.evaluate_variants`,
which uses detectors read-only: nothing here trains detectors, tunes
thresholds, or touches tuning records. Clean-test inputs must carry
``split="clean-test"`` and adversarial inputs ``split="adv_test"`` by
default; records from the tuning split are always refused. Payloads are
treated as inert strings; inputs are never mutated. No report or console
output is produced here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from xssharden.attack import EvasionMetrics, evaluate_variants

#: Required hardening comparison arms, in canonical order.
ARM_NAMES: tuple[str, ...] = ("baseline", "naive", "random_valid", "selective")

#: Default held-out split names for the two evaluation inputs.
CLEAN_SPLIT_DEFAULT: str = "clean-test"
ADV_SPLIT_DEFAULT: str = "adv_test"

#: Split names that must never be scored as evaluation inputs.
_NEVER_SCORE_SPLITS: tuple[str, ...] = ("validation",)

Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Typed result structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanTestMetrics:
    """Clean-test confusion counts and derived rates."""

    total: int
    positives: int
    negatives: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    detector_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a plain JSON-serializable dict."""
        return asdict(self)


@dataclass(frozen=True)
class ArmEvaluation:
    """Per-arm clean-test and adversarial outcomes."""

    arm: str
    clean_metrics: CleanTestMetrics
    clean_records: list[Record] = field(default_factory=list)
    adversarial_metrics: EvasionMetrics | None = None
    adversarial_records: list[Record] = field(default_factory=list)
    clean_count: int = 0
    adversarial_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return scored records plus metrics as plain dicts."""
        assert self.adversarial_metrics is not None
        return {
            "arm": self.arm,
            "clean_metrics": self.clean_metrics.to_dict(),
            "clean_records": [dict(record) for record in self.clean_records],
            "adversarial_metrics": self.adversarial_metrics.to_dict(),
            "adversarial_records": [
                dict(record) for record in self.adversarial_records
            ],
            "clean_count": self.clean_count,
            "adversarial_count": self.adversarial_count,
        }


@dataclass(frozen=True)
class HardeningEvaluation:
    """Independent evaluation of all four hardening arms."""

    arms: dict[str, ArmEvaluation] = field(default_factory=dict)
    clean_test_count: int = 0
    adversarial_count: int = 0
    clean_split: str = CLEAN_SPLIT_DEFAULT
    adv_split: str = ADV_SPLIT_DEFAULT

    def to_dict(self) -> dict[str, Any]:
        """Return per-arm results and counts as plain dicts."""
        return {
            "arms": {
                name: self.arms[name].to_dict() for name in ARM_NAMES
            },
            "clean_test_count": self.clean_test_count,
            "adversarial_count": self.adversarial_count,
            "clean_split": self.clean_split,
            "adv_split": self.adv_split,
        }


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _check_split_param(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string, got {value!r}")
    if value in _NEVER_SCORE_SPLITS:
        raise ValueError(
            f"{name}={value!r} is a tuning split and must never be "
            "used as an evaluation input"
        )
    return value


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


def _check_record_shape(
    items: list[Record], *, name: str, expected_split: str
) -> None:
    for index, item in enumerate(items):
        payload = item.get("payload", None)
        if "payload" not in item:
            raise ValueError(
                f"{name} record at index {index}: "
                "missing required field 'payload'"
            )
        if not isinstance(payload, str):
            raise TypeError(
                f"{name} record at index {index}: 'payload' must be str, "
                f"got {type(payload).__name__}"
            )
        if not payload.strip():
            raise ValueError(
                f"{name} record at index {index}: "
                "'payload' must be a non-empty string"
            )
        if "label" not in item:
            raise ValueError(
                f"{name} record at index {index}: "
                "missing required field 'label'"
            )
        label = item["label"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(
                f"{name} record at index {index}: 'label' must be 0 or 1, "
                f"got {label!r}"
            )
        if label not in (0, 1):
            raise ValueError(
                f"{name} record at index {index}: 'label' must be 0 or 1, "
                f"got {label!r}"
            )
        split = item.get("split", None)
        if split in _NEVER_SCORE_SPLITS:
            raise ValueError(
                f"{name} record at index {index}: split {split!r} is a "
                "tuning split and must never be used as an evaluation input"
            )
        if split != expected_split:
            raise ValueError(
                f"{name} record at index {index}: split must be "
                f"{expected_split!r}, got {split!r}"
            )


def _check_arm_detectors(value: Any) -> dict[str, Any]:
    if value is None or not isinstance(value, dict):
        raise TypeError(
            "arm_detectors must be a dict keyed by arm name, "
            f"got {type(value).__name__}"
        )
    missing = [name for name in ARM_NAMES if name not in value]
    if missing:
        raise ValueError(f"arm_detectors is missing arms: {missing}")
    unexpected = [name for name in value if name not in ARM_NAMES]
    if unexpected:
        raise ValueError(f"arm_detectors has unexpected arms: {unexpected}")
    return dict(value)


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _clean_metrics(
    scored: list[Record], *, detector_threshold: float
) -> CleanTestMetrics:
    rows = list(scored)
    total = len(rows)
    positives = sum(1 for row in rows if int(row["label"]) == 1)
    negatives = total - positives
    tp = sum(
        1
        for row in rows
        if int(row["label"]) == 1 and int(row["predicted_label"]) == 1
    )
    tn = sum(
        1
        for row in rows
        if int(row["label"]) == 0 and int(row["predicted_label"]) == 0
    )
    fp = sum(
        1
        for row in rows
        if int(row["label"]) == 0 and int(row["predicted_label"]) == 1
    )
    fn = sum(
        1
        for row in rows
        if int(row["label"]) == 1 and int(row["predicted_label"]) == 0
    )
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return CleanTestMetrics(
        total=total,
        positives=positives,
        negatives=negatives,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=_safe_div(tp + tn, total),
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=_safe_div(fp, fp + tn),
        detector_threshold=float(detector_threshold),
    )


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def evaluate_arms(
    arm_detectors: dict[str, Any],
    clean_test_records: Iterable[Record],
    adversarial_records: Iterable[Record],
    *,
    clean_split: str = CLEAN_SPLIT_DEFAULT,
    adv_split: str = ADV_SPLIT_DEFAULT,
) -> HardeningEvaluation:
    """Score the four hardening-arm detectors on held-out inputs.

    Parameters
    ----------
    arm_detectors:
        Dict with exactly the keys ``baseline``/``naive``/``random_valid``/
        ``selective``, each an already-trained detector. Detectors are
        scored read-only through the shared attack scorer; they are
        never trained or tuned here.
    clean_test_records:
        Non-empty iterable of dicts, each with a non-empty ``payload``
        string, a binary ``label`` (0/1), and ``split`` equal to
        ``clean_split`` (``"clean-test"`` by default). Every other key
        is preserved verbatim in the scored outputs.
    adversarial_records:
        Non-empty iterable of dicts with the same shape, each with
        ``split`` equal to ``adv_split`` (``"adv_test"`` by default).
    clean_split:
        Required ``split`` value for clean-test inputs. Tuning splits
        are refused.
    adv_split:
        Required ``split`` value for adversarial inputs. Tuning splits
        are refused and it must differ from ``clean_split``.

    Returns
    -------
    HardeningEvaluation:
        Per-arm clean-test confusion/accuracy/precision/recall/F1/FPR
        with explicit denominators plus adversarial evasion metrics
        (raw evasion rate and valid-malicious evasion rate) with
        explicit denominators, scored records preserving every input
        key, and input record counts.
    """
    clean_name = _check_split_param("clean_split", clean_split)
    adv_name = _check_split_param("adv_split", adv_split)
    if clean_name == adv_name:
        raise ValueError(
            f"clean_split and adv_split must differ, got {clean_name!r}"
        )
    detectors = _check_arm_detectors(arm_detectors)

    clean_items = _materialize("clean_test_records", clean_test_records)
    _check_record_shape(
        clean_items, name="clean_test_records", expected_split=clean_name
    )
    adv_items = _materialize("adversarial_records", adversarial_records)
    _check_record_shape(
        adv_items, name="adversarial_records", expected_split=adv_name
    )

    arms: dict[str, ArmEvaluation] = {}
    for name in ARM_NAMES:
        detector = detectors[name]
        clean_scored = evaluate_variants(
            [dict(item) for item in clean_items],
            detector,
            allowed_splits=(clean_name,),
        )
        adv_scored = evaluate_variants(
            [dict(item) for item in adv_items],
            detector,
            allowed_splits=(adv_name,),
        )
        arms[name] = ArmEvaluation(
            arm=name,
            clean_metrics=_clean_metrics(
                clean_scored.records,
                detector_threshold=clean_scored.metrics.detector_threshold,
            ),
            clean_records=[dict(row) for row in clean_scored.records],
            adversarial_metrics=adv_scored.metrics,
            adversarial_records=[dict(row) for row in adv_scored.records],
            clean_count=len(clean_items),
            adversarial_count=len(adv_items),
        )

    return HardeningEvaluation(
        arms=arms,
        clean_test_count=len(clean_items),
        adversarial_count=len(adv_items),
        clean_split=clean_name,
        adv_split=adv_name,
    )


def evaluate_hardening_result(
    hardening_result: Any,
    clean_test_records: Iterable[Record],
    adversarial_records: Iterable[Record],
    *,
    clean_split: str = CLEAN_SPLIT_DEFAULT,
    adv_split: str = ADV_SPLIT_DEFAULT,
) -> HardeningEvaluation:
    """Score a hardening run on held-out clean and adversarial inputs.

    Parameters
    ----------
    hardening_result:
        A hardening run exposing per-arm detectors via
        ``arm_detectors`` (or the ``detectors`` alias), or a plain dict
        holding one of those mappings. The run itself is left unchanged.
    clean_test_records:
        Held-out clean-test records (``split`` equal to ``clean_split``
        by default).
    adversarial_records:
        Held-out adversarial records (``split`` equal to ``adv_split``
        by default).
    clean_split:
        Required ``split`` value for clean-test inputs.
    adv_split:
        Required ``split`` value for adversarial inputs.

    Returns
    -------
    HardeningEvaluation:
        The same per-arm comparison produced by
        :func:`evaluate_arms`.
    """
    detectors: Any = None
    if isinstance(hardening_result, dict):
        detectors = hardening_result.get(
            "arm_detectors", hardening_result.get("detectors", None)
        )
    else:
        detectors = getattr(hardening_result, "arm_detectors", None)
        if detectors is None:
            detectors = getattr(hardening_result, "detectors", None)
    if detectors is None:
        raise TypeError(
            "hardening_result must expose arm_detectors (or detectors), "
            f"got {type(hardening_result).__name__}"
        )
    return evaluate_arms(
        detectors,
        clean_test_records,
        adversarial_records,
        clean_split=clean_split,
        adv_split=adv_split,
    )
