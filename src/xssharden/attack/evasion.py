"""Detector-attack (evasion) measurement for validated variant records.

Scores already-validated XSS variant records with an already-fitted
detector and reports how many real (behaviorally valid) attacks evade it.

The detector is used read-only through ``predict_proba()`` and
``predict()``: this module never fits, calibrates, or otherwise mutates
it. Payloads are treated as inert strings throughout — nothing is
rendered, executed, fetched, or sent anywhere, and this module never
imports or invokes the execution validator.

Leakage safety: records carrying a held-out ``split`` (``validation``,
``clean-test``, or ``test``) are rejected unless the caller explicitly
passes ``allowed_splits`` containing that split name (for example a
named adversarial split such as ``"adv_test"``).

Metric definitions (all denominators explicit):

- ``evasion_rate`` = ``evasion_count / malicious_count`` (all candidates,
  malicious denominator).
- ``valid_malicious_evasion_rate`` (V-ASR) = ``valid_evasion_count /
  valid_malicious_count`` — only records with ``valid is True`` count.
  Invalid execution results are never silently treated as valid attacks.
- ``raw_evasion_rate`` = ``evasion_count / total_records`` (all
  candidates, total denominator) — the realizability-gap counterpart to
  V-ASR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

#: Splits that must never be scored as attack-evaluation inputs unless the
#: caller explicitly allow-lists them via ``allowed_splits``.
BLOCKED_SPLITS: tuple[str, ...] = ("validation", "clean-test", "test")

Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Typed result structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvasionMetrics:
    """Aggregate evasion metrics with explicit denominators."""

    total_records: int
    valid_records: int
    malicious_count: int
    evasion_count: int
    evasion_rate: float
    valid_malicious_count: int
    valid_evasion_count: int
    valid_malicious_evasion_rate: float
    raw_evasion_rate: float
    detector_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a plain JSON-serializable dict."""
        return asdict(self)


@dataclass(frozen=True)
class EvasionEvaluation:
    """Per-record evasion scores plus aggregate metrics."""

    records: list[Record] = field(default_factory=list)
    metrics: EvasionMetrics | None = None  # always set by evaluate_variants

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"records": [...], "metrics": {...}}`` as plain dicts."""
        assert self.metrics is not None
        return {
            "records": [dict(record) for record in self.records],
            "metrics": self.metrics.to_dict(),
        }


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _materialize_records(records: Any) -> list[Record]:
    if records is None:
        raise TypeError(
            "records must be an iterable of dicts, got None"
        )
    if isinstance(records, (dict, str, bytes, bytearray)):
        raise TypeError(
            "records must be an iterable of dicts "
            f"(e.g. list), got {type(records).__name__}"
        )
    try:
        items = list(records)
    except TypeError as exc:
        raise TypeError(
            "records must be an iterable of dicts, "
            f"got {type(records).__name__}"
        ) from exc
    if len(items) == 0:
        raise ValueError("records must be a non-empty sequence")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"record at index {index} must be dict, "
                f"got {type(item).__name__}"
            )
        if "payload" not in item:
            raise ValueError(
                f"record at index {index}: missing required field 'payload'"
            )
        payload = item["payload"]
        if not isinstance(payload, str):
            raise TypeError(
                f"record at index {index}: 'payload' must be str, "
                f"got {type(payload).__name__}"
            )
        if not payload.strip():
            raise ValueError(
                f"record at index {index}: 'payload' must be a non-empty string"
            )
        if "label" not in item:
            raise ValueError(
                f"record at index {index}: missing required field 'label'"
            )
        label = item["label"]
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError(
                f"record at index {index}: 'label' must be 0 or 1, "
                f"got {label!r}"
            )
        if int(label) not in (0, 1):
            raise ValueError(
                f"record at index {index}: 'label' must be 0 or 1, "
                f"got {label!r}"
            )
    return items


def _check_allowed_splits(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            "allowed_splits must be a sequence of str or None, "
            f"got {type(value).__name__}"
        )
    items = list(value)
    for entry in items:
        if not isinstance(entry, str):
            raise TypeError(
                "allowed_splits entries must be str, "
                f"got {type(entry).__name__}"
            )
    return tuple(items)


def _check_split(record: Record, index: int, allowed: tuple[str, ...] | None) -> None:
    split = record.get("split")
    if split is None:
        return
    if not isinstance(split, str):
        raise TypeError(
            f"record at index {index}: 'split' must be str or absent, "
            f"got {type(split).__name__}"
        )
    if allowed is not None:
        if split not in allowed:
            raise ValueError(
                f"record at index {index}: split {split!r} is not in "
                f"allowed_splits={list(allowed)}; refusing to score "
                "held-out records as attack-evaluation inputs"
            )
        return
    if split in BLOCKED_SPLITS:
        raise ValueError(
            f"record at index {index}: split {split!r} is held out "
            f"{list(BLOCKED_SPLITS)} and must never be scored as an "
            "attack-evaluation input; pass "
            f"allowed_splits=[{split!r}] to explicitly allow a named "
            "adversarial split"
        )


def _check_threshold(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"threshold must be a float in [0, 1], got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"threshold must be a float in [0, 1], got {value!r}"
        ) from exc
    if not np.isfinite(result) or not (0.0 <= result <= 1.0):
        raise ValueError(
            f"threshold must be a finite float in [0, 1], got {value!r}"
        )
    return result


def _require_detector(detector: Any) -> None:
    proba_fn = getattr(detector, "predict_proba", None)
    if not callable(proba_fn):
        raise TypeError(
            "detector must expose a callable predict_proba(), "
            f"got {type(detector).__name__}"
        )
    predict_fn = getattr(detector, "predict", None)
    if not callable(predict_fn):
        raise TypeError(
            "detector must expose a callable predict(), "
            f"got {type(detector).__name__}"
        )


# ---------------------------------------------------------------------------
# Detector-output validation helpers
# ---------------------------------------------------------------------------


def _scores_from_proba(proba: Any, count: int) -> list[float]:
    try:
        matrix = np.asarray(proba, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "detector predict_proba() must return a numeric (n, 2) array"
        ) from exc
    if matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] != 2:
        raise ValueError(
            "detector predict_proba() shape mismatch: expected "
            f"({count}, 2), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "detector predict_proba() must return finite probabilities"
        )
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError(
            "detector predict_proba() must return probabilities in [0, 1]"
        )
    return [float(value) for value in matrix[:, 1].tolist()]


def _labels_from_predictions(preds: Any, count: int) -> list[int]:
    try:
        array = np.asarray(preds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "detector predict() must return a binary (n,) array"
        ) from exc
    if array.ndim != 1 or array.shape[0] != count:
        raise ValueError(
            "detector predict() shape mismatch: expected "
            f"({count},), got {array.shape}"
        )
    labels: list[int] = []
    for value in array.tolist():
        if isinstance(value, bool):
            labels.append(int(value))
        elif isinstance(value, int):
            if value not in (0, 1):
                raise ValueError(
                    f"detector predict() must return only 0/1, got {value!r}"
                )
            labels.append(value)
        elif isinstance(value, float):
            if value not in (0.0, 1.0):
                raise ValueError(
                    f"detector predict() must return only 0/1, got {value!r}"
                )
            labels.append(int(value))
        else:
            raise ValueError(
                f"detector predict() must return only 0/1, got {value!r}"
            )
    return labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    scored: Sequence[Record], *, detector_threshold: float
) -> EvasionMetrics:
    """Aggregate per-record evasion flags into :class:`EvasionMetrics`.

    Only records with ``valid is True`` contribute to the valid-only
    counts; anything else (``False``, missing, or any other value) is
    treated as not-valid so invalid execution results can never inflate
    the valid attack success rate.
    """
    rows = list(scored)
    total = len(rows)
    valid = sum(1 for row in rows if row.get("valid") is True)
    malicious = sum(1 for row in rows if int(row["label"]) == 1)
    evasions = sum(
        1
        for row in rows
        if int(row["label"]) == 1 and int(row["predicted_label"]) == 0
    )
    valid_malicious = sum(
        1
        for row in rows
        if row.get("valid") is True and int(row["label"]) == 1
    )
    valid_evasions = sum(
        1
        for row in rows
        if row.get("valid") is True
        and int(row["label"]) == 1
        and int(row["predicted_label"]) == 0
    )
    return EvasionMetrics(
        total_records=total,
        valid_records=valid,
        malicious_count=malicious,
        evasion_count=evasions,
        evasion_rate=(evasions / malicious) if malicious else 0.0,
        valid_malicious_count=valid_malicious,
        valid_evasion_count=valid_evasions,
        valid_malicious_evasion_rate=(
            (valid_evasions / valid_malicious) if valid_malicious else 0.0
        ),
        raw_evasion_rate=(evasions / total) if total else 0.0,
        detector_threshold=float(detector_threshold),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def evaluate_variants(
    records: Iterable[Record],
    detector: Any,
    *,
    threshold: float | None = None,
    allowed_splits: Sequence[str] | None = None,
) -> EvasionEvaluation:
    """Score validated variant records with an already-fitted detector.

    Parameters
    ----------
    records:
        Non-empty iterable of dicts, each with a non-empty ``payload``
        string and a binary ``label`` (0/1). Every other input key
        (``variant_id``, ``seed_id``, ``valid``, ``split``, provenance,
        ...) is preserved verbatim. Payloads are passed to the detector
        as inert strings and are never executed.
    detector:
        An already-fitted detector exposing ``predict_proba()`` and
        ``predict()``. It is used read-only: this function never calls
        ``fit``/``calibrate_threshold`` and never assigns attributes.
    threshold:
        Optional explicit decision threshold in [0, 1]. When given,
        predicted labels are derived from the malicious-class scores
        (``score >= threshold``) without mutating the detector. When
        omitted, the detector's own ``predict()`` outputs are used and
        the reported ``detector_threshold`` is the detector's
        ``threshold_`` attribute (defaulting to 0.5 when absent).
    allowed_splits:
        Optional explicit allow-list of ``split`` values. By default,
        records with ``split`` in ``BLOCKED_SPLITS``
        (``validation``/``clean-test``/``test``) raise ``ValueError``;
        pass the held-out name explicitly (for example
        ``allowed_splits=["adv_test"]``) to score a named adversarial
        split.

    Returns
    -------
    EvasionEvaluation:
        ``.records`` holds one scored dict per input (input order), each
        adding ``detector_score``/``probability`` (the malicious-class
        probability), ``predicted_label``, ``detector_threshold``, and
        ``evaded`` (``True`` only when ``label == 1`` and the detector
        predicts ``0``). ``.metrics`` holds the aggregate
        :class:`EvasionMetrics`.
    """
    _require_detector(detector)
    items = _materialize_records(records)
    allowed = _check_allowed_splits(allowed_splits)
    for index, item in enumerate(items):
        _check_split(item, index, allowed)

    if threshold is not None:
        effective_threshold = _check_threshold(threshold)
    else:
        raw_default = getattr(detector, "threshold_", 0.5)
        effective_threshold = _check_threshold(raw_default)

    proba = detector.predict_proba(items)
    scores = _scores_from_proba(proba, len(items))
    if threshold is not None:
        predicted = [int(score >= effective_threshold) for score in scores]
    else:
        predicted = _labels_from_predictions(detector.predict(items), len(items))

    scored: list[Record] = []
    for item, score, pred in zip(items, scores, predicted):
        merged = dict(item)
        merged["detector_score"] = float(score)
        merged["probability"] = float(score)
        merged["predicted_label"] = int(pred)
        merged["detector_threshold"] = float(effective_threshold)
        merged["evaded"] = bool(int(item["label"]) == 1 and int(pred) == 0)
        scored.append(merged)

    metrics = compute_metrics(scored, detector_threshold=effective_threshold)
    return EvasionEvaluation(records=scored, metrics=metrics)
