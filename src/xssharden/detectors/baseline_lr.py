"""Baseline XSS detector: character TF-IDF + calibrated Logistic Regression.

This is Detector 1 from the research plan: the simple baseline. All text
features come from the single shared :func:`xssharden.features.text.featurize`
pipeline so every detector uses identical features.

Two kinds of calibration are kept explicit and separate:

1. Probability calibration — a ``CalibratedClassifierCV`` (sigmoid) wrapper
   around ``LogisticRegression``, fitted with cross-validation *inside* the
   training records only.
2. Decision-threshold calibration — :meth:`BaselineLRDetector.calibrate_threshold`
   picks the operating threshold on held-out validation records at a requested
   false-positive rate (for example 0.01), keeping the threshold fixed for
   later adversarial comparisons.

Train/validation separation is enforced by the ``split`` field: :meth:`fit`
accepts only ``split == "train"`` records and :meth:`calibrate_threshold`
accepts only ``split == "validation"`` records. ``clean-test`` records are
never usable for fitting or threshold tuning. Payload strings are treated as
plain text and are never run as code.

Persistence
-----------
A fitted detector can be saved to and loaded from disk with :meth:`save` and
:meth:`load` (classmethod). Serialization uses ``joblib`` (bundled with
scikit-learn) and stores the full object — vectorizer, calibrated classifier,
and decision threshold — in a single ``.pkl`` file.
"""

from __future__ import annotations

from typing import Any, Sequence
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from xssharden.features.text import featurize

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
CLEAN_TEST_SPLIT = "clean-test"

Record = dict[str, Any]


def _payloads_from_inputs(inputs: Sequence[Record] | Sequence[str]) -> list[str]:
    """Extract raw payload strings from record dicts or plain strings."""
    if inputs is None or len(inputs) == 0:  # type: ignore[arg-type]
        raise ValueError("inputs must be a non-empty sequence")
    first = inputs[0]
    if isinstance(first, dict):
        payloads: list[str] = []
        for index, record in enumerate(inputs):  # type: ignore[union-attr]
            if not isinstance(record, dict):
                raise TypeError(
                    f"record at index {index} must be dict, "
                    f"got {type(record).__name__}"
                )
            payload = record.get("payload")
            if not isinstance(payload, str):
                raise TypeError(
                    f"record at index {index}: 'payload' must be str, "
                    f"got {type(payload).__name__}"
                )
            if not payload.strip():
                raise ValueError(
                    f"record at index {index}: 'payload' must be non-empty"
                )
            payloads.append(payload)
        return payloads
    payloads = []
    for index, item in enumerate(inputs):  # type: ignore[union-attr]
        if not isinstance(item, str):
            raise TypeError(
                f"payload at index {index} must be str, "
                f"got {type(item).__name__}"
            )
        payloads.append(item)
    if len(payloads) == 0:
        raise ValueError("inputs must be a non-empty sequence")
    return payloads


def _labels_from_records(records: Sequence[Record]) -> np.ndarray:
    """Extract integer 0/1 labels from training/validation records."""
    labels: list[int] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"record at index {index} must be dict, "
                f"got {type(record).__name__}"
            )
        if "label" not in record:
            raise ValueError(f"record at index {index}: missing 'label'")
        label = record["label"]
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
        labels.append(int(label))
    return np.asarray(labels, dtype=int)


def _check_target_fpr(target_fpr: float) -> float:
    """Validate the requested false-positive rate."""
    try:
        value = float(target_fpr)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"target_fpr must be a float in (0, 1), got {target_fpr!r}"
        ) from exc
    if not np.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError(
            f"target_fpr must be a float in (0, 1), got {target_fpr!r}"
        )
    return value


class BaselineLRDetector:
    """TF-IDF character features plus calibrated Logistic Regression.

    Parameters
    ----------
    random_state:
        Deterministic seed for the base classifier and the internal
        cross-validation splitter.
    C:
        Inverse regularization strength for ``LogisticRegression``.
    max_iter:
        Maximum solver iterations for ``LogisticRegression``.
    calibration_method:
        Probability-calibration method; only ``"sigmoid"`` is supported.
    calibration_cv:
        Number of cross-validation folds used inside the training data
        for probability calibration.
    """

    def __init__(
        self,
        random_state: int = 42,
        C: float = 1.0,  # noqa: N803 - matches scikit-learn naming
        max_iter: int = 1000,
        calibration_method: str = "sigmoid",
        calibration_cv: int = 3,
    ) -> None:
        if calibration_method != "sigmoid":
            raise ValueError(
                "calibration_method must be 'sigmoid', "
                f"got {calibration_method!r}"
            )
        if calibration_cv < 2:
            raise ValueError(
                f"calibration_cv must be >= 2, got {calibration_cv}"
            )
        self.random_state = random_state
        self.C = C
        self.max_iter = max_iter
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv
        self.threshold_: float = 0.5
        self.vectorizer_: Any | None = None
        self.calibrator_: CalibratedClassifierCV | None = None

    # -- fitting ------------------------------------------------------
    def fit(self, train_records: Sequence[Record]) -> BaselineLRDetector:
        """Fit the vectorizer and calibrated classifier on training records.

        Only ``split == "train"`` records are accepted so validation and
        ``clean-test`` data can never leak into fitting.
        """
        if train_records is None or len(train_records) == 0:  # type: ignore[arg-type]
            raise ValueError("train_records must be a non-empty sequence")
        for index, record in enumerate(train_records):
            if not isinstance(record, dict):
                raise TypeError(
                    f"record at index {index} must be dict, "
                    f"got {type(record).__name__}"
                )
            split = record.get("split")
            if split == CLEAN_TEST_SPLIT:
                raise ValueError(
                    "fit must never use clean-test records; "
                    f"record at index {index} has split='clean-test'"
                )
            if split != TRAIN_SPLIT:
                raise ValueError(
                    "fit expects only split='train' records; "
                    f"record at index {index} has split={split!r}"
                )
        payloads = _payloads_from_inputs(train_records)
        y = _labels_from_records(train_records)
        if len(np.unique(y)) < 2:
            raise ValueError("train_records must contain both classes 0 and 1")
        n_class0 = int((y == 0).sum())
        n_class1 = int((y == 1).sum())
        if min(n_class0, n_class1) < self.calibration_cv:
            raise ValueError(
                f"train_records has too few examples per class for "
                f"calibration_cv={self.calibration_cv}: each class needs at "
                f"least calibration_cv examples "
                f"(got class 0 count={n_class0}, class 1 count={n_class1})"
            )

        matrix, vectorizer = featurize(payloads)
        base = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        splitter = StratifiedKFold(
            n_splits=self.calibration_cv,
            shuffle=True,
            random_state=self.random_state,
        )
        calibrator = CalibratedClassifierCV(
            estimator=base,
            method=self.calibration_method,
            cv=splitter,
        )
        calibrator.fit(matrix, y)
        self.vectorizer_ = vectorizer
        self.calibrator_ = calibrator
        self.threshold_ = 0.5
        return self

    # -- inference ----------------------------------------------------
    def _require_fitted(self) -> None:
        if self.vectorizer_ is None or self.calibrator_ is None:
            raise RuntimeError("BaselineLRDetector must be fitted via fit() first")

    def _matrix_for(self, inputs: Sequence[Record] | Sequence[str]) -> csr_matrix:
        self._require_fitted()
        payloads = _payloads_from_inputs(inputs)
        matrix, _ = featurize(payloads, vectorizer=self.vectorizer_)
        return matrix

    def predict_proba(
        self, inputs: Sequence[Record] | Sequence[str]
    ) -> np.ndarray:
        """Return calibrated ``(n, 2)`` class probabilities."""
        matrix = self._matrix_for(inputs)
        assert self.calibrator_ is not None
        return np.asarray(self.calibrator_.predict_proba(matrix))

    def predict(self, inputs: Sequence[Record] | Sequence[str]) -> np.ndarray:
        """Return binary labels using the current operating threshold."""
        proba = self.predict_proba(inputs)[:, 1]
        return (proba >= self.threshold_).astype(int)

    # -- threshold calibration ----------------------------------------
    def calibrate_threshold(
        self,
        validation_records: Sequence[Record],
        target_fpr: float = 0.01,
    ) -> float:
        """Calibrate the decision threshold on validation records.

        The threshold is the ``1 - target_fpr`` quantile (higher
        interpolation) of the benign validation scores, so the empirical
        validation FPR stays at or below the request up to quantisation.
        The fitted vectorizer is reused without refitting.
        """
        self._require_fitted()
        target = _check_target_fpr(target_fpr)
        if validation_records is None or len(validation_records) == 0:  # type: ignore[arg-type]
            raise ValueError("validation_records must be a non-empty sequence")
        for index, record in enumerate(validation_records):
            if not isinstance(record, dict):
                raise TypeError(
                    f"record at index {index} must be dict, "
                    f"got {type(record).__name__}"
                )
            split = record.get("split")
            if split == CLEAN_TEST_SPLIT:
                raise ValueError(
                    "calibrate_threshold must never use clean-test records; "
                    f"record at index {index} has split='clean-test'"
                )
            if split != VALIDATION_SPLIT:
                raise ValueError(
                    "calibrate_threshold expects only split='validation' "
                    f"records; record at index {index} has split={split!r}"
                )
        y_true = _labels_from_records(validation_records)
        benign_mask = y_true == 0
        if int(benign_mask.sum()) == 0:
            raise ValueError(
                "validation_records must contain at least one benign (label 0) "
                "record to measure FPR"
            )
        scores = self.predict_proba(validation_records)[:, 1]
        benign_scores = scores[benign_mask]
        if len(benign_scores) < 100:
            warnings.warn(
                "threshold calibration uses fewer than 100 benign validation "
                f"records ({len(benign_scores)}); the empirical threshold may "
                "be unstable",
                UserWarning,
                stacklevel=2,
            )
        quantile = 1.0 - target
        try:
            threshold = float(
                np.quantile(benign_scores, quantile, method="higher")
            )
        except TypeError:  # numpy < 1.22 fallback
            threshold = float(
                np.quantile(benign_scores, quantile, interpolation="higher")
            )
        threshold = min(max(threshold, 0.0), 1.0)
        self.threshold_ = threshold
        return threshold

    # -- persistence --------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the fitted detector to *path* using ``joblib``.

        The full detector object (vectorizer, calibrated classifier, and
        decision threshold) is written to a single ``.pkl`` file. The
        detector must be fitted before calling this method.

        Parameters
        ----------
        path:
            Filesystem path to write. Parent directories must exist.
        """
        import joblib

        self._require_fitted()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaselineLRDetector":
        """Load a previously saved :class:`BaselineLRDetector` from *path*.

        Parameters
        ----------
        path:
            Filesystem path written by :meth:`save`.

        Returns
        -------
        BaselineLRDetector
            The loaded, fitted detector.

        Raises
        ------
        TypeError
            If the loaded object is not a :class:`BaselineLRDetector`.
        """
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a BaselineLRDetector at {path!r}, "
                f"got {type(obj).__name__}"
            )
        return obj

    # -- reporting helper ----------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Return a small summary of the fitted detector state."""
        self._require_fitted()
        assert self.vectorizer_ is not None
        vocab_size = len(getattr(self.vectorizer_, "vocabulary_", {}))
        return {
            "detector": "baseline-lr",
            "random_state": self.random_state,
            "C": self.C,
            "max_iter": self.max_iter,
            "calibration_method": self.calibration_method,
            "calibration_cv": self.calibration_cv,
            "threshold": self.threshold_,
            "vocab_size": vocab_size,
            "analyzer": getattr(self.vectorizer_, "analyzer", None),
            "ngram_range": getattr(self.vectorizer_, "ngram_range", None),
        }
