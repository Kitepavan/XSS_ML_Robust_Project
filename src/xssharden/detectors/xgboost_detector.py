"""Second XSS detector: lexical/handcrafted features plus XGBoost.

This is Detector 2 from the research plan. All features come from the
single shared :func:`xssharden.features.lexical.featurize_lexical`
extractor, used identically for fitting and inference — the pre-existing
character TF-IDF pipeline is intentionally not used here.

The ``xgboost`` package is imported lazily inside :func:`_load_xgboost`
so that importing this module (and running feature-extraction code) never
requires the dependency. Fitting or predicting without ``xgboost``
installed raises a clear ``ImportError`` with the exact install command.
No other gradient-boosting implementation is substituted.

Split discipline mirrors the baseline detector: :meth:`fit` accepts only
``split == "train"`` records and :meth:`calibrate_threshold` accepts only
``split == "validation"`` records, so ``clean-test`` (and final ``test``)
data can never leak into fitting or threshold tuning. Payload strings are
treated as inert text and are never run as code.

Persistence
-----------
A fitted detector can be saved to and loaded from disk with :meth:`save` and
:meth:`load` (classmethod). Serialization uses ``joblib`` (bundled with
scikit-learn) and stores the full object — XGBoost booster and threshold — in
a single ``.pkl`` file.
"""

from __future__ import annotations

from typing import Any, Sequence
import warnings

import numpy as np

from xssharden.features.lexical import FEATURE_NAMES, featurize_lexical

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
CLEAN_TEST_SPLIT = "clean-test"

INSTALL_HINT = 'pip install "xssharden[xgb]"  # or: pip install xgboost>=2.0'

Record = dict[str, Any]


def _load_xgboost() -> Any:
    """Import and return the ``xgboost`` package, lazily.

    Raises
    ------
    ImportError
        If ``xgboost`` is not installed, with the exact install command.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGBoostDetector but is not installed. "
            f"Install the detector extra with {INSTALL_HINT}."
        ) from exc
    return xgb


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


class XGBoostDetector:
    """Lexical/handcrafted features plus an XGBoost classifier.

    Parameters
    ----------
    random_state:
        Deterministic seed forwarded to ``XGBClassifier``.
    n_estimators:
        Number of boosting rounds; must be a positive integer.
    max_depth:
        Maximum tree depth; must be a positive integer.
    learning_rate:
        Boosting learning rate; must be in ``(0, 1]``.
    subsample:
        Row subsample ratio; ``1.0`` (the default) disables sampling so
        fitting is fully deterministic given ``random_state``.
    colsample_bytree:
        Column subsample ratio per tree; ``1.0`` disables sampling.
    reg_lambda:
        L2 regularization term; must be non-negative.
    n_jobs:
        Parallel threads for XGBoost; ``1`` (the default) keeps training
        deterministic.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        n_jobs: int = 1,
    ) -> None:
        if not isinstance(n_estimators, (int, np.integer)) or int(n_estimators) < 1:
            raise ValueError(f"n_estimators must be a positive int, got {n_estimators!r}")
        if not isinstance(max_depth, (int, np.integer)) or int(max_depth) < 1:
            raise ValueError(f"max_depth must be a positive int, got {max_depth!r}")
        try:
            learning_rate = float(learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"learning_rate must be a float in (0, 1], got {learning_rate!r}"
            ) from exc
        if not np.isfinite(learning_rate) or not (0.0 < learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be a float in (0, 1], got {learning_rate!r}"
            )
        for name, value in (("subsample", subsample), ("colsample_bytree", colsample_bytree)):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must be a float in (0, 1], got {value!r}"
                ) from exc
            if not np.isfinite(numeric) or not (0.0 < numeric <= 1.0):
                raise ValueError(f"{name} must be a float in (0, 1], got {value!r}")
        try:
            reg_lambda = float(reg_lambda)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"reg_lambda must be a non-negative float, got {reg_lambda!r}"
            ) from exc
        if not np.isfinite(reg_lambda) or reg_lambda < 0.0:
            raise ValueError(
                f"reg_lambda must be a non-negative float, got {reg_lambda!r}"
            )
        self.random_state = random_state
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = learning_rate
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.threshold_: float = 0.5
        self.model_: Any | None = None
        self.n_features_in_: int | None = None

    # -- fitting ------------------------------------------------------
    def fit(self, train_records: Sequence[Record]) -> XGBoostDetector:
        """Fit the XGBoost classifier on training records.

        Only ``split == "train"`` records are accepted so validation,
        ``clean-test``, and final ``test`` data can never leak into fitting.
        Features come from the shared lexical extractor.
        """
        xgb = _load_xgboost()
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

        matrix = featurize_lexical(payloads)
        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            eval_metric="logloss",
        )
        model.fit(matrix, y)
        self.model_ = model
        self.n_features_in_ = int(matrix.shape[1])
        self.threshold_ = 0.5
        return self

    # -- inference ----------------------------------------------------
    def _require_fitted(self) -> None:
        if self.model_ is None or self.n_features_in_ is None:
            raise RuntimeError("XGBoostDetector must be fitted via fit() first")

    def _matrix_for(self, inputs: Sequence[Record] | Sequence[str]) -> np.ndarray:
        self._require_fitted()
        payloads = _payloads_from_inputs(inputs)
        return featurize_lexical(payloads)

    def predict_proba(
        self, inputs: Sequence[Record] | Sequence[str]
    ) -> np.ndarray:
        """Return ``(n, 2)`` class probabilities from the fitted booster."""
        matrix = self._matrix_for(inputs)
        assert self.model_ is not None
        return np.asarray(self.model_.predict_proba(matrix))

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
        The lexical extractor is stateless, so no refitting can occur.
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

        The full detector object (XGBoost booster and decision threshold)
        is written to a single ``.pkl`` file. The detector must be fitted
        before calling this method.

        Parameters
        ----------
        path:
            Filesystem path to write. Parent directories must exist.
        """
        import joblib

        self._require_fitted()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostDetector":
        """Load a previously saved :class:`XGBoostDetector` from *path*.

        Parameters
        ----------
        path:
            Filesystem path written by :meth:`save`.

        Returns
        -------
        XGBoostDetector
            The loaded, fitted detector.

        Raises
        ------
        TypeError
            If the loaded object is not a :class:`XGBoostDetector`.
        """
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected an XGBoostDetector at {path!r}, "
                f"got {type(obj).__name__}"
            )
        return obj

    # -- reporting helper ----------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Return a small summary of the fitted detector state."""
        self._require_fitted()
        return {
            "detector": "xgboost-lexical",
            "model": type(self.model_).__name__,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "threshold": self.threshold_,
            "n_features": self.n_features_in_,
            "feature_names": list(FEATURE_NAMES),
        }
