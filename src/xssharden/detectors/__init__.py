"""XSSHarden detector subpackage (baseline ML detectors).

Public API
----------
- :class:`~xssharden.detectors.baseline_lr.BaselineLRDetector` — TF-IDF
  character n-gram + calibrated Logistic Regression.
- :class:`~xssharden.detectors.xgboost_detector.XGBoostDetector` — lexical
  handcrafted features + XGBoost (requires the ``xgb`` extra).
- :func:`load_detector` — load a previously saved detector from a ``.pkl``
  file regardless of its concrete type.
"""

from __future__ import annotations

from typing import Any


def load_detector(path: str) -> Any:
    """Load a saved detector from *path* (joblib format).

    Supports :class:`~xssharden.detectors.baseline_lr.BaselineLRDetector`
    and :class:`~xssharden.detectors.xgboost_detector.XGBoostDetector`.

    Parameters
    ----------
    path:
        Filesystem path previously written by :meth:`save`.

    Returns
    -------
    BaselineLRDetector | XGBoostDetector
        The loaded, fitted detector object.

    Raises
    ------
    TypeError
        If the loaded object is not a recognised detector type.
    """
    import joblib

    from xssharden.detectors.baseline_lr import BaselineLRDetector
    from xssharden.detectors.xgboost_detector import XGBoostDetector

    obj = joblib.load(path)
    if not isinstance(obj, (BaselineLRDetector, XGBoostDetector)):
        raise TypeError(
            f"Expected a BaselineLRDetector or XGBoostDetector at {path!r}, "
            f"got {type(obj).__name__}"
        )
    return obj
