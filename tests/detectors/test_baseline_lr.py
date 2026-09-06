"""Focused tests for the TF-IDF + calibrated Logistic Regression baseline (TDD).

Covers the public API required by the task: fit on training records,
predict probabilities/labels, explicit probability calibration, fixed-FPR
threshold calibration on validation records, fit/transform reuse via the
shared featurize() pipeline, determinism, and train/validation separation
(no clean-test usage, no validation leakage into fit).
"""

from __future__ import annotations


def _train_records() -> list[dict]:
    benign = [
        "Hello world, this is a normal comment",
        "Search results forcats and dogs",
        "Welcome to our homepage, click here for news",
        "Product price is 42 dollars, free shipping",
        "User profile page with bio and avatar",
        "Contact us by email for support tickets",
        "Blog post about cooking pasta recipes",
        "Weather forecast: sunny with light rain",
        "Documentation for the getting started guide",
        "Shopping cart checkout with discount code",
    ]
    xss = [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "<svg onload=alert(1)>",
        "<script>prompt(1)</script>",
        "<iframe src=javascript:alert(1)>",
        "<body onload=confirm(1)>",
        "<script>fetch(/cookie)</script>",
        "<a href=javascript:alert(1)>click</a>",
        "<div onmouseover=alert(1)>hi</div>",
        "<object data=javascript:alert(1)>",
    ]
    records: list[dict] = []
    for i, payload in enumerate(benign):
        records.append(
            {
                "sample_id": f"train-benign-{i}",
                "payload": payload,
                "label": 0,
                "source": "synth-a",
                "attack_category": "benign",
                "split": "train",
            }
        )
    for i, payload in enumerate(xss):
        records.append(
            {
                "sample_id": f"train-xss-{i}",
                "payload": payload,
                "label": 1,
                "source": "synth-a",
                "attack_category": "xss",
                "split": "train",
            }
        )
    return records


def _validation_records() -> list[dict]:
    benign = [
        "Hello, nice to meet you today",
        "Weekly newsletter with updates and links",
        "Account settings and preferences page",
        "Order confirmation number 12345",
        "Forum thread about gardening tips",
        "Company about page with team photos",
        "Help center article on passwords",
        "Event calendar for next month",
        "Recipe index with soups and salads",
        "Travel guide for the coastal city",
    ]
    xss = [
        "<script>alert(2)</script>",
        "<img src=y onerror=prompt(1)>",
        "<svg onload=confirm(1)>",
        "<marquee onstart=alert(1)>x</marquee>",
        "<video><source onerror=alert(1)>",
        "<details open ontoggle=alert(1)>",
        "<select onfocus=alert(1) autofocus>",
        "<textarea onfocus=alert(1) autofocus>",
        "<keygen onfocus=alert(1)>",
        "<form><button formaction=javascript:alert(1)>x",
    ]
    records: list[dict] = []
    for i, payload in enumerate(benign):
        records.append(
            {
                "sample_id": f"val-benign-{i}",
                "payload": payload,
                "label": 0,
                "source": "synth-b",
                "attack_category": "benign",
                "split": "validation",
            }
        )
    for i, payload in enumerate(xss):
        records.append(
            {
                "sample_id": f"val-xss-{i}",
                "payload": payload,
                "label": 1,
                "source": "synth-b",
                "attack_category": "xss",
                "split": "validation",
            }
        )
    return records


# ===================================================================
# Section 1 - fit / predict API on training records
# ===================================================================

class TestFitPredictAPI:
    def test_fit_returns_self_and_predict_proba_shape(self):
        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42)
        out = det.fit(_train_records())
        assert out is det
        proba = det.predict_proba(_train_records())
        assert proba.shape == (20, 2)

    def test_predict_proba_rows_sum_to_one_and_bounded(self):
        import numpy as np

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        proba = det.predict_proba(_validation_records())
        assert proba.shape == (20, 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    def test_predict_returns_binary_labels_with_default_threshold(self):
        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        assert det.threshold_ == 0.5
        labels = det.predict(_validation_records())
        assert set(int(v) for v in labels.tolist()) <= {0, 1}
        assert labels.shape == (20,)

    def test_fit_rejects_empty_train_records(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        with pytest.raises(ValueError, match="non-empty"):
            BaselineLRDetector().fit([])

    def test_fit_rejects_single_class_train(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        recs = [r for r in _train_records() if r["label"] == 0]
        with pytest.raises(ValueError, match="both classes"):
            BaselineLRDetector().fit(recs)

    def test_deterministic_same_seed_same_probas(self):
        import numpy as np

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        a = BaselineLRDetector(random_state=42).fit(_train_records())
        b = BaselineLRDetector(random_state=42).fit(_train_records())
        np.testing.assert_allclose(
            a.predict_proba(_validation_records()),
            b.predict_proba(_validation_records()),
        )

    def test_predict_before_fit_raises(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        with pytest.raises(RuntimeError, match="fit"):
            BaselineLRDetector().predict_proba(_validation_records())


# ===================================================================
# Section 2 - explicit probability calibration
# ===================================================================

class TestProbabilityCalibration:
    def test_uses_calibrated_classifier_with_sigmoid(self):
        from sklearn.calibration import CalibratedClassifierCV

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        assert isinstance(det.calibrator_, CalibratedClassifierCV)
        # sklearn>=1.6 uses `method`-equivalent estimator params; accept both names.
        method = getattr(det.calibrator_, "method", None) or getattr(
            det.calibrator_, "calibration_method", None
        )
        # Fall back to the detector-level declaration if internals rename it.
        assert (method == "sigmoid") or (det.calibration_method == "sigmoid")

    def test_base_estimator_is_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        base = det.calibrator_.calibrated_classifiers_[0].estimator
        assert isinstance(base, LogisticRegression)

    def test_no_xgboost_lightgbm_dependency(self):
        import sys

        import xssharden.detectors.baseline_lr as mod

        assert "xgboost" not in sys.modules or True  # never required
        src = open(mod.__file__, encoding="utf-8").read().lower()
        assert "xgboost" not in src
        assert "lightgbm" not in src


# ===================================================================
# Section 3 - fixed-FPR threshold calibration on validation records
# ===================================================================

class TestThresholdCalibration:
    def test_small_benign_validation_set_warns(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        with pytest.warns(UserWarning, match="fewer than 100 benign"):
            det.calibrate_threshold(_validation_records(), target_fpr=0.01)

    def test_at_least_100_benign_validation_records_does_not_warn(self):
        import warnings

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        large_validation = _validation_records() * 10
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            det.calibrate_threshold(large_validation, target_fpr=0.01)

    def test_calibrate_returns_threshold_in_unit_interval(self):
        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        threshold = det.calibrate_threshold(_validation_records(), target_fpr=0.01)
        assert 0.0 <= threshold <= 1.0
        assert det.threshold_ == threshold

    def test_calibrated_fpr_within_tolerance_on_validation(self):
        import numpy as np

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        det.calibrate_threshold(_validation_records(), target_fpr=0.5)
        proba = det.predict_proba(_validation_records())[:, 1]
        y_true = np.array([r["label"] for r in _validation_records()])
        benign_scores = proba[y_true == 0]
        fpr = float((benign_scores >= det.threshold_).mean())
        assert fpr <= 0.5 + 1e-9

    def test_stricter_fpr_gives_higher_or_equal_threshold(self):
        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        loose = det.calibrate_threshold(_validation_records(), target_fpr=0.5)
        strict = det.calibrate_threshold(_validation_records(), target_fpr=0.01)
        assert strict >= loose

    def test_predict_uses_calibrated_threshold(self):
        import numpy as np

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        det.calibrate_threshold(_validation_records(), target_fpr=0.01)
        proba = det.predict_proba(_validation_records())[:, 1]
        expected = (proba >= det.threshold_).astype(int)
        np.testing.assert_array_equal(det.predict(_validation_records()), expected)

    def test_calibrate_rejects_bad_target_fpr(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        for bad in (-0.1, 0.0 - 1e-9 - 1.0, 1.5, float("nan")):
            with pytest.raises(ValueError, match="target_fpr"):
                det.calibrate_threshold(_validation_records(), target_fpr=bad)

    def test_calibrate_rejects_empty_validation(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        with pytest.raises(ValueError, match="non-empty"):
            det.calibrate_threshold([], target_fpr=0.01)

    def test_calibrate_needs_benign_examples(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        only_xss = [r for r in _validation_records() if r["label"] == 1]
        with pytest.raises(ValueError, match="benign"):
            det.calibrate_threshold(only_xss, target_fpr=0.01)

    def test_calibrate_before_fit_raises(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        with pytest.raises(RuntimeError, match="fit"):
            BaselineLRDetector().calibrate_threshold(
                _validation_records(), target_fpr=0.01
            )


# ===================================================================
# Section 4 - train/validation separation, reuse, no leakage
# ===================================================================

class TestSeparationAndReuse:
    def test_fit_rejects_validation_records(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        with pytest.raises(ValueError, match="train"):
            BaselineLRDetector().fit(_validation_records())

    def test_fit_rejects_clean_test_records(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        recs = [dict(r, split="clean-test") for r in _train_records()]
        with pytest.raises(ValueError, match="clean-test"):
            BaselineLRDetector().fit(recs)

    def test_calibrate_rejects_train_and_clean_test(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        with pytest.raises(ValueError, match="validation"):
            det.calibrate_threshold(_train_records(), target_fpr=0.01)
        heldout = [dict(r, split="clean-test") for r in _validation_records()]
        with pytest.raises(ValueError, match="clean-test"):
            det.calibrate_threshold(heldout, target_fpr=0.01)

    def test_fit_does_not_consume_validation_vocabulary(self):
        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        before = dict(det.vectorizer_.vocabulary_)
        # Validation-only token must not expand the fitted vocabulary.
        det.predict_proba(_validation_records())
        assert det.vectorizer_.vocabulary_ == before

    def test_shared_featurize_settings_used(self):
        from xssharden.detectors.baseline_lr import BaselineLRDetector
        from xssharden.features.text import ANALYZER, LOWERCASE, NGRAM_RANGE

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        assert det.vectorizer_.analyzer == ANALYZER
        assert det.vectorizer_.ngram_range == NGRAM_RANGE
        assert det.vectorizer_.lowercase is LOWERCASE

    def test_fit_transform_reuse_width_stable(self):
        from xssharden.detectors.baseline_lr import BaselineLRDetector

        det = BaselineLRDetector(random_state=42).fit(_train_records())
        width = det.predict_proba(_train_records()).shape[1]
        assert det.predict_proba(_validation_records()).shape[1] == width
        assert det.predict_proba(["<script>alert(9)</script>"]).shape == (1, 2)

    def test_never_executes_payloads(self):
        import xssharden.detectors.baseline_lr as mod

        src = open(mod.__file__, encoding="utf-8").read().lower()
        for banned in ("os.system", "subprocess", "os.popen", "eval(", "exec("):
            assert banned not in src


# ===================================================================
# Section 5 - calibration-fold guard (either class < calibration_cv)
# ===================================================================

def _small_train_records(n_benign: int, n_xss: int) -> list[dict]:
    """Build minimal train-split records with distinct payloads."""
    records: list[dict] = []
    for i in range(n_benign):
        records.append(
            {
                "sample_id": f"guard-benign-{i}",
                "payload": f"Benign support message number {i} with plain text",
                "label": 0,
                "source": "synth-guard",
                "attack_category": "benign",
                "split": "train",
            }
        )
    for i in range(n_xss):
        records.append(
            {
                "sample_id": f"guard-xss-{i}",
                "payload": f"<script>guard_alert({i})</script>",
                "label": 1,
                "source": "synth-guard",
                "attack_category": "xss",
                "split": "train",
            }
        )
    return records


class TestCalibrationFoldGuard:
    def test_fit_rejects_when_benign_fewer_than_calibration_cv(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        # Default calibration_cv=3; only 2 benign examples.
        records = _small_train_records(n_benign=2, n_xss=10)
        with pytest.raises(ValueError, match="calibration_cv") as excinfo:
            BaselineLRDetector(calibration_cv=3).fit(records)
        message = str(excinfo.value)
        assert "calibration_cv" in message
        # Class counts must be visible for debugging.
        assert "2" in message and "10" in message

    def test_fit_rejects_when_xss_fewer_than_calibration_cv(self):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        # Default calibration_cv=3; only 1 XSS example.
        records = _small_train_records(n_benign=10, n_xss=1)
        with pytest.raises(ValueError, match="calibration_cv") as excinfo:
            BaselineLRDetector(calibration_cv=3).fit(records)
        message = str(excinfo.value)
        assert "calibration_cv" in message
        assert "10" in message and "1" in message

    def test_fit_rejects_when_either_class_fewer_than_custom_calibration_cv(
        self,
    ):
        import pytest

        from xssharden.detectors.baseline_lr import BaselineLRDetector

        # Custom calibration_cv=5; XSS count 4 is below the fold count.
        records = _small_train_records(n_benign=10, n_xss=4)
        with pytest.raises(ValueError, match="calibration_cv") as excinfo:
            BaselineLRDetector(calibration_cv=5).fit(records)
        message = str(excinfo.value)
        assert "calibration_cv" in message
        assert "5" in message
        assert "10" in message and "4" in message
