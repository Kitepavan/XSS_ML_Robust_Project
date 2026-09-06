"""Focused tests for the XGBoost + lexical/handcrafted detector (TDD).

Covers the public API: fit on training records only, predict_proba/predict,
fixed-FPR threshold calibration on validation records, determinism, split
leakage guards, and dependency error behavior.

Tests require the real ``xgboost`` package and skip with a clear reason when
it is genuinely unavailable. No other model (RandomForest/GradientBoosting)
is ever substituted.
"""

from __future__ import annotations

import pytest

xgboost = pytest.importorskip(
    "xgboost",
    reason=(
        "xgboost is not installed; install the detector extra with "
        "'pip install \"xssharden[xgb]\"' (or 'pip install xgboost') "
        "to run these tests"
    ),
)


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
                "sample_id": f"xgb-train-benign-{i}",
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
                "sample_id": f"xgb-train-xss-{i}",
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
                "sample_id": f"xgb-val-benign-{i}",
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
                "sample_id": f"xgb-val-xss-{i}",
                "payload": payload,
                "label": 1,
                "source": "synth-b",
                "attack_category": "xss",
                "split": "validation",
            }
        )
    return records


def _make_detector(**overrides):
    from xssharden.detectors.xgboost_detector import XGBoostDetector

    params = {"random_state": 42, "n_estimators": 20, "max_depth": 3}
    params.update(overrides)
    return XGBoostDetector(**params)


# ===================================================================
# Section 1 - fit / predict API on training records
# ===================================================================

class TestFitPredictAPI:
    def test_fit_returns_self_and_predict_proba_shape(self):
        det = _make_detector()
        out = det.fit(_train_records())
        assert out is det
        proba = det.predict_proba(_train_records())
        assert proba.shape == (20, 2)

    def test_predict_proba_rows_sum_to_one_and_bounded(self):
        import numpy as np

        det = _make_detector().fit(_train_records())
        proba = det.predict_proba(_validation_records())
        assert proba.shape == (20, 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    def test_predict_returns_binary_labels_with_default_threshold(self):
        det = _make_detector().fit(_train_records())
        assert det.threshold_ == 0.5
        labels = det.predict(_validation_records())
        assert set(int(v) for v in labels.tolist()) <= {0, 1}
        assert labels.shape == (20,)

    def test_predict_accepts_plain_strings(self):
        det = _make_detector().fit(_train_records())
        proba = det.predict_proba(["<script>alert(9)</script>", "hello"])
        assert proba.shape == (2, 2)

    def test_fit_rejects_empty_train_records(self):
        from xssharden.detectors.xgboost_detector import XGBoostDetector

        with pytest.raises(ValueError, match="non-empty"):
            XGBoostDetector().fit([])

    def test_fit_rejects_single_class_train(self):
        from xssharden.detectors.xgboost_detector import XGBoostDetector

        recs = [r for r in _train_records() if r["label"] == 0]
        with pytest.raises(ValueError, match="both classes"):
            _make_detector().fit(recs)

    def test_deterministic_same_seed_same_probas(self):
        import numpy as np

        a = _make_detector().fit(_train_records())
        b = _make_detector().fit(_train_records())
        np.testing.assert_allclose(
            a.predict_proba(_validation_records()),
            b.predict_proba(_validation_records()),
        )

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            _make_detector().predict_proba(_validation_records())

    def test_uses_xgboost_classifier_not_substitute(self):
        from xgboost import XGBClassifier

        det = _make_detector().fit(_train_records())
        assert isinstance(det.model_, XGBClassifier)

    def test_no_randomforest_or_gradientboosting_in_source(self):
        import xssharden.detectors.xgboost_detector as mod

        src = open(mod.__file__, encoding="utf-8").read()
        assert "RandomForest" not in src
        assert "GradientBoosting" not in src
        assert "HistGradientBoosting" not in src


# ===================================================================
# Section 2 - lexical extractor consistency
# ===================================================================

class TestLexicalConsistency:
    def test_feature_width_matches_lexical_names(self):
        from xssharden.features.lexical import FEATURE_NAMES

        det = _make_detector().fit(_train_records())
        assert det.predict_proba(_train_records()).shape[1] == 2
        assert det.n_features_in_ == len(FEATURE_NAMES)

    def test_fit_and_inference_width_stable(self):
        det = _make_detector().fit(_train_records())
        width = det.predict_proba(_train_records()).shape[1]
        assert det.predict_proba(_validation_records()).shape[1] == width
        assert det.predict_proba(["<script>alert(9)</script>"]).shape == (1, 2)

    def test_uses_shared_lexical_module(self):
        import xssharden.detectors.xgboost_detector as mod

        src = open(mod.__file__, encoding="utf-8").read()
        assert "lexical" in src
        assert "featurize_lexical" in src

    def test_does_not_use_char_tfidf(self):
        import xssharden.detectors.xgboost_detector as mod

        src = open(mod.__file__, encoding="utf-8").read()
        assert "features.text" not in src
        assert "TfidfVectorizer" not in src


# ===================================================================
# Section 3 - fixed-FPR threshold calibration on validation records
# ===================================================================

class TestThresholdCalibration:
    def test_small_benign_validation_set_warns(self):
        det = _make_detector().fit(_train_records())
        with pytest.warns(UserWarning, match="fewer than 100 benign"):
            det.calibrate_threshold(_validation_records(), target_fpr=0.01)

    def test_at_least_100_benign_validation_records_does_not_warn(self):
        import warnings

        det = _make_detector().fit(_train_records())
        large_validation = _validation_records() * 10
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            det.calibrate_threshold(large_validation, target_fpr=0.01)

    def test_calibrate_returns_threshold_in_unit_interval(self):
        det = _make_detector().fit(_train_records())
        threshold = det.calibrate_threshold(_validation_records(), target_fpr=0.01)
        assert 0.0 <= threshold <= 1.0
        assert det.threshold_ == threshold

    def test_calibrated_fpr_within_tolerance_on_validation(self):
        import numpy as np

        det = _make_detector().fit(_train_records())
        det.calibrate_threshold(_validation_records(), target_fpr=0.5)
        proba = det.predict_proba(_validation_records())[:, 1]
        y_true = np.array([r["label"] for r in _validation_records()])
        benign_scores = proba[y_true == 0]
        # Threshold must equal the higher-interpolation (1 - FPR) quantile,
        # the same fixed-FPR rule as the baseline detector.
        expected = float(np.quantile(benign_scores, 0.5, method="higher"))
        assert det.threshold_ == expected
        # Booster scores tie on tiny fixtures, so the decision-rule (``>=``)
        # rate can include the tied mass; the strict rate stays bounded.
        strict_fpr = float((benign_scores > det.threshold_).mean())
        assert strict_fpr <= 0.5 + 1e-9

    def test_stricter_fpr_gives_higher_or_equal_threshold(self):
        det = _make_detector().fit(_train_records())
        loose = det.calibrate_threshold(_validation_records(), target_fpr=0.5)
        strict = det.calibrate_threshold(_validation_records(), target_fpr=0.01)
        assert strict >= loose

    def test_predict_uses_calibrated_threshold(self):
        import numpy as np

        det = _make_detector().fit(_train_records())
        det.calibrate_threshold(_validation_records(), target_fpr=0.01)
        proba = det.predict_proba(_validation_records())[:, 1]
        expected = (proba >= det.threshold_).astype(int)
        np.testing.assert_array_equal(det.predict(_validation_records()), expected)

    def test_calibrate_rejects_bad_target_fpr(self):
        det = _make_detector().fit(_train_records())
        for bad in (-0.1, 0.0, 1.0, 1.5, float("nan")):
            with pytest.raises(ValueError, match="target_fpr"):
                det.calibrate_threshold(_validation_records(), target_fpr=bad)

    def test_calibrate_rejects_empty_validation(self):
        det = _make_detector().fit(_train_records())
        with pytest.raises(ValueError, match="non-empty"):
            det.calibrate_threshold([], target_fpr=0.01)

    def test_calibrate_needs_benign_examples(self):
        det = _make_detector().fit(_train_records())
        only_xss = [r for r in _validation_records() if r["label"] == 1]
        with pytest.raises(ValueError, match="benign"):
            det.calibrate_threshold(only_xss, target_fpr=0.01)

    def test_calibrate_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            _make_detector().calibrate_threshold(
                _validation_records(), target_fpr=0.01
            )


# ===================================================================
# Section 4 - split leakage guards
# ===================================================================

class TestSplitLeakage:
    def test_fit_rejects_validation_records(self):
        with pytest.raises(ValueError, match="train"):
            _make_detector().fit(_validation_records())

    def test_fit_rejects_clean_test_records(self):
        recs = [dict(r, split="clean-test") for r in _train_records()]
        with pytest.raises(ValueError, match="clean-test"):
            _make_detector().fit(recs)

    def test_fit_rejects_test_split_records(self):
        recs = [dict(r, split="test") for r in _train_records()]
        with pytest.raises(ValueError, match="train"):
            _make_detector().fit(recs)

    def test_fit_rejects_mixed_splits(self):
        recs = _train_records() + [dict(_validation_records()[0])]
        with pytest.raises(ValueError, match="train"):
            _make_detector().fit(recs)

    def test_calibrate_rejects_train_clean_test_and_test(self):
        det = _make_detector().fit(_train_records())
        with pytest.raises(ValueError, match="validation"):
            det.calibrate_threshold(_train_records(), target_fpr=0.01)
        heldout = [dict(r, split="clean-test") for r in _validation_records()]
        with pytest.raises(ValueError, match="clean-test"):
            det.calibrate_threshold(heldout, target_fpr=0.01)
        final = [dict(r, split="test") for r in _validation_records()]
        with pytest.raises(ValueError, match="validation"):
            det.calibrate_threshold(final, target_fpr=0.01)


# ===================================================================
# Section 5 - dependency and reporting behavior
# ===================================================================

class TestDependencyAndReporting:
    def test_xgboost_import_is_lazy(self):
        import ast

        import xssharden.detectors.xgboost_detector as mod

        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        top_level = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        for node in top_level:
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] != "xgboost" for alias in node.names
                )
            else:
                assert (node.module or "").split(".")[0] != "xgboost"
        src = open(mod.__file__, encoding="utf-8").read()
        assert "import xgboost" in src

    def test_missing_xgboost_raises_clear_dependency_error(self, monkeypatch):
        import sys

        import xssharden.detectors.xgboost_detector as mod

        monkeypatch.setitem(sys.modules, "xgboost", None)
        with pytest.raises(ImportError, match="xgboost"):
            mod._load_xgboost()
        try:
            exc = None
            mod._load_xgboost()
        except ImportError as err:
            exc = err
        assert exc is not None
        assert "pip install" in str(exc)

    def test_describe_reports_state(self):
        det = _make_detector().fit(_train_records())
        info = det.describe()
        assert info["random_state"] == 42
        assert info["threshold"] == det.threshold_
        assert info["n_features"] == det.n_features_in_
        assert "feature_names" in info
        assert isinstance(info["feature_names"], list)

    def test_describe_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            _make_detector().describe()

    def test_never_executes_payloads(self):
        import xssharden.detectors.xgboost_detector as mod

        src = open(mod.__file__, encoding="utf-8").read().lower()
        for banned in ("os.system", "subprocess", "os.popen", "eval(", "exec("):
            assert banned not in src
