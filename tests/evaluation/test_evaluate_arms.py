"""Focused tests for independent four-arm hardening evaluation (Task 19).

Uses fake detectors only: no network, no model training, no payload
execution. Every test is deterministic.
"""

from __future__ import annotations

import copy
import json

import pytest


class PayloadFakeDetector:
    """Fake detector keyed by payload string.

    ``table`` maps payload -> (malicious_score, predicted_label).
    ``predict_proba``/``predict`` are read-only; ``fit`` and
    ``calibrate_threshold`` raise so any training/tuning attempt fails.
    """

    def __init__(self, table, threshold=0.5):
        self._table = dict(table)
        self.threshold_ = float(threshold)
        self.calls: list[str] = []

    def predict_proba(self, records):
        import numpy as np

        self.calls.append("predict_proba")
        rows = []
        for record in records:
            score, _ = self._table[record["payload"]]
            rows.append([1.0 - float(score), float(score)])
        return np.asarray(rows, dtype=float)

    def predict(self, records):
        import numpy as np

        self.calls.append("predict")
        return np.asarray(
            [self._table[record["payload"]][1] for record in records], dtype=int
        )

    def fit(self, *args, **kwargs):  # pragma: no cover - guard only
        raise AssertionError("evaluation must never train the detector")

    def calibrate_threshold(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("evaluation must never tune the detector")


def _clean_records():
    return [
        {"sample_id": "c1", "payload": "hello world", "label": 0,
         "source": "s1", "attack_category": "benign", "split": "clean-test",
         "note": "keep-me"},
        {"sample_id": "c2", "payload": "good day friend", "label": 0,
         "source": "s1", "attack_category": "benign", "split": "clean-test"},
        {"sample_id": "c3", "payload": "<script>alert(1)</script>", "label": 1,
         "source": "s2", "attack_category": "xss", "split": "clean-test"},
        {"sample_id": "c4", "payload": "<img src=x onerror=alert(1)>", "label": 1,
         "source": "s2", "attack_category": "xss", "split": "clean-test"},
    ]


def _adv_records():
    return [
        {"variant_id": "a1", "seed_id": "s", "payload": "<svg onload=alert(1)>",
         "label": 1, "valid": True, "mutation_category": "encoding",
         "split": "adv_test", "probe": "p1"},
        {"variant_id": "a2", "seed_id": "s", "payload": "<b>not here</b>",
         "label": 1, "valid": True, "mutation_category": "case_variation",
         "split": "adv_test"},
        {"variant_id": "a3", "seed_id": "s", "payload": "<div>broken",
         "label": 1, "valid": False, "mutation_category": "encoding",
         "split": "adv_test"},
        {"variant_id": "a4", "seed_id": "s", "payload": "just text",
         "label": 0, "valid": True, "mutation_category": "encoding",
         "split": "adv_test"},
    ]


def _mixed_table():
    # c1 TN, c2 FP, c3 TP, c4 FN  -> every clean confusion cell == 1.
    # a1 valid evasion, a2 valid caught, a3 invalid evasion, a4 benign.
    return {
        "hello world": (0.1, 0),
        "good day friend": (0.9, 1),
        "<script>alert(1)</script>": (0.8, 1),
        "<img src=x onerror=alert(1)>": (0.2, 0),
        "<svg onload=alert(1)>": (0.1, 0),
        "<b>not here</b>": (0.9, 1),
        "<div>broken": (0.05, 0),
        "just text": (0.9, 1),
    }


def _four_detectors():
    table = _mixed_table()
    perfect = PayloadFakeDetector(
        {payload: (0.05 if label == 0 else 0.95, label)
         for payload, label in [
            ("hello world", 0), ("good day friend", 0),
            ("<script>alert(1)</script>", 1),
            ("<img src=x onerror=alert(1)>", 1),
            ("<svg onload=alert(1)>", 1), ("<b>not here</b>", 1),
            ("<div>broken", 1), ("just text", 0)]})
    worst = PayloadFakeDetector(
        {payload: (0.95 if label == 0 else 0.05, 1 - label)
         for payload, label in [
            ("hello world", 0), ("good day friend", 0),
            ("<script>alert(1)</script>", 1),
            ("<img src=x onerror=alert(1)>", 1),
            ("<svg onload=alert(1)>", 1), ("<b>not here</b>", 1),
            ("<div>broken", 1), ("just text", 0)]})
    mixed = PayloadFakeDetector(dict(table))
    all_benign = PayloadFakeDetector({payload: (0.1, 0) for payload in table})
    return {
        "baseline": worst,
        "naive": mixed,
        "random_valid": all_benign,
        "selective": perfect,
    }


class TestFourArmComparison:
    def test_all_four_arms_evaluated_with_counts(self):
        from xssharden.evaluation import evaluate_arms

        result = evaluate_arms(_four_detectors(), _clean_records(), _adv_records())
        assert set(result.arms) == {"baseline", "naive", "random_valid", "selective"}
        assert result.clean_test_count == 4
        assert result.adversarial_count == 4
        for name in ("baseline", "naive", "random_valid", "selective"):
            arm = result.arms[name]
            assert arm.arm == name
            assert arm.clean_count == 4
            assert arm.adversarial_count == 4

    def test_perfect_arm_beats_worst_arm(self):
        from xssharden.evaluation import evaluate_arms

        result = evaluate_arms(_four_detectors(), _clean_records(), _adv_records())
        selective = result.arms["selective"]
        baseline = result.arms["baseline"]
        assert selective.clean_metrics.accuracy == 1.0
        assert baseline.clean_metrics.accuracy == 0.0
        assert (selective.adversarial_metrics.valid_malicious_evasion_rate
                == 0.0)
        assert (baseline.adversarial_metrics.valid_malicious_evasion_rate
                == 1.0)

    def test_result_is_json_serializable(self):
        from xssharden.evaluation import evaluate_arms

        result = evaluate_arms(_four_detectors(), _clean_records(), _adv_records())
        payload = json.dumps(result.to_dict())
        assert json.loads(payload)["clean_test_count"] == 4


class TestMetricCorrectness:
    def test_clean_confusion_and_derived_metrics(self):
        from xssharden.evaluation import evaluate_arms

        detectors = _four_detectors()
        result = evaluate_arms(
            {"baseline": detectors["naive"], "naive": detectors["naive"],
             "random_valid": detectors["naive"],
             "selective": detectors["naive"]},
            _clean_records(), _adv_records())
        clean = result.arms["baseline"].clean_metrics
        assert (clean.tp, clean.tn, clean.fp, clean.fn) == (1, 1, 1, 1)
        assert clean.total == 4
        assert (clean.positives, clean.negatives) == (2, 2)
        assert clean.accuracy == pytest.approx(0.5)
        assert clean.precision == pytest.approx(0.5)
        assert clean.recall == pytest.approx(0.5)
        assert clean.f1 == pytest.approx(0.5)
        assert clean.fpr == pytest.approx(0.5)

    def test_adversarial_rates_match_scoring_semantics(self):
        from xssharden.evaluation import evaluate_arms

        detectors = _four_detectors()
        result = evaluate_arms(
            {"baseline": detectors["naive"], "naive": detectors["naive"],
             "random_valid": detectors["naive"],
             "selective": detectors["naive"]},
            _clean_records(), _adv_records())
        adv = result.arms["naive"].adversarial_metrics
        assert adv.total_records == 4
        assert adv.malicious_count == 3
        assert adv.evasion_count == 2  # a1 valid + a3 invalid
        assert adv.evasion_rate == pytest.approx(2 / 3)
        assert adv.valid_malicious_count == 2
        assert adv.valid_evasion_count == 1
        assert adv.valid_malicious_evasion_rate == pytest.approx(0.5)
        assert adv.raw_evasion_rate == pytest.approx(0.5)

    def test_explicit_denominators_in_dicts(self):
        from xssharden.evaluation import evaluate_arms

        result = evaluate_arms(_four_detectors(), _clean_records(), _adv_records())
        arm_dict = result.arms["naive"].to_dict()
        assert arm_dict["clean_metrics"]["total"] == 4
        assert arm_dict["adversarial_metrics"]["total_records"] == 4
        assert arm_dict["adversarial_metrics"]["valid_malicious_count"] == 2


class TestSplitGuards:
    def _arms(self):
        return _four_detectors()

    def test_clean_record_with_wrong_split_rejected(self):
        from xssharden.evaluation import evaluate_arms

        clean = _clean_records()
        clean[0] = dict(clean[0], split="adv_test")
        with pytest.raises(ValueError):
            evaluate_arms(self._arms(), clean, _adv_records())

    def test_clean_record_with_validation_split_rejected(self):
        from xssharden.evaluation import evaluate_arms

        clean = _clean_records()
        clean[1] = dict(clean[1], split="validation")
        with pytest.raises(ValueError):
            evaluate_arms(self._arms(), clean, _adv_records())

    def test_clean_record_missing_split_rejected(self):
        from xssharden.evaluation import evaluate_arms

        clean = _clean_records()
        bad = dict(clean[0])
        del bad["split"]
        with pytest.raises(ValueError):
            evaluate_arms(self._arms(), [bad], _adv_records())

    def test_adv_record_with_clean_test_split_rejected(self):
        from xssharden.evaluation import evaluate_arms

        adv = _adv_records()
        adv[0] = dict(adv[0], split="clean-test")
        with pytest.raises(ValueError):
            evaluate_arms(self._arms(), _clean_records(), adv)

    def test_adv_record_with_validation_split_rejected(self):
        from xssharden.evaluation import evaluate_arms

        adv = _adv_records()
        adv[2] = dict(adv[2], split="validation")
        with pytest.raises(ValueError):
            evaluate_arms(self._arms(), _clean_records(), adv)

    def test_adv_record_with_test_split_rejected(self):
        from xssharden.evaluation import evaluate_arms

        adv = _adv_records()
        adv[0] = dict(adv[0], split="test")
        with pytest.raises(ValueError):
            evaluate_arms(self._arms(), _clean_records(), adv)

    def test_refusing_tuning_split_even_when_requested(self):
        from xssharden.evaluation import evaluate_arms

        with pytest.raises(ValueError):
            evaluate_arms(self._arms(), _clean_records(), _adv_records(),
                           adv_split="validation")


class TestNoFittingNoTuning:
    def test_only_read_only_scoring_calls(self):
        from xssharden.evaluation import evaluate_arms

        arms = _four_detectors()
        evaluate_arms(arms, _clean_records(), _adv_records())
        for detector in arms.values():
            assert set(detector.calls) == {"predict_proba", "predict"}

    def test_module_source_has_no_forbidden_references(self):
        import pathlib

        package = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src"
            / "xssharden"
            / "evaluation"
        )
        sources = [path.read_text(encoding="utf-8").lower()
                   for path in package.glob("*.py")]
        assert sources, "evaluation package has no modules"
        banned = (
            "playwright",
            "xssharden.validation",
            "xssharden.validator",
            "from xssharden import validation",
            "validate_payload",
            "validate_variants",
            "browser",
            "chromium",
            "selenium",
            "subprocess",
            "os.system",
            "urllib",
            "requests.",
            ".fit(",
            "calibrat",
            "predict_proba",
            "sklearn",
        )
        for source in sources:
            for token in banned:
                assert token not in source, (
                    f"evaluation module must not reference {token!r}")

    def test_payloads_never_executed(self):
        import pathlib

        package = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src"
            / "xssharden"
            / "evaluation"
        )
        sources = [path.read_text(encoding="utf-8").lower()
                   for path in package.glob("*.py")]
        for source in sources:
            for token in ("eval(", "exec(", "os.popen", "__import__"):
                assert token not in source


class TestZeroDenominators:
    def _single_arm(self, detector):
        return {"baseline": detector, "naive": detector,
                "random_valid": detector, "selective": detector}

    def test_all_benign_clean_gives_safe_zero_precision(self):
        from xssharden.evaluation import evaluate_arms

        clean = [dict(record, label=0) for record in _clean_records()]
        table = {record["payload"]: (0.1, 0) for record in clean}
        table.update({
            "<svg onload=alert(1)>": (0.1, 0),
            "<b>not here</b>": (0.9, 1),
            "<div>broken": (0.05, 0),
            "just text": (0.1, 0),
        })
        result = evaluate_arms(
            self._single_arm(PayloadFakeDetector(table)),
            clean, _adv_records())
        metrics = result.arms["baseline"].clean_metrics
        assert metrics.positives == 0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.fpr == 0.0
        assert metrics.accuracy == 1.0

    def test_no_valid_malicious_adv_gives_zero_valid_rate(self):
        from xssharden.evaluation import evaluate_arms

        adv = [dict(record, valid=False) if record["label"] == 1 else record
               for record in _adv_records()]
        result = evaluate_arms(
            self._single_arm(PayloadFakeDetector(_mixed_table())),
            _clean_records(), adv)
        metrics = result.arms["baseline"].adversarial_metrics
        assert metrics.valid_malicious_count == 0
        assert metrics.valid_malicious_evasion_rate == 0.0

    def test_empty_inputs_rejected(self):
        from xssharden.evaluation import evaluate_arms

        arms = _four_detectors()
        with pytest.raises(ValueError):
            evaluate_arms(arms, [], _adv_records())
        with pytest.raises(ValueError):
            evaluate_arms(arms, _clean_records(), [])


class TestProvenanceDeterminismAndImmutability:
    def test_extra_keys_preserved_in_scored_records(self):
        from xssharden.evaluation import evaluate_arms

        result = evaluate_arms(_four_detectors(), _clean_records(), _adv_records())
        clean_scored = result.arms["naive"].clean_records
        assert clean_scored[0]["note"] == "keep-me"
        assert clean_scored[0]["sample_id"] == "c1"
        adv_scored = result.arms["naive"].adversarial_records
        assert adv_scored[0]["probe"] == "p1"
        assert adv_scored[0]["variant_id"] == "a1"
        assert "detector_score" in clean_scored[0]
        assert "predicted_label" in adv_scored[0]

    def test_inputs_are_not_mutated(self):
        from xssharden.evaluation import evaluate_arms

        arms = _four_detectors()
        clean = _clean_records()
        adv = _adv_records()
        clean_before = copy.deepcopy(clean)
        adv_before = copy.deepcopy(adv)
        arms_before = list(arms)
        evaluate_arms(arms, clean, adv)
        assert clean == clean_before
        assert adv == adv_before
        assert list(arms) == arms_before

    def test_repeated_runs_are_identical(self):
        from xssharden.evaluation import evaluate_arms

        first = evaluate_arms(
            _four_detectors(), _clean_records(), _adv_records()).to_dict()
        second = evaluate_arms(
            _four_detectors(), _clean_records(), _adv_records()).to_dict()
        assert first == second


class TestInputValidation:
    def test_missing_arm_rejected(self):
        from xssharden.evaluation import evaluate_arms

        arms = _four_detectors()
        del arms["selective"]
        with pytest.raises(ValueError):
            evaluate_arms(arms, _clean_records(), _adv_records())

    def test_unexpected_arm_rejected(self):
        from xssharden.evaluation import evaluate_arms

        arms = _four_detectors()
        arms["extra"] = arms["baseline"]
        with pytest.raises(ValueError):
            evaluate_arms(arms, _clean_records(), _adv_records())

    def test_detector_without_scoring_api_rejected(self):
        from xssharden.evaluation import evaluate_arms

        arms = _four_detectors()
        arms["naive"] = object()
        with pytest.raises(TypeError):
            evaluate_arms(arms, _clean_records(), _adv_records())

    def test_malformed_records_rejected(self):
        from xssharden.evaluation import evaluate_arms

        arms = _four_detectors()
        bad_clean = _clean_records()
        bad_clean[0] = {"label": 0, "split": "clean-test"}
        with pytest.raises(ValueError):
            evaluate_arms(arms, bad_clean, _adv_records())
        bad_label = _clean_records()
        bad_label[0] = dict(bad_label[0], label=2)
        with pytest.raises(ValueError):
            evaluate_arms(arms, bad_label, _adv_records())
        assert _clean_records()[0]["payload"].strip() != ""

    def test_detector_output_problems_surface(self):
        from xssharden.evaluation import evaluate_arms

        class BadProba(PayloadFakeDetector):
            def predict_proba(self, records):
                import numpy as np

                return np.asarray([[0.5, 99.0]] * len(records))

        arms = _four_detectors()
        arms["naive"] = BadProba(_mixed_table())
        with pytest.raises(ValueError):
            evaluate_arms(arms, _clean_records(), _adv_records())

    def test_none_inputs_rejected(self):
        from xssharden.evaluation import evaluate_arms

        arms = _four_detectors()
        with pytest.raises(TypeError):
            evaluate_arms(arms, None, _adv_records())
        with pytest.raises(TypeError):
            evaluate_arms(None, _clean_records(), _adv_records())

    def test_hardening_result_wrapper(self):
        from xssharden.evaluation import evaluate_arms, evaluate_hardening_result
        from xssharden.hardening.arms import HardeningResult

        arms = _four_detectors()
        hardening = HardeningResult(
            arm_records={name: [] for name in arms},
            arm_detectors=dict(arms),
            added_counts={name: 0 for name in arms},
            arm_sizes={name: 0 for name in arms},
        )
        wrapped = evaluate_hardening_result(
            hardening, _clean_records(), _adv_records())
        direct = evaluate_arms(arms, _clean_records(), _adv_records())
        assert wrapped.to_dict() == direct.to_dict()
