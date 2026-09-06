"""Focused regression tests for four-arm hardening orchestration."""

from __future__ import annotations

import copy

import pytest


class FakeDetector:
    def __init__(self, scores=None):
        self.scores = list(scores or [])
        self.threshold_ = 0.5
        self.fit_calls = 0
        self.calibration_calls = 0

    def fit(self, records):
        self.fit_calls += 1
        self.fitted_records = [dict(record) for record in records]
        return self

    def predict_proba(self, records):
        import numpy as np

        scores = self.scores or [0.2 if record["label"] == 0 else 0.8 for record in records]
        if len(scores) != len(records):
            scores = [scores[index % len(scores)] for index in range(len(records))]
        return np.asarray([[1.0 - score, score] for score in scores])

    def predict(self, records):
        return (self.predict_proba(records)[:, 1] >= self.threshold_).astype(int)

    def calibrate_threshold(self, records):
        self.calibration_calls += 1
        assert all(record["split"] == "validation" for record in records)
        return self.threshold_


def _train():
    return [
        {"sample_id": "b", "payload": "plain text", "label": 0, "source": "s", "attack_category": "benign", "split": "train"},
        {"sample_id": "x", "payload": "<script>alert(1)</script>", "label": 1, "source": "s", "attack_category": "xss", "split": "train"},
    ]


def _variants():
    return [
        {"variant_id": "v1", "seed_id": "s", "payload": "<svg onload=alert(1)>", "label": 1, "valid": True, "mutation_category": "encoding", "split": "adv_dev"},
        {"variant_id": "v2", "seed_id": "s", "payload": "<img src=x onerror=alert(1)>", "label": 1, "valid": False, "mutation_category": "case_variation", "split": "adv_dev"},
        {"variant_id": "v3", "seed_id": "s", "payload": "benign", "label": 0, "valid": True, "mutation_category": "encoding", "split": "adv_dev"},
    ]


def test_four_fresh_arms_and_validity_gate():
    from xssharden.hardening import run_hardening_arms

    created = []

    def factory():
        detector = FakeDetector()
        created.append(detector)
        return detector

    result = run_hardening_arms(_train(), _variants(), factory, budget=1)
    assert set(result.arm_records) == {"baseline", "naive", "random_valid", "selective"}
    assert len(created) == 4
    assert result.added_counts["baseline"] == 0
    assert result.added_counts["naive"] == 1
    assert result.added_counts["random_valid"] == 1
    assert result.added_counts["selective"] == 1
    assert all(record["split"] == "train" for record in result.arm_records["naive"][2:])


def test_budgeted_selection_is_deterministic_and_inputs_are_unchanged():
    from xssharden.hardening import run_hardening_arms

    train = _train()
    variants = _variants()
    train_before = copy.deepcopy(train)
    variants_before = copy.deepcopy(variants)
    first = run_hardening_arms(train, variants, FakeDetector, budget=1, seed=7)
    second = run_hardening_arms(train, variants, FakeDetector, budget=1, seed=7)
    assert [r["payload"] for r in first.arm_records["selective"]] == [r["payload"] for r in second.arm_records["selective"]]
    assert train == train_before
    assert variants == variants_before


def test_validation_records_must_be_validation_split():
    from xssharden.hardening import run_hardening_arms

    bad = [{"payload": "x", "label": 0, "split": "clean-test"}]
    with pytest.raises(ValueError, match="split='validation'"):
        run_hardening_arms(_train(), _variants(), FakeDetector, budget=1, validation_records=bad)


def test_calibration_is_called_for_each_arm_on_validation_records():
    from xssharden.hardening import run_hardening_arms

    made = []

    def factory():
        detector = FakeDetector()
        made.append(detector)
        return detector

    validation = [{"payload": "validation", "label": 0, "split": "validation"}]
    run_hardening_arms(_train(), _variants(), factory, budget=1, validation_records=validation)
    assert [detector.calibration_calls for detector in made] == [1, 1, 1, 1]
