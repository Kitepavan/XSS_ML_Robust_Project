"""Focused tests for the detector-attack/evasion measurement module (TDD).

The evasion module scores already-validated variant records with an
already-fitted detector (``predict_proba``/``predict`` only — never fit or
calibrate), preserves every input key, flags evasions (only possible for
``label == 1`` predicted as ``0``), and reports aggregate metrics with
explicit valid-only versus raw denominators.

Tests use fake detectors only: no browser, no network, no model fitting.
"""

from __future__ import annotations

from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_UNSET: Any = object()


def _record(
    index: int,
    *,
    label: int = 1,
    valid: bool | None = True,
    split: str | None = "adv_dev",
    payload: Any = _UNSET,
    extra: dict | None = None,
) -> dict:
    """Build one validated-variant-style record with full provenance."""
    if payload is _UNSET:
        payload = f"<svg onload=alert({index})>"
    record: dict = {
        "variant_id": f"V{index:04d}",
        "seed_id": "S001",
        "payload": payload,
        "mutation_category": "encoding",
        "context_target": "reflected_html",
        "source": "adv_dev",
        "generator": "programmatic",
        "label": label,
    }
    if valid is not None:
        record["valid"] = valid
    if split is not None:
        record["split"] = split
    else:
        record.pop("split", None)
    if extra:
        record.update(extra)
    return record


class FakeDetector:
    """Minimal already-fitted detector stand-in.

    Returns canned probability/prediction rows in input order. ``fit`` and
    ``calibrate_threshold`` raise so any fitting/calibration attempt fails
    loudly instead of silently mutating state.
    """

    def __init__(self, proba, preds, threshold: float = 0.5) -> None:
        import numpy as np

        self._proba = np.asarray(proba, dtype=float)
        self._preds = np.asarray(preds)
        self.threshold_ = float(threshold)

    def predict_proba(self, inputs):
        # No length assertion here: shape mismatches must surface as
        # ValueError from the module under test, not AssertionError.
        return self._proba

    def predict(self, inputs):
        return self._preds

    def fit(self, *args, **kwargs):  # pragma: no cover - guard only
        raise AssertionError("evasion measurement must never fit the detector")

    def calibrate_threshold(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("evasion measurement must never calibrate")


def _proba(scores) -> list[list[float]]:
    """Build (n, 2) probability rows from malicious-class scores."""
    return [[1.0 - s, s] for s in scores]


# ===================================================================
# Section 1 - per-record scoring with a fake detector
# ===================================================================


class TestPerRecordScoring:
    def test_scored_records_preserve_every_input_key(self):
        from xssharden.attack import evaluate_variants

        record = _record(
            0,
            label=1,
            valid=True,
            extra={"custom_note": "keep-me", "attack_category": "xss"},
        )
        before = dict(record)
        detector = FakeDetector(_proba([0.2]), [0])
        evaluation = evaluate_variants([record], detector)
        scored = evaluation.records[0]
        for key, value in before.items():
            assert scored[key] == value, f"input key {key!r} was not preserved"
        assert scored["detector_score"] == pytest.approx(0.2)
        assert scored["probability"] == pytest.approx(0.2)
        assert scored["predicted_label"] == 0
        assert scored["detector_threshold"] == pytest.approx(0.5)
        assert scored["evaded"] is True

    def test_evasion_only_for_malicious_predicted_benign(self):
        from xssharden.attack import evaluate_variants

        records = [
            _record(0, label=1, valid=True),  # pred 0 -> evasion
            _record(1, label=1, valid=True),  # pred 1 -> no evasion
            _record(2, label=0, valid=True),  # pred 1 -> FP, not an evasion
            _record(3, label=0, valid=True),  # pred 0 -> correct benign
        ]
        detector = FakeDetector(_proba([0.1, 0.9, 0.8, 0.2]), [0, 1, 1, 0])
        scored = evaluate_variants(records, detector).records
        assert [row["evaded"] for row in scored] == [True, False, False, False]
        assert all(isinstance(row["evaded"], bool) for row in scored)

    def test_inputs_are_not_mutated(self):
        import copy

        from xssharden.attack import evaluate_variants

        records = [_record(0, label=1), _record(1, label=0)]
        snapshot = copy.deepcopy(records)
        detector = FakeDetector(_proba([0.1, 0.9]), [0, 1])
        evaluate_variants(records, detector)
        assert records == snapshot

    def test_detector_threshold_defaults_to_detector_attr(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.65]), [1], threshold=0.7)
        scored = evaluate_variants([_record(0)], detector).records
        assert scored[0]["detector_threshold"] == pytest.approx(0.7)

    def test_explicit_threshold_overrides_without_mutating_detector(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.2]), [1], threshold=0.5)
        scored = evaluate_variants(
            [_record(0)], detector, threshold=0.9
        ).records
        assert scored[0]["detector_threshold"] == pytest.approx(0.9)
        assert detector.threshold_ == pytest.approx(0.5)


# ===================================================================
# Section 2 - aggregate metrics: valid-only versus raw denominators
# ===================================================================


class TestAggregateMetrics:
    def _mixed_records(self) -> list[dict]:
        return [
            _record(0, label=1, valid=True),  # evasion, valid
            _record(1, label=1, valid=True),  # detected, valid
            _record(2, label=1, valid=False),  # evasion, invalid
            _record(3, label=1, valid=None),  # evasion, no validity flag
            _record(4, label=0, valid=True),  # benign correct
            _record(5, label=0, valid=False),  # benign FP
        ]

    def test_metrics_use_explicit_denominators(self):
        from xssharden.attack import evaluate_variants

        records = self._mixed_records()
        detector = FakeDetector(
            _proba([0.1, 0.9, 0.2, 0.3, 0.2, 0.8]), [0, 1, 0, 0, 0, 1]
        )
        metrics = evaluate_variants(records, detector).metrics
        assert metrics.total_records == 6
        assert metrics.valid_records == 3
        assert metrics.malicious_count == 4
        assert metrics.evasion_count == 3
        assert metrics.evasion_rate == pytest.approx(3 / 4)
        assert metrics.valid_malicious_count == 2
        assert metrics.valid_evasion_count == 1
        assert metrics.valid_malicious_evasion_rate == pytest.approx(1 / 2)
        assert metrics.raw_evasion_rate == pytest.approx(3 / 6)

    def test_invalid_evasions_never_count_as_valid_attacks(self):
        from xssharden.attack import evaluate_variants

        records = [
            _record(0, label=1, valid=False),
            _record(1, label=1, valid=False),
        ]
        detector = FakeDetector(_proba([0.1, 0.2]), [0, 0])
        metrics = evaluate_variants(records, detector).metrics
        assert metrics.evasion_count == 2
        assert metrics.evasion_rate == pytest.approx(1.0)
        assert metrics.valid_records == 0
        assert metrics.valid_malicious_count == 0
        assert metrics.valid_evasion_count == 0
        assert metrics.valid_malicious_evasion_rate == pytest.approx(0.0)

    def test_zero_malicious_gives_zero_rates_not_nan(self):
        import math

        from xssharden.attack import evaluate_variants

        records = [_record(0, label=0, valid=True), _record(1, label=0)]
        detector = FakeDetector(_proba([0.2, 0.3]), [0, 0])
        metrics = evaluate_variants(records, detector).metrics
        assert metrics.malicious_count == 0
        assert metrics.evasion_count == 0
        for rate in (
            metrics.evasion_rate,
            metrics.valid_malicious_evasion_rate,
            metrics.raw_evasion_rate,
        ):
            assert rate == 0.0
            assert not math.isnan(rate)

    def test_metrics_serialize_to_dict(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.1]), [0])
        evaluation = evaluate_variants([_record(0, label=1)], detector)
        as_dict = evaluation.metrics.to_dict()
        assert as_dict["total_records"] == 1
        assert as_dict["evasion_count"] == 1
        assert "valid_malicious_evasion_rate" in as_dict
        assert "raw_evasion_rate" in as_dict
        assert evaluation.to_dict()["metrics"] == as_dict


# ===================================================================
# Section 3 - leakage safety: split guards
# ===================================================================


class TestSplitGuards:
    @pytest.mark.parametrize("split", ["validation", "clean-test", "test"])
    def test_held_out_splits_rejected_by_default(self, split: str):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.1]), [0])
        with pytest.raises(ValueError, match="split"):
            evaluate_variants([_record(0, split=split)], detector)

    def test_named_adversarial_split_allowed_explicitly(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.1]), [0])
        evaluation = evaluate_variants(
            [_record(0, split="test")], detector, allowed_splits=["test"]
        )
        assert evaluation.records[0]["evaded"] is True

    def test_allowlist_still_rejects_unlisted_held_out_split(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.1]), [0])
        with pytest.raises(ValueError, match="split"):
            evaluate_variants(
                [_record(0, split="validation")],
                detector,
                allowed_splits=["test"],
            )

    def test_records_without_split_are_accepted(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.1]), [0])
        evaluation = evaluate_variants([_record(0, split=None)], detector)
        assert evaluation.metrics.total_records == 1

    def test_adv_dev_accepted_by_default(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.9]), [1])
        evaluation = evaluate_variants([_record(0, split="adv_dev")], detector)
        assert evaluation.records[0]["evaded"] is False


# ===================================================================
# Section 4 - error cases
# ===================================================================


class TestErrorCases:
    def test_empty_inputs_rejected(self):
        from xssharden.attack import evaluate_variants

        with pytest.raises(ValueError, match="non-empty"):
            evaluate_variants([], FakeDetector(_proba([0.1]), [0]))

    def test_non_iterable_or_non_dict_inputs_rejected(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.1]), [0])
        for bad in (None, {"payload": "x"}, "<script>alert(1)</script>"):
            with pytest.raises(TypeError, match="records"):
                evaluate_variants(bad, detector)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="dict"):
            evaluate_variants(["<script>alert(1)</script>"], detector)  # type: ignore[list-item]

    @pytest.mark.parametrize(
        "payload", [None, 123, b"<script>", "", "   "],
    )
    def test_malformed_payloads_rejected(self, payload):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.1]), [0])
        with pytest.raises((TypeError, ValueError), match="[Pp]ayload"):
            evaluate_variants([_record(0, payload=payload)], detector)  # type: ignore[arg-type]

    def test_missing_payload_rejected(self):
        from xssharden.attack import evaluate_variants

        record = _record(0)
        del record["payload"]
        with pytest.raises(ValueError, match="[Pp]ayload"):
            evaluate_variants([record], FakeDetector(_proba([0.1]), [0]))

    @pytest.mark.parametrize("label", [True, False, 2, -1, "1", 1.0, None])
    def test_non_binary_labels_rejected(self, label):
        from xssharden.attack import evaluate_variants

        record = _record(0)
        record["label"] = label
        with pytest.raises(ValueError, match="[Ll]abel"):
            evaluate_variants([record], FakeDetector(_proba([0.1]), [0]))

    def test_missing_label_rejected(self):
        from xssharden.attack import evaluate_variants

        record = _record(0)
        del record["label"]
        with pytest.raises(ValueError, match="[Ll]abel"):
            evaluate_variants([record], FakeDetector(_proba([0.1]), [0]))

    def test_detector_without_proba_or_predict_rejected(self):
        from xssharden.attack import evaluate_variants

        class NoProba:
            def predict(self, inputs):
                return [0 for _ in inputs]

        class NoPredict:
            def predict_proba(self, inputs):
                import numpy as np

                return np.zeros((len(inputs), 2))

        with pytest.raises(TypeError, match="predict_proba"):
            evaluate_variants([_record(0)], NoProba())
        with pytest.raises(TypeError, match="predict"):
            evaluate_variants([_record(0)], NoPredict())

    def test_proba_shape_mismatch_rejected(self):
        import numpy as np

        from xssharden.attack import evaluate_variants

        records = [_record(0), _record(1)]
        for bad_proba in (
            np.zeros((2, 3)),
            np.zeros(2),
            np.zeros((3, 2)),
            np.zeros((1, 2)),
        ):
            with pytest.raises(ValueError, match="[Ss]hape"):
                evaluate_variants(
                    records, FakeDetector(bad_proba, [0, 0])
                )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_probabilities_rejected(self, bad: float):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector([[0.5, bad]], [0])
        with pytest.raises(ValueError, match="[Pp]robabilit"):
            evaluate_variants([_record(0)], detector)

    def test_out_of_range_probabilities_rejected(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector([[0.2, 1.5]], [0])
        with pytest.raises(ValueError, match="[Pp]robabilit"):
            evaluate_variants([_record(0)], detector)

    def test_non_binary_predictions_rejected(self):
        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.1]), [2])
        with pytest.raises(ValueError, match="[Pp]redict"):
            evaluate_variants([_record(0)], detector)

    def test_prediction_length_mismatch_rejected(self):
        from xssharden.attack import evaluate_variants

        records = [_record(0), _record(1)]
        detector = FakeDetector(_proba([0.1, 0.2]), [0])
        with pytest.raises(ValueError, match="[Ss]hape|[Pp]redict|[Ll]ength"):
            evaluate_variants(records, detector)

    @pytest.mark.parametrize("threshold", [float("nan"), -0.1, 1.5, "high"])
    def test_bad_threshold_rejected(self, threshold):
        from xssharden.attack import evaluate_variants

        with pytest.raises((TypeError, ValueError), match="[Tt]hreshold"):
            evaluate_variants(
                [_record(0)],
                FakeDetector(_proba([0.1]), [0]),
                threshold=threshold,  # type: ignore[arg-type]
            )


# ===================================================================
# Section 5 - no detector mutation, no browser involvement
# ===================================================================


class TestNoMutationNoBrowser:
    def test_detector_state_unchanged(self):
        import copy

        import numpy as np

        from xssharden.attack import evaluate_variants

        detector = FakeDetector(_proba([0.1, 0.9]), [0, 1], threshold=0.62)
        before = copy.deepcopy(detector.__dict__)
        evaluate_variants([_record(0, label=1), _record(1, label=0)], detector)
        after = detector.__dict__
        assert set(after) == set(before)
        assert after["threshold_"] == before["threshold_"]
        np.testing.assert_array_equal(after["_proba"], before["_proba"])
        np.testing.assert_array_equal(after["_preds"], before["_preds"])

    def test_detector_fit_and_calibrate_never_called(self):
        from xssharden.attack import evaluate_variants

        calls: list[str] = []

        class WatchedDetector(FakeDetector):
            def fit(self, *args, **kwargs):
                calls.append("fit")
                return self

            def calibrate_threshold(self, *args, **kwargs):
                calls.append("calibrate_threshold")
                return 0.5

        detector = WatchedDetector(_proba([0.1]), [0])
        evaluate_variants([_record(0)], detector)
        assert calls == []

    def test_module_never_touches_browser_validation(self):
        import pathlib

        package = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src"
            / "xssharden"
            / "attack"
        )
        sources = [path.read_text(encoding="utf-8").lower() for path in package.glob("*.py")]
        assert sources, "attack package has no modules"
        banned = (
            "playwright",
            "xssharden.validation",
            "xssharden.validator",
            "from xssharden import validation",
            "validate_payload",
            "validate_variants",
            "browser",
            "chromium",
            "subprocess",
            "os.system",
            "urllib",
            "requests.",
        )
        for source in sources:
            for token in banned:
                assert token not in source, f"attack module must not reference {token!r}"

    def test_payloads_never_executed(self):
        import pathlib

        package = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src"
            / "xssharden"
            / "attack"
        )
        sources = [path.read_text(encoding="utf-8").lower() for path in package.glob("*.py")]
        for source in sources:
            for token in ("eval(", "exec(", "os.popen", "__import__"):
                assert token not in source
