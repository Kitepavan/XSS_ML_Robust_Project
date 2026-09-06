"""Focused tests for the validity-gated variant selection engine (TDD).

The selection engine picks a budgeted subset of already-scored variant
records for hardening. Validity is a hard gate (only ``valid is True``
malicious records are eligible), held-out splits are never selected,
exact payload duplicates are removed, and ranking is either
detector-impact (lowest malicious probability first) or a seeded
random-valid control. Payloads stay inert strings throughout.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest


_UNSET: Any = object()


def _record(
    index: int,
    *,
    label: int = 1,
    valid: Any = True,
    split: Any = "adv_dev",
    payload: Any = _UNSET,
    score: Any = _UNSET,
    category: str = "encoding",
    variant_id: Any = _UNSET,
    include_score: bool = True,
    include_valid: bool = True,
    include_split: bool = True,
) -> dict:
    """Build one scored-variant-style record with full provenance."""
    if payload is _UNSET:
        payload = f"<svg onload=alert({index})>"
    if variant_id is _UNSET:
        variant_id = f"V{index:04d}"
    record: dict = {
        "variant_id": variant_id,
        "seed_id": "S001",
        "payload": payload,
        "mutation_category": category,
        "context_target": "reflected_html",
        "source": "adv_dev",
        "generator": "programmatic",
        "label": label,
    }
    if include_valid:
        record["valid"] = valid
    if include_split:
        if split is None:
            pass  # explicitly no split key
        else:
            record["split"] = split
    if include_score and score is not _UNSET:
        record["detector_score"] = score
        record["probability"] = score
    return record


def _eligible_pool(count: int = 6, start: int = 0) -> list[dict]:
    """Build ``count`` eligible records with ascending scores."""
    return [
        _record(start + index, score=0.05 * (index + 1))
        for index in range(count)
    ]


class TestValidityHardGate:
    def test_invalid_records_never_selected(self):
        from xssharden.selection import select_variants

        records = [
            _record(0, valid=False, score=0.01),  # highest impact, still ineligible
            _record(1, valid=True, score=0.9),
        ]
        result = select_variants(records, budget=2, strategy="impact")
        assert [row["variant_id"] for row in result.records] == ["V0001"]
        assert result.eligible_count == 1

    def test_missing_valid_flag_never_selected(self):
        from xssharden.selection import select_variants

        records = [
            _record(0, score=0.01, include_valid=False),
            _record(1, valid=True, score=0.9),
        ]
        result = select_variants(records, budget=2, strategy="impact")
        assert [row["variant_id"] for row in result.records] == ["V0001"]

    def test_truthy_non_true_valid_never_selected(self):
        from xssharden.selection import select_variants

        records = [
            _record(0, valid=1, score=0.01),
            _record(1, valid="yes", score=0.02),
            _record(2, valid=True, score=0.9),
        ]
        result = select_variants(records, budget=3, strategy="impact")
        assert [row["variant_id"] for row in result.records] == ["V0002"]

    def test_benign_records_never_selected(self):
        from xssharden.selection import select_variants

        records = [
            _record(0, label=0, valid=True, score=0.01),
            _record(1, label=1, valid=True, score=0.9),
        ]
        result = select_variants(records, budget=2, strategy="impact")
        assert [row["variant_id"] for row in result.records] == ["V0001"]

    def test_filtered_counts_report_exclusion_reasons(self):
        from xssharden.selection import select_variants

        records = [
            _record(0, valid=False, score=0.1),
            _record(1, label=0, valid=True, score=0.1),
            _record(2, score=0.1, include_valid=False),
            _record(3, valid=True, score=0.4),
        ]
        result = select_variants(records, budget=10, strategy="impact")
        assert result.eligible_count == 1
        assert result.selected_count == 1
        assert sum(result.filtered_counts.values()) == 3
        assert set(result.filtered_counts) >= {"not_valid", "benign"}


class TestSplitGuards:
    @pytest.mark.parametrize("split", ["validation", "clean-test", "test"])
    def test_held_out_splits_rejected_by_default(self, split: str):
        from xssharden.selection import select_variants

        with pytest.raises(ValueError, match="held out"):
            select_variants(
                [_record(0, split=split, score=0.1)], budget=1, strategy="impact"
            )

    @pytest.mark.parametrize("split", ["validation", "clean-test", "test"])
    def test_held_out_splits_rejected_for_random_control(self, split: str):
        from xssharden.selection import select_variants

        with pytest.raises(ValueError, match="held out"):
            select_variants(
                [_record(0, split=split, score=0.1)],
                budget=1,
                strategy="random_valid",
            )

    def test_blocked_split_cannot_be_allow_listed(self):
        from xssharden.selection import select_variants

        with pytest.raises(ValueError, match="held-out"):
            select_variants(
                [_record(0, split="validation", score=0.1)],
                budget=1,
                strategy="impact",
                allowed_splits=("adv_dev", "validation"),
            )

    def test_non_allow_listed_non_blocked_split_excluded_not_selected(self):
        from xssharden.selection import select_variants

        records = [
            _record(0, split="train", score=0.01),
            _record(1, split="adv_dev", score=0.9),
        ]
        result = select_variants(records, budget=2, strategy="impact")
        assert [row["variant_id"] for row in result.records] == ["V0001"]
        assert result.filtered_counts.get("split_not_allowed") == 1

    def test_records_without_split_are_accepted(self):
        from xssharden.selection import select_variants

        result = select_variants(
            [_record(0, split=None, include_split=True, score=0.2)],
            budget=1,
            strategy="impact",
        )
        assert result.selected_count == 1

    def test_custom_allowed_split_accepted(self):
        from xssharden.selection import select_variants

        result = select_variants(
            [_record(0, split="adv_pool", score=0.2)],
            budget=1,
            strategy="impact",
            allowed_splits=("adv_dev", "adv_pool"),
        )
        assert result.selected_count == 1


class TestDeduplication:
    def test_exact_duplicate_payloads_deduplicated_within_candidates(self):
        from xssharden.selection import select_variants

        records = [
            _record(0, payload="<svg onload=alert(1)>", score=0.5),
            _record(1, payload="<svg onload=alert(1)>", score=0.01),
            _record(2, payload="<svg onload=alert(2)>", score=0.9),
        ]
        result = select_variants(records, budget=10, strategy="impact")
        assert [row["variant_id"] for row in result.records] == ["V0000", "V0002"]
        assert result.filtered_counts.get("duplicate_within_candidates") == 1

    def test_duplicates_removed_against_training_records(self):
        from xssharden.selection import select_variants

        training = [_record(100, payload="<svg onload=alert(1)>", score=0.9)]
        records = [
            _record(0, payload="<svg onload=alert(1)>", score=0.01),
            _record(1, payload="<svg onload=alert(9)>", score=0.9),
        ]
        result = select_variants(
            records, budget=10, strategy="impact", training_records=training
        )
        assert [row["variant_id"] for row in result.records] == ["V0001"]
        assert result.filtered_counts.get("duplicate_of_training") == 1

    def test_random_control_uses_same_deduplicated_pool(self):
        from xssharden.selection import select_variants

        training = [{"payload": "<svg onload=alert(1)>", "label": 1}]
        records = [
            _record(0, payload="<svg onload=alert(1)>"),
            _record(1, payload="<svg onload=alert(1)>"),
            _record(2, payload="<svg onload=alert(2)>"),
        ]
        result = select_variants(
            records,
            budget=10,
            strategy="random_valid",
            training_records=training,
            seed=7,
        )
        assert [row["variant_id"] for row in result.records] == ["V0002"]
        assert result.eligible_count == 1


class TestImpactOrdering:
    def test_impact_ranks_lowest_malicious_probability_first(self):
        from xssharden.selection import select_variants

        records = [
            _record(0, score=0.9),
            _record(1, score=0.1),
            _record(2, score=0.5),
        ]
        result = select_variants(records, budget=3, strategy="impact")
        assert [row["detector_score"] for row in result.records] == [0.1, 0.5, 0.9]

    def test_impact_tie_breaking_is_deterministic(self):
        from xssharden.selection import select_variants

        forward = [_record(index, score=0.3) for index in range(5)]
        backward = list(reversed([dict(row) for row in forward]))
        first = select_variants(forward, budget=5, strategy="impact")
        second = select_variants(backward, budget=5, strategy="impact")
        # Same candidate set: identical score multiset selected either way.
        assert [row["variant_id"] for row in first.records] == [
            f"V{index:04d}" for index in range(5)
        ]
        assert sorted(row["variant_id"] for row in second.records) == sorted(
            row["variant_id"] for row in first.records
        )
        again = select_variants(forward, budget=5, strategy="impact")
        assert [row["variant_id"] for row in again.records] == [
            row["variant_id"] for row in first.records
        ]

    def test_probability_key_accepted_as_score_alias(self):
        from xssharden.selection import select_variants

        record = _record(0, score=_UNSET, include_score=False)
        record["probability"] = 0.25
        result = select_variants([record], budget=1, strategy="impact")
        assert result.selected_count == 1

    def test_budget_selects_top_impact_prefix(self):
        from xssharden.selection import select_variants

        result = select_variants(_eligible_pool(6), budget=2, strategy="impact")
        assert [row["detector_score"] for row in result.records] == [0.05, 0.10]


class TestRandomValidControl:
    def test_random_selection_is_reproducible_for_same_seed(self):
        from xssharden.selection import select_variants

        records = _eligible_pool(12)
        first = select_variants(
            records, budget=4, strategy="random_valid", seed=123
        )
        second = select_variants(
            records, budget=4, strategy="random_valid", seed=123
        )
        assert [row["variant_id"] for row in first.records] == [
            row["variant_id"] for row in second.records
        ]

    def test_random_selection_differs_across_seeds(self):
        from xssharden.selection import select_variants

        records = _eligible_pool(12)
        outcomes = {
            tuple(
                row["variant_id"]
                for row in select_variants(
                    records, budget=4, strategy="random_valid", seed=seed
                ).records
            )
            for seed in range(6)
        }
        assert len(outcomes) > 1

    def test_random_selection_needs_no_detector_scores(self):
        from xssharden.selection import select_variants

        records = [_record(index, score=_UNSET, include_score=False) for index in range(4)]
        result = select_variants(records, budget=2, strategy="random_valid", seed=42)
        assert result.selected_count == 2
        assert result.eligible_count == 4

    def test_seed_recorded_in_audit(self):
        from xssharden.selection import select_variants

        result = select_variants(
            _eligible_pool(4), budget=2, strategy="random_valid", seed=99
        )
        assert result.seed == 99
        assert result.to_dict()["seed"] == 99


class TestCategoryCaps:
    def test_max_per_category_enforced_for_impact(self):
        from xssharden.selection import select_variants

        records = [
            _record(0, score=0.01, category="encoding"),
            _record(1, score=0.02, category="encoding"),
            _record(2, score=0.03, category="encoding"),
            _record(3, score=0.50, category="whitespace_comment"),
        ]
        result = select_variants(
            records, budget=3, strategy="impact", max_per_category=1
        )
        assert [row["variant_id"] for row in result.records] == ["V0000", "V0003"]
        assert result.to_dict()["per_category_counts"] == {
            "encoding": 1,
            "whitespace_comment": 1,
        }

    def test_max_per_category_enforced_for_random_control(self):
        from xssharden.selection import select_variants

        records = [
            _record(index, category="encoding" if index < 6 else "case_variation")
            for index in range(9)
        ]
        result = select_variants(
            records,
            budget=6,
            strategy="random_valid",
            seed=42,
            max_per_category=2,
        )
        counts: dict[str, int] = {}
        for row in result.records:
            counts[row["mutation_category"]] = counts.get(row["mutation_category"], 0) + 1
        assert all(count <= 2 for count in counts.values())

    def test_invalid_category_cap_rejected(self):
        from xssharden.selection import select_variants

        with pytest.raises(ValueError, match="max_per_category"):
            select_variants(
                _eligible_pool(3),
                budget=2,
                strategy="impact",
                max_per_category=0,
            )


class TestBudgetAndAudit:
    def test_budget_shortfall_selects_all_eligible(self):
        from xssharden.selection import select_variants

        result = select_variants(_eligible_pool(3), budget=10, strategy="impact")
        assert result.selected_count == 3
        assert result.eligible_count == 3
        assert result.to_dict()["shortfall"] is True
        assert result.to_dict()["requested_budget"] == 10

    def test_no_shortfall_when_budget_met(self):
        from xssharden.selection import select_variants

        result = select_variants(_eligible_pool(4), budget=2, strategy="impact")
        assert result.to_dict()["shortfall"] is False

    @pytest.mark.parametrize("budget", [0, -1, 2.5, "3", True, None])
    def test_non_positive_or_non_integer_budget_rejected(self, budget: Any):
        from xssharden.selection import select_variants

        with pytest.raises((TypeError, ValueError), match="budget"):
            select_variants(_eligible_pool(3), budget=budget, strategy="impact")

    def test_unknown_strategy_rejected(self):
        from xssharden.selection import select_variants

        with pytest.raises(ValueError, match="strategy"):
            select_variants(_eligible_pool(3), budget=1, strategy="topk")

    def test_audit_metadata_complete_and_serializable(self):
        from xssharden.selection import select_variants

        result = select_variants(_eligible_pool(4), budget=2, strategy="impact")
        assert result.strategy == "impact"
        assert result.requested_budget == 2
        assert result.selected_count == 2
        assert result.eligible_count == 4
        assert result.seed == 42
        payload = result.to_dict()
        assert payload["strategy"] == "impact"
        assert payload["requested_budget"] == 2
        assert payload["selected_count"] == 2
        assert payload["eligible_count"] == 4
        assert payload["seed"] == 42
        assert payload["filtered_counts"] == result.filtered_counts
        assert payload["per_category_counts"] == result.per_category_counts
        assert len(payload["records"]) == 2


class TestMalformedScores:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5])
    def test_malformed_impact_scores_rejected(self, bad: float):
        from xssharden.selection import select_variants

        with pytest.raises(ValueError, match="[Ss]core|probability|range|finite"):
            select_variants([_record(0, score=bad)], budget=1, strategy="impact")

    def test_missing_score_rejected_for_impact(self):
        from xssharden.selection import select_variants

        with pytest.raises(ValueError, match="[Ss]core"):
            select_variants(
                [_record(0, score=_UNSET, include_score=False)],
                budget=1,
                strategy="impact",
            )

    def test_non_numeric_score_rejected_for_impact(self):
        from xssharden.selection import select_variants

        with pytest.raises((TypeError, ValueError), match="[Ss]core"):
            select_variants([_record(0, score="low")], budget=1, strategy="impact")

    def test_malformed_records_rejected(self):
        from xssharden.selection import select_variants

        with pytest.raises((TypeError, ValueError)):
            select_variants([{"label": 1, "valid": True}], budget=1)  # no payload
        with pytest.raises((TypeError, ValueError)):
            select_variants(
                [_record(0, label=7, score=0.1)], budget=1  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError):
            select_variants(None, budget=1)  # type: ignore[arg-type]


class TestImmutabilityAndProvenance:
    def test_inputs_are_not_mutated(self):
        from xssharden.selection import select_variants

        records = _eligible_pool(5)
        training = [_record(100, payload="<svg onload=alert(0)>", score=0.9)]
        snapshot = copy.deepcopy(records)
        training_snapshot = copy.deepcopy(training)
        select_variants(
            records,
            budget=3,
            strategy="impact",
            training_records=training,
            max_per_category=2,
        )
        assert records == snapshot
        assert training == training_snapshot

    def test_selected_records_preserve_every_input_key(self):
        from xssharden.selection import select_variants

        record = _record(0, score=0.2, category="case_variation")
        record["custom_provenance"] = {"annotator": "test"}
        result = select_variants([record], budget=1, strategy="impact")
        selected = result.records[0]
        for key, value in record.items():
            assert selected[key] == value
        assert selected is not record

    def test_training_records_never_selected_or_mutated(self):
        from xssharden.selection import select_variants

        training = _eligible_pool(2)
        records = _eligible_pool(2)
        snapshot = copy.deepcopy(training)
        select_variants(records, budget=2, strategy="impact", training_records=training)
        assert training == snapshot


class TestNoBrowserNoFitting:
    def test_module_never_touches_browser_validation_or_fitting(self):
        import pathlib

        package = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src"
            / "xssharden"
            / "selection"
        )
        sources = [path.read_text(encoding="utf-8").lower() for path in package.glob("*.py")]
        assert sources, "selection package has no modules"
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
            ".fit(",
            "calibrat",
            "predict_proba",
            "sklearn",
        )
        for source in sources:
            for token in banned:
                assert token not in source, f"selection module must not reference {token!r}"

    def test_payloads_never_executed(self):
        import pathlib

        package = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src"
            / "xssharden"
            / "selection"
        )
        sources = [path.read_text(encoding="utf-8").lower() for path in package.glob("*.py")]
        for source in sources:
            for token in ("eval(", "exec(", "os.popen", "__import__"):
                assert token not in source
