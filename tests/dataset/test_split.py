"""Tests for leakage-safe splitting and reproducibility (Task 3)."""

from __future__ import annotations

import pytest

from xssharden.dataset.split import split_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> dict:
    """Return a fully-formed record, allowing selective overrides."""
    base = {
        "sample_id": "S001",
        "payload": "<script>alert(1)</script>",
        "label": 1,
        "source": "github",
        "attack_category": "reflected_html",
        "split": "train",
    }
    base.update(overrides)
    return base


def _make_records(n: int, *, source: str = "github") -> list[dict]:
    """Return *n* records with sequential sample_ids."""
    return [
        _make_record(sample_id=f"S{i:04d}", source=source)
        for i in range(n)
    ]


def _make_multi_source_records(
    n: int, *, group_size: int = 5, prefix: str = "source"
) -> list[dict]:
    """Return *n* records spread across many small source groups.

    Each group holds at most ``group_size`` records so group-atomic
    splitting can still approximate the target ratios. Used wherever
    ratio or seed-variation assertions are intended.
    """
    records: list[dict] = []
    idx = 0
    group_idx = 0
    while idx < n:
        src_name = f"{prefix}_{group_idx}"
        for _ in range(min(group_size, n - idx)):
            records.append(
                _make_record(sample_id=f"{src_name}_S{idx:04d}", source=src_name)
            )
            idx += 1
        group_idx += 1
    return records


# ===================================================================
# Section 1 – Deterministic assignments
# ===================================================================

class TestDeterministicAssignments:
    """split_records must produce identical output for the same seed."""

    def test_same_seed_same_result(self):
        records = _make_multi_source_records(100, group_size=5)
        first = split_records(records, seed=42)
        second = split_records(records, seed=42)
        assert first == second

    def test_different_seed_different_result(self):
        records = _make_multi_source_records(100, group_size=5)
        first = split_records(records, seed=42)
        second = split_records(records, seed=99)
        # With high probability the split assignments differ.
        splits_first = [r["split"] for r in first]
        splits_second = [r["split"] for r in second]
        assert splits_first != splits_second

    def test_output_contains_all_original_records(self):
        records = _make_records(50)
        result = split_records(records, seed=7)
        result_ids = {r["sample_id"] for r in result}
        original_ids = {r["sample_id"] for r in records}
        assert result_ids == original_ids

    def test_split_field_is_set(self):
        """Every record must have its 'split' field overwritten."""
        records = _make_records(30)
        result = split_records(records, seed=1)
        valid_splits = {"train", "validation", "clean-test"}
        for r in result:
            assert r["split"] in valid_splits, f"Unexpected split value: {r['split']}"


# ===================================================================
# Section 2 – Disjoint split membership
# ===================================================================

class TestDisjointMembership:
    """No sample_id may appear in more than one split."""

    def test_no_overlap_between_train_and_validation(self):
        records = _make_multi_source_records(200, group_size=5)
        result = split_records(records, seed=42)
        train_ids = {r["sample_id"] for r in result if r["split"] == "train"}
        val_ids = {r["sample_id"] for r in result if r["split"] == "validation"}
        assert train_ids.isdisjoint(val_ids)

    def test_no_overlap_between_train_and_test(self):
        records = _make_multi_source_records(200, group_size=5)
        result = split_records(records, seed=42)
        train_ids = {r["sample_id"] for r in result if r["split"] == "train"}
        test_ids = {r["sample_id"] for r in result if r["split"] == "clean-test"}
        assert train_ids.isdisjoint(test_ids)

    def test_no_overlap_between_validation_and_test(self):
        records = _make_multi_source_records(200, group_size=5)
        result = split_records(records, seed=42)
        val_ids = {r["sample_id"] for r in result if r["split"] == "validation"}
        test_ids = {r["sample_id"] for r in result if r["split"] == "clean-test"}
        assert val_ids.isdisjoint(test_ids)

    def test_all_records_assigned_exactly_one_split(self):
        records = _make_records(100)
        result = split_records(records, seed=42)
        all_ids = [r["sample_id"] for r in result]
        assert len(all_ids) == len(set(all_ids)), "Duplicate sample_ids in output"


# ===================================================================
# Section 3 – Ratio validation
# ===================================================================

class TestRatioValidation:
    """Ratios must sum to 1.0 and each split must approximate the target."""

    def test_default_ratios_sum_to_one(self):
        records = _make_multi_source_records(1000, group_size=5)
        result = split_records(records, seed=42)
        counts = {}
        for r in result:
            counts[r["split"]] = counts.get(r["split"], 0) + 1
        total = sum(counts.values())
        assert total == 1000

    def test_default_ratios_approximately_correct(self):
        records = _make_multi_source_records(1000, group_size=5)
        result = split_records(records, seed=42)
        counts = {}
        for r in result:
            counts[r["split"]] = counts.get(r["split"], 0) + 1
        # Allow ±2% tolerance for rounding
        assert abs(counts.get("train", 0) / 1000 - 0.70) < 0.05
        assert abs(counts.get("validation", 0) / 1000 - 0.15) < 0.05
        assert abs(counts.get("clean-test", 0) / 1000 - 0.15) < 0.05

    def test_custom_ratios(self):
        records = _make_multi_source_records(1000, group_size=5)
        result = split_records(records, seed=42, ratios=(0.80, 0.10, 0.10))
        counts = {}
        for r in result:
            counts[r["split"]] = counts.get(r["split"], 0) + 1
        assert abs(counts.get("train", 0) / 1000 - 0.80) < 0.05
        assert abs(counts.get("validation", 0) / 1000 - 0.10) < 0.05
        assert abs(counts.get("clean-test", 0) / 1000 - 0.10) < 0.05

    def test_ratios_not_summing_to_one_raises(self):
        records = _make_records(10)
        with pytest.raises(ValueError, match="sum"):
            split_records(records, seed=42, ratios=(0.50, 0.30, 0.30))

    def test_wrong_number_of_ratios_raises(self):
        records = _make_records(10)
        with pytest.raises(ValueError):
            split_records(records, seed=42, ratios=(0.50, 0.50))

    def test_negative_ratio_raises(self):
        records = _make_records(10)
        with pytest.raises(ValueError):
            split_records(records, seed=42, ratios=(-0.10, 0.60, 0.50))

    def test_empty_records_returns_empty(self):
        result = split_records([], seed=42)
        assert result == []


# ===================================================================
# Section 4 – Source/cluster grouping when metadata exists
# ===================================================================

class TestSourceClusterGrouping:
    """Records sharing the same source must stay together in one split."""

    def test_single_source_stays_together(self):
        """All records from one source must land in the same split."""
        records = _make_records(20, source="single_source")
        result = split_records(records, seed=42)
        splits = {r["split"] for r in result}
        # With a single source, all records should be in one split.
        assert len(splits) == 1

    def test_multiple_sources_respect_grouping(self):
        """Each source's records must be entirely in one split."""
        records = []
        for src_idx in range(10):
            src_name = f"source_{src_idx}"
            for i in range(5):
                records.append(_make_record(
                    sample_id=f"{src_name}_S{i:04d}",
                    source=src_name,
                ))
        result = split_records(records, seed=42)
        # Group result by source
        source_splits: dict[str, set[str]] = {}
        for r in result:
            source_splits.setdefault(r["source"], set()).add(r["split"])
        # Each source should be in exactly one split
        for src, splits in source_splits.items():
            assert len(splits) == 1, (
                f"Source '{src}' was split across multiple splits: {splits}"
            )

    def test_source_grouping_with_cluster_field(self):
        """When a 'cluster' field exists, grouping uses it instead of source."""
        records = []
        for cluster_idx in range(5):
            cluster_name = f"cluster_{cluster_idx}"
            for i in range(10):
                records.append(_make_record(
                    sample_id=f"{cluster_name}_S{i:04d}",
                    source="mixed",
                    cluster=cluster_name,
                ))
        result = split_records(records, seed=42)
        cluster_splits: dict[str, set[str]] = {}
        for r in result:
            cluster_name = r.get("cluster", r["source"])
            cluster_splits.setdefault(cluster_name, set()).add(r["split"])
        for cl, splits in cluster_splits.items():
            assert len(splits) == 1, (
                f"Cluster '{cl}' was split across multiple splits: {splits}"
            )

    def test_oversized_single_source_stays_atomic(self):
        """Even a group larger than any target split stays in one split."""
        records = _make_records(500, source="oversized_source")
        result = split_records(records, seed=42)
        splits = {r["split"] for r in result}
        assert len(splits) == 1

    def test_oversized_cluster_stays_atomic(self):
        """An oversized cluster group must not be expanded into singletons."""
        records = [
            _make_record(
                sample_id=f"big_S{i:04d}",
                source="mixed",
                cluster="big_cluster",
            )
            for i in range(300)
        ]
        result = split_records(records, seed=42)
        splits = {r["split"] for r in result}
        assert len(splits) == 1

    def test_grouping_preserves_all_records(self):
        """Grouped splitting must not lose any records."""
        records = []
        for src_idx in range(5):
            for i in range(10):
                records.append(_make_record(
                    sample_id=f"src{src_idx}_S{i:04d}",
                    source=f"source_{src_idx}",
                ))
        result = split_records(records, seed=42)
        assert len(result) == len(records)


# ===================================================================
# Section 5 – Edge cases
# ===================================================================

class TestSplitEdgeCases:
    """Edge cases for split_records."""

    def test_single_record(self):
        records = [_make_record()]
        result = split_records(records, seed=42)
        assert len(result) == 1
        assert result[0]["split"] in {"train", "validation", "clean-test"}

    def test_two_records(self):
        records = _make_records(2)
        result = split_records(records, seed=42)
        assert len(result) == 2

    def test_does_not_mutate_input(self):
        records = _make_records(10)
        original_splits = [r["split"] for r in records]
        split_records(records, seed=42)
        current_splits = [r["split"] for r in records]
        assert current_splits == original_splits
