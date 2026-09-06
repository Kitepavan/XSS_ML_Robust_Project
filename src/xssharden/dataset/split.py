"""Leakage-safe reproducible splitting for XSSHarden records."""

from __future__ import annotations

import random
from typing import Any

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
CLEAN_TEST_SPLIT = "clean-test"

VALID_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, CLEAN_TEST_SPLIT)

DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def _validate_ratios(ratios: tuple[float, ...] | list[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError(f"ratios must contain exactly 3 values, got {len(ratios)}")
    for value in ratios:
        if value < 0:
            raise ValueError(f"ratios must be non-negative, got {ratios}")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}")
    return (float(ratios[0]), float(ratios[1]), float(ratios[2]))


def _group_key(record: dict[str, Any], index: int, use_cluster: bool) -> str:
    if use_cluster and "cluster" in record:
        return str(record.get("cluster"))
    if use_cluster:
        # Mixed corpora: records without a cluster fall back to source.
        source = record.get("source")
        if source is not None:
            return str(source)
        return str(record.get("sample_id", f"__idx_{index}"))
    source = record.get("source")
    if source is not None:
        return str(source)
    return str(record.get("sample_id", f"__idx_{index}"))


def split_records(
    records: list[dict[str, Any]],
    seed: int = 42,
    ratios: tuple[float, float, float] | list[float] = DEFAULT_RATIOS,
) -> list[dict[str, Any]]:
    """Split records into train/validation/clean-test groups.

    Source/cluster grouping is always preserved: records sharing the same
    grouping key are assigned to the same split, even when a group is
    larger than any target split. When a ``cluster`` field is present
    anywhere in the input it takes precedence over ``source``. As a
    consequence, a corpus dominated by one oversized group cannot meet the
    target ratios — that skew is reported honestly rather than hidden by
    splitting the group.

    Assignment is deterministic for a given ``seed`` and does not mutate the
    input records.
    """
    r_train, r_val, r_test = _validate_ratios(ratios)

    if not records:
        return []

    # Copy so the input is never mutated.
    copies = [dict(record) for record in records]
    n = len(copies)

    use_cluster = any("cluster" in record for record in copies)

    # Build groups of indices sharing a grouping key.
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(copies):
        key = _group_key(record, idx, use_cluster)
        groups.setdefault(key, []).append(idx)

    # Target counts that sum exactly to n.
    n_train = int(round(n * r_train))
    n_val = int(round(n * r_val))
    n_test = n - n_train - n_val
    # Guard against pathological rounding producing negatives.
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    if n_val < 0:
        n_val = 0
        n_train = n - n_test
    targets = [n_train, n_val, n_test]

    # Groups are always atomic: every member of a source/cluster group
    # shares one split, no matter how large the group is.
    units: list[list[int]] = []
    for key in sorted(groups.keys()):
        units.append(list(groups[key]))

    # Deterministic seeded ordering of groups.
    rng = random.Random(seed)
    order = list(range(len(units)))
    rng.shuffle(order)

    assigned = [0, 0, 0]
    split_index_for_row: dict[int, int] = {}
    for unit_pos in order:
        members = units[unit_pos]
        # Pick the split with the largest remaining need.
        remaining = [targets[i] - assigned[i] for i in range(3)]
        # Deterministic tie-break: train, then validation, then clean-test.
        best = max(range(3), key=lambda i: (remaining[i], -i))
        for idx in members:
            split_index_for_row[idx] = best
        assigned[best] += len(members)

    for idx, record in enumerate(copies):
        record["split"] = VALID_SPLITS[split_index_for_row[idx]]

    return copies
