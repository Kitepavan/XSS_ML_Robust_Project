"""Validity-gated variant selection for detector hardening.

Picks a budgeted subset of already-scored variant records for
augmentation. Validity is a hard gate — never a weighted trade-off:
only records with ``valid is True`` and ``label == 1`` are eligible,
and invalid, missing-valid, benign, held-out, or duplicated records
are never selected.

Pipeline::

    candidate records
        ↓
    validity hard gate (``valid is True`` and ``label == 1`` only)
        ↓
    split safety (held-out splits rejected; non-allow-listed excluded)
        ↓
    exact-payload deduplication (against training records, then within
    candidates, keeping the first occurrence in stable input order)
        ↓
    detector-impact ranking (``I(x) = 1 - detector_score``) or seeded
    random-valid sampling
        ↓
    per-category cap + fixed budget ``B``

Payloads are treated as inert strings throughout: this module never
imports or invokes the execution validator, never trains or tunes any
detector (scores arrive precomputed on the records), and never
renders, runs, or fetches anything.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

#: Splits that must never be selected from. They cannot be allow-listed:
#: any record carrying one raises ``ValueError``, and passing one via
#: ``allowed_splits`` raises ``ValueError`` as well.
BLOCKED_SPLITS: tuple[str, ...] = ("validation", "clean-test", "test")

#: Default eligible split for hardening candidates.
DEFAULT_ALLOWED_SPLITS: tuple[str, ...] = ("adv_dev",)

Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Typed result structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionResult:
    """Budgeted selection plus audit metadata.

    ``records`` holds verbatim copies of the selected input records in
    selection order. ``filtered_counts`` maps exclusion reasons to the
    number of records removed for that reason. ``shortfall`` is
    ``True`` when fewer eligible candidates existed than requested, in
    which case every eligible candidate is returned and nothing is
    fabricated.
    """

    records: list[Record] = field(default_factory=list)
    strategy: str = ""
    requested_budget: int = 0
    selected_count: int = 0
    eligible_count: int = 0
    seed: int = 42
    shortfall: bool = False
    filtered_counts: dict[str, int] = field(default_factory=dict)
    per_category_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as plain JSON-serializable dicts."""
        payload = asdict(self)
        payload["records"] = [dict(record) for record in self.records]
        return payload


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _check_budget(budget: Any) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"budget must be a positive integer, got {budget!r}")
    if budget <= 0:
        raise ValueError(f"budget must be a positive integer, got {budget!r}")
    return budget


def _check_strategy(strategy: Any) -> str:
    if strategy not in ("impact", "random_valid"):
        raise ValueError(
            "strategy must be 'impact' or 'random_valid', "
            f"got {strategy!r}"
        )
    return strategy


def _check_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    return seed


def _check_max_per_category(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"max_per_category must be a positive integer or None, "
            f"got {value!r}"
        )
    if value <= 0:
        raise ValueError(
            f"max_per_category must be a positive integer or None, "
            f"got {value!r}"
        )
    return value


def _check_allowed_splits(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            "allowed_splits must be a sequence of str or None, "
            f"got {type(value).__name__}"
        )
    items = list(value)
    for entry in items:
        if not isinstance(entry, str):
            raise TypeError(
                "allowed_splits entries must be str, "
                f"got {type(entry).__name__}"
            )
    for entry in items:
        if entry in BLOCKED_SPLITS:
            raise ValueError(
                f"allowed_splits must never contain held-out split "
                f"{entry!r} {list(BLOCKED_SPLITS)}; refusing to select "
                "from final held-out data"
            )
    return tuple(items)


def _materialize_records(records: Any, *, name: str) -> list[Record]:
    if records is None:
        raise TypeError(f"{name} must be an iterable of dicts, got None")
    if isinstance(records, (dict, str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be an iterable of dicts (e.g. list), "
            f"got {type(records).__name__}"
        )
    try:
        items = list(records)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable of dicts, "
            f"got {type(records).__name__}"
        ) from exc
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"{name} record at index {index} must be dict, "
                f"got {type(item).__name__}"
            )
    return items


def _check_candidate_shape(item: Record, index: int) -> None:
    payload = item.get("payload", _MISSING)
    if payload is _MISSING:
        raise ValueError(
            f"record at index {index}: missing required field 'payload'"
        )
    if not isinstance(payload, str):
        raise TypeError(
            f"record at index {index}: 'payload' must be str, "
            f"got {type(payload).__name__}"
        )
    if not payload.strip():
        raise ValueError(
            f"record at index {index}: 'payload' must be a non-empty string"
        )
    label = item.get("label", _MISSING)
    if label is _MISSING:
        raise ValueError(
            f"record at index {index}: missing required field 'label'"
        )
    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError(
            f"record at index {index}: 'label' must be 0 or 1, got {label!r}"
        )
    if label not in (0, 1):
        raise ValueError(
            f"record at index {index}: 'label' must be 0 or 1, got {label!r}"
        )


class _Missing:
    pass


_MISSING = _Missing()


def _check_split(item: Record, index: int, allowed: tuple[str, ...] | None) -> bool:
    """Enforce split safety.

    Returns ``True`` when the record's split is eligible. Raises
    ``ValueError`` for held-out splits; returns ``False`` for
    non-allow-listed (but not held-out) splits and for malformed
    ``split`` values.
    """
    split = item.get("split")
    if split is None:
        return True
    if not isinstance(split, str):
        raise TypeError(
            f"record at index {index}: 'split' must be str or absent, "
            f"got {type(split).__name__}"
        )
    if split in BLOCKED_SPLITS:
        raise ValueError(
            f"record at index {index}: split {split!r} is held out "
            f"{list(BLOCKED_SPLITS)} and must never be selected for "
            "hardening; refusing to use final held-out data"
        )
    if allowed is not None and split not in allowed:
        return False
    return True


def _score_of(item: Record, index: int) -> float:
    """Return the malicious-class score for impact ranking.

    Accepts ``detector_score`` (as written by the evasion module) with
    ``probability`` as an alias. Requires a finite float in [0, 1].
    """
    raw = item.get("detector_score", _MISSING)
    if raw is _MISSING:
        raw = item.get("probability", _MISSING)
    if raw is _MISSING:
        raise ValueError(
            f"record at index {index}: impact ranking requires "
            "'detector_score' (or 'probability')"
        )
    if isinstance(raw, bool):
        raise TypeError(
            f"record at index {index}: 'detector_score' must be a finite "
            f"float in [0, 1], got {raw!r}"
        )
    try:
        score = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"record at index {index}: 'detector_score' must be a finite "
            f"float in [0, 1], got {raw!r}"
        ) from exc
    if not math.isfinite(score) or not (0.0 <= score <= 1.0):
        raise ValueError(
            f"record at index {index}: 'detector_score' must be a finite "
            f"float in [0, 1], got {raw!r}"
        )
    return score


def _category_of(item: Record) -> str:
    for key in ("mutation_category", "attack_category", "category"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _training_payloads(training_records: Iterable[Record] | None) -> set[str]:
    if training_records is None:
        return set()
    items = _materialize_records(training_records, name="training_records")
    payloads: set[str] = set()
    for index, item in enumerate(items):
        payload = item.get("payload", _MISSING)
        if payload is _MISSING:
            raise ValueError(
                f"training record at index {index}: "
                "missing required field 'payload'"
            )
        if not isinstance(payload, str):
            raise TypeError(
                f"training record at index {index}: 'payload' must be str, "
                f"got {type(payload).__name__}"
            )
        payloads.add(payload)
    return payloads


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def select_variants(
    records: Iterable[Record],
    budget: int,
    *,
    strategy: str = "impact",
    seed: int = 42,
    max_per_category: int | None = None,
    training_records: Iterable[Record] | None = None,
    allowed_splits: Sequence[str] | None = DEFAULT_ALLOWED_SPLITS,
) -> SelectionResult:
    """Select a budgeted subset of valid, scored variant records.

    Parameters
    ----------
    records:
        Candidate records, each with a non-empty ``payload`` string and
        a binary ``label`` (0/1). Impact ranking additionally requires
        a finite ``detector_score`` (or ``probability`` alias) in
        [0, 1] on every eligible record. All other keys (``valid``,
        ``split``, ``variant_id``, provenance, ...) are preserved
        verbatim. Inputs are never mutated and payloads stay inert.
    budget:
        Positive integer: how many records to select at most. When
        fewer eligible candidates exist, all of them are returned and
        the shortfall is reported instead of fabricating records.
    strategy:
        ``"impact"`` ranks eligible records by detector impact
        ``I(x) = 1 - detector_score`` (lowest malicious probability
        first) with deterministic tie-breaking on stable input order
        then ``variant_id``/``payload``. ``"random_valid"`` draws the
        mandatory budget-matched random-valid control from the same
        eligible, deduplicated pool with a seeded deterministic RNG.
    seed:
        Deterministic RNG seed for ``"random_valid"`` (recorded in the
        audit metadata for both strategies).
    max_per_category:
        Optional positive integer cap per mutation category, enforced
        deterministically for both strategies.
    training_records:
        Optional records whose exact payloads are excluded from
        selection (deduplication against training data).
    allowed_splits:
        Explicit allow-list of eligible ``split`` values (default
        ``("adv_dev",)``); ``None`` allows any non-held-out split.
        Held-out splits (``validation``/``clean-test``/``test``) can
        never be allow-listed and any record carrying one raises
        ``ValueError``. Records with another non-allow-listed split
        are excluded and counted, never silently selected.

    Returns
    -------
    SelectionResult:
        Selected record copies in selection order plus audit metadata
        (strategy, requested budget, selected/eligible counts,
        filtered counts by reason, seed, shortfall flag, and
        per-category counts).
    """
    budget = _check_budget(budget)
    strategy = _check_strategy(strategy)
    seed = _check_seed(seed)
    cap = _check_max_per_category(max_per_category)
    allowed = _check_allowed_splits(allowed_splits)
    items = _materialize_records(records, name="records")
    seen_training = _training_payloads(training_records)

    filtered_counts: dict[str, int] = {}

    def _count(reason: str) -> None:
        filtered_counts[reason] = filtered_counts.get(reason, 0) + 1

    # -- Hard gates: shape, split safety, validity, dedup -------------------
    eligible: list[tuple[Record, int]] = []  # (record, input index)
    seen_payloads: set[str] = set()
    for index, item in enumerate(items):
        _check_candidate_shape(item, index)
        if not _check_split(item, index, allowed):
            _count("split_not_allowed")
            continue
        if item.get("valid") is not True or int(item["label"]) != 1:
            if int(item["label"]) != 1:
                _count("benign")
            else:
                _count("not_valid")
            continue
        payload = item["payload"]
        if payload in seen_training:
            _count("duplicate_of_training")
            continue
        if payload in seen_payloads:
            _count("duplicate_within_candidates")
            continue
        seen_payloads.add(payload)
        eligible.append((item, index))

    # -- Ordering ------------------------------------------------------------
    if strategy == "impact":
        scored = [
            (_score_of(item, index), index, item) for item, index in eligible
        ]
        scored.sort(
            key=lambda triple: (
                triple[0],
                triple[1],
                str(triple[2].get("variant_id", "")),
                triple[2]["payload"],
            )
        )
        ordered: list[Record] = [triple[2] for triple in scored]
    else:
        rng = random.Random(seed)
        ordered = [item for item, _ in eligible]
        rng.shuffle(ordered)

    # -- Budget + category cap ----------------------------------------------
    selected: list[Record] = []
    per_category_counts: dict[str, int] = {}
    for item in ordered:
        if len(selected) >= budget:
            break
        category = _category_of(item)
        if cap is not None and per_category_counts.get(category, 0) >= cap:
            continue
        selected.append(dict(item))
        per_category_counts[category] = per_category_counts.get(category, 0) + 1

    shortfall = len(eligible) < budget
    return SelectionResult(
        records=selected,
        strategy=strategy,
        requested_budget=budget,
        selected_count=len(selected),
        eligible_count=len(eligible),
        seed=seed,
        shortfall=shortfall,
        filtered_counts=dict(filtered_counts),
        per_category_counts=dict(per_category_counts),
    )
