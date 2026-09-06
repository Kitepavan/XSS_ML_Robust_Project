"""Dataset cleaning and deduplication for XSSHarden records."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def normalize_payload(payload: str) -> str:
    """Return a deterministic, normalised form of *payload*.

    The normalisation is intentionally lightweight:

    1. Convert to lowercase.
    2. Strip whitespace immediately after ``<`` and before ``>``.
    3. Collapse every run of whitespace characters to a single ASCII space.
    4. Strip leading / trailing whitespace.

    This is *not* an HTML canonicalisation; it is a practical de-duplication
    aid that treats ``<Script>``, ``<script>``, and ``<  script  >`` as
    equivalent while remaining cheap and deterministic.
    """
    lowered = payload.lower()
    # Strip spaces around tag delimiters for stable de-duplication.
    no_boundary_ws = re.sub(r"<\s+", "<", lowered)
    no_boundary_ws = re.sub(r"\s+>", ">", no_boundary_ws)
    no_boundary_ws = re.sub(r">\s+", ">", no_boundary_ws)
    no_boundary_ws = re.sub(r"\s+<", "<", no_boundary_ws)
    collapsed = re.sub(r"\s+", " ", no_boundary_ws)
    return collapsed.strip()


def _content_key(record: dict[str, Any]) -> str:
    """Return a content-based hashable key for *record*."""
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


# Fields used when building the normalised de-duplication key.
_KEY_FIELDS = ("payload", "label", "source", "attack_category", "split")


def _normalised_key(record: dict[str, Any]) -> tuple[str, ...]:
    """Return a hashable normalised key for *record*."""
    parts: list[str] = []
    for field in _KEY_FIELDS:
        value = record.get(field, "")
        if field == "payload":
            value = normalize_payload(str(value))
        else:
            value = str(value)
        parts.append(value)
    return tuple(parts)


def deduplicate_records(
    records: list[dict[str, Any]],
    exact_only: bool = False,
) -> list[dict[str, Any]]:
    """Remove duplicate records, preserving the first occurrence of each.

    Parameters
    ----------
    records:
        Input dataset rows.
    exact_only:
        When *True*, only byte-identical duplicates are removed.
        When *False* (the default), records are compared on a
        normalised key (case-folded, whitespace-collapsed payload
        plus label / source / attack_category / split).

    Returns
    -------
    list[dict]:
        De-duplicated records in their original insertion order.
    """
    seen_exact: set[str] = set()
    seen_norm: set[tuple[str, ...]] = set()
    result: list[dict[str, Any]] = []

    for record in records:
        if exact_only:
            key = _content_key(record)
            if key in seen_exact:
                continue
            seen_exact.add(key)
        else:
            key = _normalised_key(record)
            if key in seen_norm:
                continue
            seen_norm.add(key)

        result.append(record)

    return result
