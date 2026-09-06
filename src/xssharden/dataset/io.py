"""Deterministic JSONL/CSV I/O for XSSHarden dataset records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from xssharden.dataset.schema import REQUIRED_FIELDS, validate_records

# Canonical column order: required schema fields first, extras appended sorted.
CANONICAL_FIELDS: tuple[str, ...] = tuple(REQUIRED_FIELDS.keys())

_JSONL_SUFFIXES = {".jsonl", ".json"}
_CSV_SUFFIXES = {".csv"}


def _suffix(path: str | Path) -> str:
    """Return the lowercase file suffix of *path*."""
    return Path(path).suffix.lower()


def _fieldnames(records: list[dict[str, Any]]) -> list[str]:
    """Return deterministic CSV column names for *records*."""
    extras: set[str] = set()
    for record in records:
        extras.update(k for k in record.keys() if k not in CANONICAL_FIELDS)
    return list(CANONICAL_FIELDS) + sorted(extras)


def _coerce_csv_value(field: str, value: str | None) -> Any:
    """Coerce a raw CSV string back to its schema type where known."""
    if value is None:
        return ""
    expected = REQUIRED_FIELDS.get(field)
    if expected is int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise TypeError(
                    f"{path}: line {lineno} must be a JSON object, "
                    f"got {type(obj).__name__}"
                )
            records.append(obj)
    return records


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [
            {field: _coerce_csv_value(field, value) for field, value in row.items()}
            for row in reader
        ]


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read dataset records from a JSONL or CSV file.

    Records are returned in file order and validated against the required
    schema. CSV ``label`` values are coerced back to ``int`` so a
    write/read round-trip preserves types.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the suffix is not ``.jsonl``/``.json``/``.csv`` or validation fails.
    TypeError
        If a row is not a mapping or a field has the wrong type.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    suffix = _suffix(resolved)
    if suffix in _JSONL_SUFFIXES:
        records = _read_jsonl(resolved)
    elif suffix in _CSV_SUFFIXES:
        records = _read_csv(resolved)
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}': expected one of "
            "'.jsonl', '.json', '.csv'"
        )
    validate_records(records)
    return records


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            )


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = _fieldnames(records) if records else list(CANONICAL_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def write_records(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write dataset records to a JSONL or CSV file.

    Records are validated *before* anything is written so a failed call
    never leaves a partial file behind. Output is deterministic: JSONL keys
    are sorted and CSV columns follow the canonical schema order with any
    extra fields appended alphabetically. File order is preserved.

    Raises
    ------
    ValueError
        If the suffix is not ``.jsonl``/``.json``/``.csv`` or validation fails.
    TypeError
        If a record is not a dict or a field has the wrong type.
    """
    if not isinstance(records, list):
        raise TypeError(f"records must be a list, got {type(records).__name__}")
    validate_records(records)
    resolved = Path(path)
    suffix = _suffix(resolved)
    if suffix in _JSONL_SUFFIXES:
        _write_jsonl(records, resolved)
    elif suffix in _CSV_SUFFIXES:
        _write_csv(records, resolved)
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}': expected one of "
            "'.jsonl', '.json', '.csv'"
        )
