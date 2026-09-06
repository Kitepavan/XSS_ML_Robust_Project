"""Dataset schema validation for XSSHarden records."""

from __future__ import annotations

from typing import Any

# Required fields and their expected types.
REQUIRED_FIELDS: dict[str, type] = {
    "sample_id": str,
    "payload": str,
    "label": int,
    "source": str,
    "attack_category": str,
    "split": str,
}


class _ValidationError(ValueError):
    """Raised when a record fails schema validation."""


def validate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate *records* against the required schema.

    Parameters
    ----------
    records:
        A list of dicts, each representing one dataset row.

    Returns
    -------
    list[dict]:
        The same records, passing through unchanged.

    Raises
    ------
    TypeError
        If an element of *records* is not a dict.
    ValueError
        If a required field is missing, has the wrong type, or the
        payload is empty / whitespace-only.
    """
    errors: list[str] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Record at index {idx} is not a dict (got {type(record).__name__})")

        # Check required fields exist.
        for field in REQUIRED_FIELDS:
            if field not in record:
                errors.append(f"Record index {idx}: missing required field '{field}'")

        if errors:
            # Bail early on missing fields before checking types.
            raise ValueError("; ".join(errors))

        # Check types.
        for field, expected in REQUIRED_FIELDS.items():
            if not isinstance(record[field], expected):
                raise TypeError(
                    f"Record index {idx}: field '{field}' must be {expected.__name__}, "
                    f"got {type(record[field]).__name__}"
                )

        # Payload must be non-empty (after strip).
        if not record["payload"].strip():
            raise ValueError(
                f"Record index {idx}: 'payload' must be a non-empty string"
            )

    return records
