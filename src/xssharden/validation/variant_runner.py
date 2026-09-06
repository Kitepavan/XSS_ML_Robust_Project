"""Batch validation for generated XSS variant records.

Connects generation-style variant records to the existing controlled
local browser validator (:mod:`xssharden.validation.validator`) without
duplicating any browser logic.

Each input record must be a dict with at least a non-empty ``payload``
string. Provenance keys (``sample_id``, ``variant_id``, ``seed_id``,
``mutation_category``, ``source``, ``generator``, ``split``,
``attack_category``, ...) are preserved verbatim. Per-record ``probe``
and ``context_target`` (or legacy ``context``) override the call-level
defaults; the effective values are echoed in each result.

Only local targets are accepted (the check lives in
:func:`validate_payload` and runs before any payload is touched).
Tests inject a fake ``_runner`` so no browser, payload execution, or
network traffic is involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from xssharden.validation.validator import (
    DEFAULT_CONTEXT_TARGET,
    DEFAULT_PROBE,
    DEFAULT_TIMEOUT_MS,
    RunnerFn,
    validate_payload,
)

__all__ = ["validate_variants"]


def _check_limit(limit: Any) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(f"limit must be int or None, got {type(limit).__name__}")
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit!r}")
    return limit


def _materialize_records(records: Any) -> list[dict[str, Any]]:
    if records is None:
        raise TypeError("records must be an iterable of dicts, got None")
    if isinstance(records, (dict, str, bytes, bytearray)):
        raise TypeError(
            "records must be an iterable of dicts "
            f"(e.g. list), got {type(records).__name__}"
        )
    try:
        items = list(records)
    except TypeError as exc:
        raise TypeError(
            "records must be an iterable of dicts, "
            f"got {type(records).__name__}"
        ) from exc
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"record index {index}: must be a dict, "
                f"got {type(item).__name__}"
            )
        if "payload" not in item:
            raise ValueError(
                f"record index {index}: missing required field 'payload'"
            )
        payload = item["payload"]
        if not isinstance(payload, str):
            raise TypeError(
                f"record index {index}: field 'payload' must be str, "
                f"got {type(payload).__name__}"
            )
        if not payload.strip():
            raise ValueError(
                f"record index {index}: 'payload' must be a non-empty string"
            )
    return items


def validate_variants(
    records: Iterable[dict[str, Any]],
    *,
    sandbox_url: str | None = None,
    fixture_path: str | Path | None = None,
    timeout_ms: float = DEFAULT_TIMEOUT_MS,
    probe: str = DEFAULT_PROBE,
    context_target: str = DEFAULT_CONTEXT_TARGET,
    limit: int | None = None,
    _runner: RunnerFn | None = None,
) -> list[dict[str, Any]]:
    """Validate variant records against the controlled local sandbox.

    Parameters
    ----------
    records:
        Iterable of dicts, each with at least a non-empty ``payload``.
        Provenance keys are preserved. A per-record ``probe`` overrides
        ``probe``; a per-record ``context_target`` (or legacy ``context``)
        overrides ``context_target``.
    sandbox_url / fixture_path:
        Exactly one is required. ``sandbox_url`` must be a local
        ``http(s)://localhost|127.0.0.1|::1`` URL; ``fixture_path`` must
        be a local HTML file. Rejected before any payload is touched.
    timeout_ms / probe / context_target:
        Forwarded to :func:`validate_payload` for every record.
    limit:
        Optional non-negative int; only the first ``limit`` records are
        validated (``0`` returns ``[]`` without touching the runner).
    _runner:
        Injectable ``(payload, target, probe, timeout_ms)`` hook used by
        tests; when omitted, Playwright/Chromium is used.

    Returns
    -------
    list[dict]:
        One result dict per input record, in input order. Each result
        keeps all input keys and adds/refreshes ``valid``, ``status``,
        ``timed_out``, ``error``, ``duration_ms``, ``timeout_ms``,
        ``probe``, and ``context_target`` from the validator outcome.
        Invalid, timeout, and error outcomes are preserved, never
        filtered or retried.
    """
    checked_limit = _check_limit(limit)
    if sandbox_url is not None and fixture_path is not None:
        raise ValueError("pass exactly one of sandbox_url or fixture_path, not both")
    if sandbox_url is None and fixture_path is None:
        raise ValueError("one of sandbox_url or fixture_path is required")

    items = _materialize_records(records)
    if checked_limit is not None:
        items = items[:checked_limit]

    results: list[dict[str, Any]] = []
    for item in items:
        effective_probe = item["probe"] if "probe" in item else probe
        item_context = item.get("context_target") or item.get("context")
        if isinstance(item_context, str) and item_context.strip() and item_context.strip() != "unknown":
            effective_context = item_context.strip()
        else:
            effective_context = context_target
        outcome = validate_payload(
            item["payload"],
            sandbox_url=sandbox_url,
            fixture_path=fixture_path,
            context_target=effective_context,
            probe=effective_probe,
            timeout_ms=timeout_ms,
            _runner=_runner,
        )
        merged = dict(item)
        merged["payload"] = outcome.payload
        merged["probe"] = outcome.probe
        merged["context_target"] = outcome.context_target
        if "context" in merged:
            merged["context"] = outcome.context_target
        merged["valid"] = outcome.valid
        merged["status"] = outcome.status
        merged["timed_out"] = outcome.timed_out
        merged["error"] = outcome.error
        merged["duration_ms"] = outcome.duration_ms
        merged["timeout_ms"] = outcome.timeout_ms
        results.append(merged)
    return results
