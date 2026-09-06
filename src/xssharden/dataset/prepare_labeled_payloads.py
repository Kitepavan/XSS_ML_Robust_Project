"""Generic independent labeled-payload source preparation adapter.

Reads a raw CSV or JSONL file with ``payload`` and ``label`` fields as
plain text only — payloads are never executed — and converts it to the
project schema (``sample_id,payload,label,source,attack_category,split``
plus audit extras).

Accepted labels (case-insensitive, surrounding whitespace ignored):

* ``0`` / ``"0"`` / ``"benign"`` → project label ``0`` (``benign``)
* ``1`` / ``"1"`` / ``"xss"`` → project label ``1`` (``xss``)

Anything else is rejected with :class:`ValueError`. Missing or
blank (whitespace-only) payloads are rejected, and the input must
contain both classes. Deduplication reuses
:func:`xssharden.dataset.clean.deduplicate_records` and splitting reuses
the group-atomic :func:`xssharden.dataset.split.split_records`.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from xssharden.dataset.clean import deduplicate_records
from xssharden.dataset.io import write_records
from xssharden.dataset.split import split_records

RESERVED_SOURCE = "http_params_dataset"

CONTEXT_TARGET = "unknown"

_SAMPLE_ID_HEX_CHARS = 16


def _sample_id(source_name: str, label: int, payload: str) -> str:
    """Return a deterministic content-based sample ID.

    The ID is ``<source_name>-<16 hex chars>`` where the hex digest is
    SHA-256 over ``source_name|label|payload`` (canonical integer label
    and verbatim payload). It therefore includes ``source_name``, is
    stable under raw-row reordering, and separates identically-valued
    payloads from different sources.
    """
    digest = hashlib.sha256(
        f"{source_name}|{label}|{payload}".encode("utf-8")
    ).hexdigest()[:_SAMPLE_ID_HEX_CHARS]
    return f"{source_name}-{digest}"


def _canonical_label(raw: Any, where: str) -> int:
    """Map a raw label value to canonical ``0``/``1``.

    Raises
    ------
    ValueError
        If *raw* is not ``0``/``1`` or ``benign``/``xss``
        (case-insensitive) .
    """
    if isinstance(raw, bool):
        raise ValueError(f"{where}: unsupported label {raw!r}: expected 0/1 or benign/xss")
    if isinstance(raw, int):
        if raw in (0, 1):
            return raw
        raise ValueError(
            f"{where}: unsupported label {raw!r}: expected 0/1 or benign/xss"
        )
    if isinstance(raw, float):
        raise ValueError(
            f"{where}: unsupported label {raw!r}: expected 0/1 or benign/xss"
        )
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized == "0" or normalized == "benign":
            return 0
        if normalized == "1" or normalized == "xss":
            return 1
        raise ValueError(
            f"{where}: unsupported label {raw!r}: expected 0/1 or benign/xss"
        )
    raise ValueError(
        f"{where}: unsupported label {raw!r}: expected 0/1 or benign/xss"
    )


def _read_raw_rows(path: Path) -> tuple[list[tuple[Any, Any]], str]:
    """Read raw ``(payload, label)`` pairs in file order.

    Returns the list of pairs plus the detected format (``"csv"``,
    ``"jsonl"`` or ``"json"``). ``.jsonl`` is always one JSON object
    per line (blank lines skipped). ``.json`` accepts either a top-level
    JSON array of objects or the same JSONL-style one-object-per-line
    layout. Array elements use ``record <n>`` error locations while
    JSONL lines use ``line <n>`` locations. Raises :class:`ValueError`
    when required ``payload``/``label`` fields are missing and
    :class:`ValueError` for unsupported suffixes.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: CSV has no header row")
            missing = [
                c for c in ("payload", "label") if c not in reader.fieldnames
            ]
            if missing:
                raise ValueError(
                    f"{path}: missing required columns: {missing}"
                )
            pairs = [
                (row.get("payload"), row.get("label")) for row in reader
            ]
        return pairs, "csv"
    if suffix == ".jsonl":
        pairs: list[tuple[Any, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: invalid JSON on line {lineno}: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise TypeError(
                        f"{path}: line {lineno} must be a JSON object, "
                        f"got {type(obj).__name__}"
                    )
                if "payload" not in obj:
                    raise ValueError(
                        f"{path}: line {lineno}: missing required field 'payload'"
                    )
                if "label" not in obj:
                    raise ValueError(
                        f"{path}: line {lineno}: missing required field 'label'"
                    )
                pairs.append((obj.get("payload"), obj.get("label")))
        return pairs, "jsonl"
    if suffix == ".json":
        text = path.read_text(encoding="utf-8")
        if text.lstrip().startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON: {exc}") from exc
            if not isinstance(data, list):
                raise ValueError(
                    f"{path}: top-level JSON must be an array, "
                    f"got {type(data).__name__}"
                )
            pairs = []
            for index, obj in enumerate(data, start=1):
                if not isinstance(obj, dict):
                    raise TypeError(
                        f"{path}: record {index} must be a JSON object, "
                        f"got {type(obj).__name__}"
                    )
                if "payload" not in obj:
                    raise ValueError(
                        f"{path}: record {index}: missing required field 'payload'"
                    )
                if "label" not in obj:
                    raise ValueError(
                        f"{path}: record {index}: missing required field 'label'"
                    )
                pairs.append((obj.get("payload"), obj.get("label")))
            return pairs, "json"
        pairs = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: invalid JSON on line {lineno}: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise TypeError(
                        f"{path}: line {lineno} must be a JSON object, "
                        f"got {type(obj).__name__}"
                    )
                if "payload" not in obj:
                    raise ValueError(
                        f"{path}: line {lineno}: missing required field 'payload'"
                    )
                if "label" not in obj:
                    raise ValueError(
                        f"{path}: line {lineno}: missing required field 'label'"
                    )
                pairs.append((obj.get("payload"), obj.get("label")))
        return pairs, "jsonl"
    raise ValueError(
        f"Unsupported dataset format '{suffix}': expected one of "
        "'.jsonl', '.json', '.csv'"
    )


def prepare_labeled_payloads(
    input_path: str | Path,
    output_path: str | Path,
    source_name: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepare a generic labeled payload file into project-schema records.

    Parameters
    ----------
    input_path:
        Raw CSV (header must contain ``payload,label``), JSONL
        (``.jsonl``, each object must contain ``payload`` and ``label``),
        or JSON (``.json``: either a top-level array of ``payload``/``label``
        objects or JSONL-style one-object-per-line). Read as inert text;
        payloads are never executed.
    output_path:
        Destination file written via :func:`write_records` (``.jsonl``,
        ``.json`` or ``.csv``). Parent directories are created as needed.
    source_name:
        Provenance name stored in each record's ``source`` field and
        embedded in every content-based sample ID. Must be a non-blank
        string and must not equal ``"http_params_dataset"`` (to prevent
        accidental self-merging with the first dataset).
    seed:
        Deterministic seed forwarded to :func:`split_records`.

    Returns
    -------
    dict:
        Summary with ``total_rows``, ``kept_benign``, ``kept_xss``,
        ``kept_total`` (pre-deduplication kept rows),
        ``duplicates_removed``, ``written``, ``splits`` (per-split
        counts), ``seed``, ``source``, ``input`` and ``output``.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    ValueError
        For blank ``source_name``, reserved ``source_name``,
        unsupported input suffix, missing/blank payloads, unsupported
        labels, or single-class input.
    """
    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError("source_name must be a non-blank string")
    source = source_name.strip()
    if source == RESERVED_SOURCE:
        raise ValueError(
            f"source_name {source!r} is reserved (would self-merge with "
            f"{RESERVED_SOURCE!r}); choose a distinct independent source name"
        )

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {src}")
    pairs, _ = _read_raw_rows(src)

    total_rows = len(pairs)
    records: list[dict[str, Any]] = []
    kept_benign = 0
    kept_xss = 0

    for position, (payload_raw, label_raw) in enumerate(pairs, start=1):
        where = f"{src}: row {position}"
        if payload_raw is None:
            raise ValueError(f"{where}: missing required field 'payload'")
        if not isinstance(payload_raw, str):
            raise ValueError(
                f"{where}: 'payload' must be a string, "
                f"got {type(payload_raw).__name__}"
            )
        payload = payload_raw
        if not payload.strip():
            raise ValueError(f"{where}: 'payload' must be a non-empty string")
        label = _canonical_label(label_raw, where)
        if label == 0:
            kept_benign += 1
            category = "benign"
        else:
            kept_xss += 1
            category = "xss"
        records.append(
            {
                "sample_id": _sample_id(source, label, payload),
                "payload": payload,
                "label": label,
                "source": source,
                "attack_category": category,
                "context_target": CONTEXT_TARGET,
                "raw_label": str(label_raw),
                "raw_row_number": position,
            }
        )

    if kept_benign == 0 or kept_xss == 0:
        raise ValueError(
            f"{src}: input must contain both classes (benign and xss); "
            f"got benign={kept_benign} xss={kept_xss} over {total_rows} rows"
        )

    kept_total = len(records)
    deduped = deduplicate_records(records)
    duplicates_removed = kept_total - len(deduped)
    split_out = split_records(deduped, seed=seed)
    dst = Path(output_path)
    if str(dst.parent) and str(dst.parent) not in ("", "."):
        dst.parent.mkdir(parents=True, exist_ok=True)
    write_records(split_out, dst)

    split_counts = dict(Counter(r["split"] for r in split_out))
    return {
        "total_rows": total_rows,
        "kept_benign": kept_benign,
        "kept_xss": kept_xss,
        "kept_total": kept_total,
        "duplicates_removed": duplicates_removed,
        "written": len(split_out),
        "splits": split_counts,
        "seed": seed,
        "source": source,
        "input": str(src),
        "output": str(dst),
    }
