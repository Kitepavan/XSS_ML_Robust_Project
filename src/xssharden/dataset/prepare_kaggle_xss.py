"""Kaggle XSS dataset preparation adapter.

Reads ``data/raw/kaggle_xss_dataset/XSS_dataset.csv`` (columns
``<index>,Sentence,Label``) as plain text only — payloads are never
executed — and converts it to the project schema
(``sample_id,payload,label,source,attack_category,split`` plus audit
extras).

Mapping:

* ``Sentence`` → ``payload`` (verbatim; whitespace-only payloads are
  skipped and counted, never kept).
* ``Label`` ``0`` → project label ``0`` (``benign``),
  ``Label`` ``1`` → project label ``1`` (``xss``). Anything else is
  rejected with :class:`ValueError`.

Every output record carries ``source="kaggle_xss_dataset"``,
``context_target="unknown"``, ``license_status="unknown"``,
``raw_label`` (original ``Label`` spelling), ``raw_row_number``
(1-indexed data-row position), and ``raw_source_id`` (the original
first-column index value, ``""`` when the input has no extra column),
plus a deterministic content-based ``sample_id``
(``kaggle_xss_dataset-<16 hex>`` over ``source|label|payload``).
Deduplication reuses
:func:`xssharden.dataset.clean.deduplicate_records` and splitting reuses
the group-atomic :func:`xssharden.dataset.split.split_records`.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from xssharden.dataset.clean import deduplicate_records
from xssharden.dataset.io import write_records
from xssharden.dataset.split import split_records

SOURCE = "kaggle_xss_dataset"
CONTEXT_TARGET = "unknown"
LICENSE_STATUS = "unknown"

SENTENCE_COLUMN = "Sentence"
LABEL_COLUMN = "Label"

_SAMPLE_ID_HEX_CHARS = 16


def _sample_id(source: str, label: int, payload: str) -> str:
    """Return a deterministic content-based sample ID.

    The ID is ``<source>-<16 hex chars>`` where the hex digest is
    SHA-256 over ``source|label|payload`` (canonical integer label and
    verbatim payload). It therefore includes ``source``, is stable
    under raw-row reordering, and separates identically-valued payloads
    from different sources. Raw row numbers and raw source IDs are
    deliberately excluded so reordering never changes IDs.
    """
    digest = hashlib.sha256(
        f"{source}|{label}|{payload}".encode("utf-8")
    ).hexdigest()[:_SAMPLE_ID_HEX_CHARS]
    return f"{source}-{digest}"


def _canonical_label(raw: Any, where: str) -> int:
    """Map a raw Kaggle ``Label`` value to canonical ``0``/``1``.

    Accepts integer ``0``/``1`` and their string forms (surrounding
    whitespace ignored). Everything else — including ``None``, bools,
    floats, empty strings, out-of-range numbers, and textual labels
    such as ``"benign"``/``"xss"`` — is rejected with
    :class:`ValueError` to preserve source fidelity.
    """
    if isinstance(raw, bool):
        raise ValueError(
            f"{where}: unsupported label {raw!r}: expected 0/1"
        )
    if isinstance(raw, int):
        if raw in (0, 1):
            return raw
        raise ValueError(
            f"{where}: unsupported label {raw!r}: expected 0/1"
        )
    if isinstance(raw, float):
        raise ValueError(
            f"{where}: unsupported label {raw!r}: expected 0/1"
        )
    if isinstance(raw, str):
        normalized = raw.strip()
        if normalized == "0":
            return 0
        if normalized == "1":
            return 1
        raise ValueError(
            f"{where}: unsupported label {raw!r}: expected 0/1"
        )
    raise ValueError(f"{where}: unsupported label {raw!r}: expected 0/1")


def _normalized_fieldnames(raw_names: list[str] | None, path: Path) -> list[str]:
    """Strip a BOM prefix from the first header cell, if present."""
    if not raw_names:
        raise ValueError(f"{path}: CSV has no header row")
    names = list(raw_names)
    if names and names[0].startswith("\ufeff"):
        names[0] = names[0].lstrip("\ufeff")
    return names


def prepare_kaggle_xss(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Convert the raw Kaggle ``XSS_dataset.csv`` into project-schema records.

    Parameters
    ----------
    input_path:
        Raw CSV with ``Sentence,Label`` columns plus the original
        index column (empty header name in the shipped file). Read as
        inert text; payloads are never executed. Only the ``.csv``
        suffix is accepted.
    output_path:
        Destination file written via :func:`write_records` (``.jsonl``,
        ``.json`` or ``.csv``). Parent directories are created as
        needed.
    seed:
        Deterministic seed forwarded to :func:`split_records`.

    Returns
    -------
    dict:
        Summary with ``total_rows``, ``kept_benign``, ``kept_xss``,
        ``kept_total`` (pre-deduplication kept rows),
        ``skipped_blank`` (whitespace-only/empty ``Sentence`` rows
        skipped, never kept), ``duplicates_removed``, ``written``,
        ``splits`` (per-split counts), ``seed``, ``source``,
        ``input`` and ``output``.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    ValueError
        For unsupported input suffix, missing ``Sentence``/``Label``
        columns, unsupported labels, or single-class input. Blank
        payloads are skipped (and counted), not rejected.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {src}")
    if src.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported dataset format '{src.suffix}': "
            "expected '.csv' with Sentence,Label columns"
        )

    with src.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = _normalized_fieldnames(reader.fieldnames, src)
        missing = [
            c
            for c in (SENTENCE_COLUMN, LABEL_COLUMN)
            if c not in fieldnames
        ]
        if missing:
            raise ValueError(
                f"{src}: missing required columns: {missing}"
            )
        # The original per-row source ID lives in the extra index
        # column (empty header in the shipped file). Preserve it when
        # present; otherwise fall back to "".
        id_columns = [
            c for c in fieldnames
            if c not in (SENTENCE_COLUMN, LABEL_COLUMN)
        ]
        id_column = id_columns[0] if id_columns else None
        raw_rows = list(reader)

    total_rows = len(raw_rows)
    records: list[dict[str, Any]] = []
    kept_benign = 0
    kept_xss = 0
    skipped_blank = 0

    for position, row in enumerate(raw_rows, start=1):
        where = f"{src}: row {position}"
        sentence_raw = row.get(SENTENCE_COLUMN)
        label_raw = row.get(LABEL_COLUMN)
        if sentence_raw is None or (
            isinstance(sentence_raw, str) and not sentence_raw.strip()
        ):
            skipped_blank += 1
            continue
        if not isinstance(sentence_raw, str):
            skipped_blank += 1
            continue
        payload = sentence_raw
        label = _canonical_label(label_raw, where)
        if label == 0:
            kept_benign += 1
            category = "benign"
        else:
            kept_xss += 1
            category = "xss"
        if id_column is not None:
            raw_source_id = row.get(id_column)
            raw_source_id = "" if raw_source_id is None else str(raw_source_id)
        else:
            raw_source_id = ""
        records.append(
            {
                "sample_id": _sample_id(SOURCE, label, payload),
                "payload": payload,
                "label": label,
                "source": SOURCE,
                "attack_category": category,
                "context_target": CONTEXT_TARGET,
                "license_status": LICENSE_STATUS,
                "raw_label": "" if label_raw is None else str(label_raw),
                "raw_row_number": position,
                "raw_source_id": raw_source_id,
            }
        )

    if kept_benign == 0 or kept_xss == 0:
        raise ValueError(
            f"{src}: input must contain both classes (benign and xss); "
            f"got benign={kept_benign} xss={kept_xss} over {total_rows} rows "
            f"({skipped_blank} blank rows skipped)"
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
        "skipped_blank": skipped_blank,
        "duplicates_removed": duplicates_removed,
        "written": len(split_out),
        "splits": split_counts,
        "seed": seed,
        "source": SOURCE,
        "input": str(src),
        "output": str(dst),
    }
