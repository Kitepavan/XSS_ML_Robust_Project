"""Prepare the http_params raw candidate into project-schema JSONL.

Reads ``data/raw/http_params_dataset/payload_full.csv`` (columns
``payload,length,attack_type,label``) as plain text only — payloads are
never executed. Keeps only ``norm`` (benign) and ``xss`` (malicious) rows;
all other anomaly types (``sqli``, ``cmdi``, ``path-traversal``) are
excluded and counted in the summary so no audit information is lost.
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

SOURCE = "http_params_dataset"
SAMPLE_ID_PREFIX = "httpp-"

KEPT_TYPES = ("norm", "xss")

# attack_type -> (project label, project attack_category).
# NOTE: the raw source provides no injection-context information, so raw
# ``xss`` is mapped to the generic ``xss`` category (never ``reflected_html``)
# and every record carries ``context_target="unknown"`` metadata.
_TYPE_MAP: dict[str, tuple[int, str]] = {
    "norm": (0, "benign"),
    "xss": (1, "xss"),
}

CONTEXT_TARGET = "unknown"

# Raw ``label`` value expected for each kept ``attack_type``. Any deviation
# indicates a labelling inconsistency in the raw file and is rejected.
_EXPECTED_RAW_LABEL: dict[str, str] = {
    "norm": "norm",
    "xss": "anom",
}

_SAMPLE_ID_HEX_CHARS = 16


def _sample_id(attack_type: str, payload: str) -> str:
    """Return a deterministic content-based sample ID.

    The ID is ``httpp-<16 hex chars>`` where the hex digest is SHA-256 over
    ``source|attack_type|payload``. It is therefore stable under raw-row
    reordering and independent of filtered-row position.
    """
    digest = hashlib.sha256(
        f"{SOURCE}|{attack_type}|{payload}".encode("utf-8")
    ).hexdigest()[:_SAMPLE_ID_HEX_CHARS]
    return f"{SAMPLE_ID_PREFIX}{digest}"

EXPECTED_COLUMNS = ("payload", "length", "attack_type", "label")


def _parse_declared_length(raw: str | None) -> int | None:
    """Parse a declared ``length`` cell; return None when unparseable."""
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def prepare_http_params(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Convert the raw http_params CSV into project-schema JSONL.

    Parameters
    ----------
    input_path:
        Raw CSV with ``payload,length,attack_type,label`` columns.
    output_path:
        Destination ``.jsonl`` file written in the project schema
        (``sample_id,payload,label,source,attack_category,split`` plus
        audit extras). Parent directories are created as needed.
    seed:
        Deterministic seed forwarded to :func:`split_records`.

    Returns
    -------
    dict:
        Summary with ``total_rows``, ``kept_norm``, ``kept_xss``,
        ``kept_total`` (pre-deduplication kept rows),
        ``excluded_total``, ``excluded_by_type``, ``length_mismatches``
        (kept rows whose declared length differs from
        ``len(payload)``), ``duplicates_removed``, ``written``,
        ``splits`` (per-split counts), ``seed``, ``input`` and
        ``output``.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {src}")
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{src}: CSV has no header row")
        missing = [c for c in EXPECTED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{src}: missing expected columns: {missing}")
        raw_rows = list(reader)

    total_rows = len(raw_rows)
    excluded_by_type: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    kept_norm = 0
    kept_xss = 0
    length_mismatches = 0

    for lineno, row in enumerate(raw_rows, start=2):
        attack_type = str(row.get("attack_type", ""))
        if attack_type not in _TYPE_MAP:
            excluded_by_type[attack_type] += 1
            continue
        raw_label = str(row.get("label", ""))
        expected_label = _EXPECTED_RAW_LABEL[attack_type]
        if raw_label != expected_label:
            raise ValueError(
                f"{src}: line {lineno}: inconsistent raw label for "
                f"attack_type={attack_type!r}: expected label "
                f"{expected_label!r}, got {raw_label!r}"
            )
        payload = row.get("payload", "")
        if payload is None:
            payload = ""
        payload = str(payload)
        if not payload.strip():
            excluded_by_type[f"{attack_type}:empty-payload"] += 1
            continue
        label, category = _TYPE_MAP[attack_type]
        declared = _parse_declared_length(row.get("length"))
        actual = len(payload)
        mismatch = declared is None or declared != actual
        if mismatch:
            length_mismatches += 1
        if attack_type == "norm":
            kept_norm += 1
        else:
            kept_xss += 1
        records.append(
            {
                "sample_id": _sample_id(attack_type, payload),
                "payload": payload,
                "label": label,
                "source": SOURCE,
                "attack_category": category,
                "context_target": CONTEXT_TARGET,
                "raw_attack_type": attack_type,
                "raw_label": raw_label,
                "declared_length": declared if declared is not None else "",
                "actual_length": actual,
                "length_mismatch": mismatch,
            }
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
    excluded_total = sum(excluded_by_type.values())
    return {
        "total_rows": total_rows,
        "kept_norm": kept_norm,
        "kept_xss": kept_xss,
        "kept_total": kept_total,
        "excluded_total": excluded_total,
        "excluded_by_type": dict(sorted(excluded_by_type.items())),
        "length_mismatches": length_mismatches,
        "duplicates_removed": duplicates_removed,
        "written": len(split_out),
        "splits": split_counts,
        "seed": seed,
        "input": str(src),
        "output": str(dst),
    }
