"""Safe multi-source dataset merger.

Reads already-prepared project-schema files (JSONL/JSON/CSV) through the
shared I/O layer as inert text — payloads are never executed — and merges
them into one deduplicated, leakage-safe corpus.

Pipeline::

    prepared files (project schema, inert text)
        ↓
    duplicate-path + input validation (refuse before writing)
        ↓
    distinct-``source`` gate (at least 3 groups, else refuse)
        ↓
    shared exact/normalized deduplication
        ↓
    group-atomic ``split_records`` (sources stay together)
        ↓
    deterministic project-schema output + summary

Every input key (including provenance extras such as ``raw_label``,
``context_target``, or ``license_status``) is preserved verbatim; only
``split`` is reassigned by the group-atomic splitter. Output is
deterministic for fixed inputs and ``seed``: inputs concatenate in the
given order, the first occurrence of each duplicate wins, groups shuffle
under the seeded RNG, and records serialize with sorted keys in file
order. This module performs no network access, launches no browser, and
never imports the validator.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from xssharden.dataset.clean import deduplicate_records
from xssharden.dataset.io import read_records, write_records
from xssharden.dataset.split import split_records

#: Minimum number of distinct ``source`` values required to merge.
MIN_SOURCES = 3


def _check_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    return seed


def _resolve_paths(input_paths: Sequence[str | Path]) -> list[Path]:
    if input_paths is None or isinstance(input_paths, (str, bytes, Path)):
        raise TypeError(
            "input_paths must be a sequence of at least 2 input file paths, "
            f"got {type(input_paths).__name__}"
        )
    try:
        paths = list(input_paths)
    except TypeError as exc:
        raise TypeError(
            "input_paths must be a sequence of at least 2 input file paths, "
            f"got {type(input_paths).__name__}"
        ) from exc
    if len(paths) < 2:
        raise ValueError(
            f"dataset merge requires at least 2 input files, got {len(paths)}"
        )
    for index, entry in enumerate(paths):
        if not isinstance(entry, (str, Path)):
            raise TypeError(
                f"input_paths[{index}] must be str or Path, "
                f"got {type(entry).__name__}"
            )
        if isinstance(entry, str) and not entry.strip():
            raise ValueError(f"input_paths[{index}] must be a non-blank path")
    resolved = [Path(p).resolve() for p in paths]
    seen: dict[Path, int] = {}
    for index, key in enumerate(resolved):
        if key in seen:
            raise ValueError(
                f"duplicate input path {str(paths[index])!r} "
                f"(same file as input #{seen[key] + 1}); "
                "pass each prepared file once"
            )
        seen[key] = index
    return [Path(p) for p in paths]


def merge_prepared_datasets(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Merge prepared project-schema datasets into one split corpus.

    Parameters
    ----------
    input_paths:
        At least 2 prepared dataset files (``.jsonl``/``.json``/``.csv``)
        in project schema. Each is read with the shared
        :func:`xssharden.dataset.io.read_records` as inert text;
        payloads are never executed. Every path must be distinct
        (compared by resolved absolute path) and must exist.
    output_path:
        Destination file written via :func:`write_records`
        (``.jsonl``/``.json``/``.csv``). Parent directories are created
        as needed — but only after all refusal checks pass, so a
        refused merge never creates a file.
    seed:
        Deterministic seed forwarded to :func:`split_records`.

    Returns
    -------
    dict:
        Summary with ``inputs`` (input paths as given), ``total_rows``
        (pre-deduplication rows read), ``per_input_counts`` (rows per
        input in order), ``duplicates_removed``, ``written``,
        ``sources`` (sorted distinct sources), ``per_source_counts``,
        ``splits`` (per-split counts), ``seed`` and ``output``.

    Raises
    ------
    TypeError
        For a non-sequence ``input_paths``, non-path entries, or a
        non-integer ``seed``.
    ValueError
        For fewer than 2 inputs, duplicate input paths, unsupported
        suffixes, schema violations, or fewer than 3 distinct
        ``source`` values across all inputs.
    FileNotFoundError
        If any input file does not exist.
    """
    seed = _check_seed(seed)
    if not isinstance(output_path, (str, Path)):
        raise TypeError(
            "output_path must be str or Path, "
            f"got {type(output_path).__name__}"
        )
    if isinstance(output_path, str) and not output_path.strip():
        raise ValueError("output_path must be a non-blank path")
    paths = _resolve_paths(input_paths)
    output_resolved = Path(output_path).resolve()
    if output_resolved in {path.resolve() for path in paths}:
        raise ValueError(
            f"output path {str(output_path)!r} must not overwrite an input dataset"
        )

    per_input_counts: list[int] = []
    combined: list[dict[str, Any]] = []
    for path in paths:
        records = read_records(path)
        per_input_counts.append(len(records))
        combined.extend(records)

    sources = sorted({str(r["source"]) for r in combined})
    if len(sources) < MIN_SOURCES:
        raise ValueError(
            f"dataset merge requires at least {MIN_SOURCES} distinct source "
            f"values, got {len(sources)} ({sources}); refusing to merge — "
            "add at least one more independent source before claiming "
            "meaningful train/validation/clean-test evaluation"
        )

    total_rows = len(combined)
    deduped = deduplicate_records(combined)
    duplicates_removed = total_rows - len(deduped)
    split_out = split_records(deduped, seed=seed)

    dst = Path(output_path)
    if str(dst.parent) and str(dst.parent) not in ("", "."):
        dst.parent.mkdir(parents=True, exist_ok=True)
    write_records(split_out, dst)

    per_source_counts = dict(Counter(str(r["source"]) for r in split_out))
    split_counts = dict(Counter(r["split"] for r in split_out))
    return {
        "inputs": [str(p) for p in paths],
        "total_rows": total_rows,
        "per_input_counts": per_input_counts,
        "duplicates_removed": duplicates_removed,
        "written": len(split_out),
        "sources": sources,
        "per_source_counts": per_source_counts,
        "splits": split_counts,
        "seed": seed,
        "output": str(dst),
    }
