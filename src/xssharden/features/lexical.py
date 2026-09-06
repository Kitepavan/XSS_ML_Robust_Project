"""Deterministic lexical/handcrafted structural features for XSS payloads.

This module is the feature extractor for Detector 2 (XGBoost on lexical +
handcrafted features). Every feature is a pure function of the payload
string: payloads are treated as inert text, never executed, and no browser,
validator, network, or model code is imported here.

The public entry point is :func:`featurize_lexical`, which accepts either
plain payload strings or record dicts carrying a ``"payload"`` key and
returns a dense ``numpy`` matrix with one row per input and one column per
name in :data:`FEATURE_NAMES` (fixed order). Extraction is fully
deterministic: the same inputs always produce the same matrix, independent
of input order, and no fitted state is required.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

# Fixed feature order shared by fit and inference. Names are stable API:
# detectors persist/interpret columns positionally, so never reorder or
# remove entries — only append with a version note.
FEATURE_NAMES: tuple[str, ...] = (
    "length",
    "n_lt",
    "n_gt",
    "n_tags",
    "n_script",
    "n_javascript",
    "n_event_handlers",
    "n_pct",
    "n_pct_encoded",
    "n_html_entity",
    "n_hex_escape",
    "n_single_quote",
    "n_double_quote",
    "n_backtick",
    "n_paren_open",
    "n_paren_close",
    "n_semicolon",
    "n_equals",
    "n_slash",
    "n_colon",
    "n_whitespace",
    "whitespace_ratio",
    "n_digits",
    "unique_char_ratio",
    "shannon_entropy",
)

_TAG_RE = re.compile(r"<[a-zA-Z/!?]")
_EVENT_HANDLER_RE = re.compile(r"on[a-z]+\s*=")
_PCT_ENCODED_RE = re.compile(r"%[0-9a-fA-F]{2}")
_HTML_ENTITY_RE = re.compile(r"&(#\d+|#[xX][0-9a-fA-F]+|[a-zA-Z]+;)")

Record = dict[str, Any]


def _shannon_entropy(payload: str) -> float:
    """Return the Shannon entropy (bits) of the payload's character distribution."""
    if not payload:
        return 0.0
    length = len(payload)
    counts: dict[str, int] = {}
    for char in payload:
        counts[char] = counts.get(char, 0) + 1
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def extract_features(payload: str) -> dict[str, float]:
    """Extract the ordered handcrafted feature mapping for one payload string.

    Parameters
    ----------
    payload:
        Raw payload text, treated as inert data (never executed).

    Returns
    -------
    dict
        Mapping of every name in :data:`FEATURE_NAMES` to a finite,
        non-negative float, in canonical order.
    """
    if not isinstance(payload, str):
        raise TypeError(f"payload must be str, got {type(payload).__name__}")
    lowered = payload.lower()
    length = len(payload)
    denom = max(1, length)
    whitespace = sum(1 for char in payload if char.isspace())
    unique = len(set(payload))
    features: dict[str, float] = {
        "length": float(length),
        "n_lt": float(payload.count("<")),
        "n_gt": float(payload.count(">")),
        "n_tags": float(len(_TAG_RE.findall(payload))),
        "n_script": float(lowered.count("script")),
        "n_javascript": float(lowered.count("javascript")),
        "n_event_handlers": float(len(_EVENT_HANDLER_RE.findall(lowered))),
        "n_pct": float(payload.count("%")),
        "n_pct_encoded": float(len(_PCT_ENCODED_RE.findall(payload))),
        "n_html_entity": float(len(_HTML_ENTITY_RE.findall(payload))),
        "n_hex_escape": float(payload.count("\\x") + payload.count("\\u")),
        "n_single_quote": float(payload.count("'")),
        "n_double_quote": float(payload.count('"')),
        "n_backtick": float(payload.count("`")),
        "n_paren_open": float(payload.count("(")),
        "n_paren_close": float(payload.count(")")),
        "n_semicolon": float(payload.count(";")),
        "n_equals": float(payload.count("=")),
        "n_slash": float(payload.count("/")),
        "n_colon": float(payload.count(":")),
        "n_whitespace": float(whitespace),
        "whitespace_ratio": float(whitespace / denom),
        "n_digits": float(sum(1 for char in payload if char.isdigit())),
        "unique_char_ratio": float(unique / denom),
        "shannon_entropy": float(_shannon_entropy(payload)),
    }
    assert tuple(features.keys()) == FEATURE_NAMES
    return features


def _payloads_from_inputs(inputs: Sequence[str] | Sequence[Record]) -> list[str]:
    """Extract raw payload strings from record dicts or plain strings."""
    if inputs is None or len(inputs) == 0:  # type: ignore[arg-type]
        raise ValueError("inputs must be a non-empty sequence")
    first = inputs[0]
    if isinstance(first, dict):
        payloads: list[str] = []
        for index, record in enumerate(inputs):  # type: ignore[union-attr]
            if not isinstance(record, dict):
                raise TypeError(
                    f"record at index {index} must be dict, "
                    f"got {type(record).__name__}"
                )
            payload = record.get("payload")
            if not isinstance(payload, str):
                raise TypeError(
                    f"record at index {index}: 'payload' must be str, "
                    f"got {type(payload).__name__}"
                )
            payloads.append(payload)
        return payloads
    payloads = []
    for index, item in enumerate(inputs):  # type: ignore[union-attr]
        if not isinstance(item, str):
            raise TypeError(
                f"payload at index {index} must be str, "
                f"got {type(item).__name__}"
            )
        payloads.append(item)
    if len(payloads) == 0:
        raise ValueError("inputs must be a non-empty sequence")
    return payloads


def featurize_lexical(inputs: Sequence[str] | Sequence[Record]) -> np.ndarray:
    """Featurize payloads into a dense ``(n, len(FEATURE_NAMES))`` matrix.

    Accepts plain strings or record dicts with a ``"payload"`` key (both
    forms produce identical rows). The output is deterministic in values
    and row order, requires no fitted state, and never executes payloads.

    Raises
    ------
    ValueError
        If *inputs* is empty.
    TypeError
        If any payload is not a string.
    """
    payloads = _payloads_from_inputs(inputs)
    matrix = np.empty((len(payloads), len(FEATURE_NAMES)), dtype=np.float64)
    for i, payload in enumerate(payloads):
        features = extract_features(payload)
        matrix[i] = [features[name] for name in FEATURE_NAMES]
    return matrix
