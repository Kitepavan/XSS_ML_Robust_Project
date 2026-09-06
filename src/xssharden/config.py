"""Experiment ID configuration for XSSHarden."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def make_experiment_id(config: dict[str, Any], seed: int = 42) -> str:
    """Return a canonical stable hex ID for an experiment.

    The ID is the SHA-256 hash of a canonical JSON encoding (sorted keys,
    compact separators) of ``{"config": config, "seed": seed}``. Key order
    in ``config`` therefore does not affect the result, while any change to
    values or to ``seed`` changes the ID. Nested dicts/lists are supported.
    """
    canonical = json.dumps(
        {"config": config, "seed": seed},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
