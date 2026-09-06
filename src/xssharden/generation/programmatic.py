"""Deterministic programmatic XSS variant generation.

The engine exposes :func:`generate_variants`, which accepts seed records
(each with at least a ``sample_id`` and a ``payload``) and produces variant
records carrying full seed provenance.

Payloads are treated as inert text throughout: every mutation is a pure
string transformation defined below. Nothing is rendered, run, fetched, or
sent anywhere. This module depends only on the standard library and never
touches the browser validator, the detectors, or the network.

Supported mutation categories (auditable, deterministic):

- ``encoding``: URL-encoding and HTML-entity encoding of markup characters.
- ``whitespace_comment``: whitespace and comment-token insertion.
- ``tag_event_substitution``: tag / event-handler token substitution.
- ``case_variation``: casing transformations.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Iterable, Sequence

GENERATOR_NAME = "programmatic"
GENERATOR_VERSION = "programmatic-v1"

ENCODING_CATEGORY = "encoding"
WHITESPACE_COMMENT_CATEGORY = "whitespace_comment"
TAG_EVENT_CATEGORY = "tag_event_substitution"
CASE_CATEGORY = "case_variation"

SUPPORTED_CATEGORIES: tuple[str, ...] = (
    ENCODING_CATEGORY,
    WHITESPACE_COMMENT_CATEGORY,
    TAG_EVENT_CATEGORY,
    CASE_CATEGORY,
)

VARIANT_SPLIT = "adv_dev"
DEFAULT_CONTEXT_TARGET = "reflected_html"

ALLOWED_SEED_SPLITS: tuple[str, ...] = ("train", "adv_dev")

_VARIANT_ID_PREFIX = "var-"
_VARIANT_ID_HEX_CHARS = 16


# ---------------------------------------------------------------------------
# Pure string transforms (one category at a time)
# ---------------------------------------------------------------------------


def _encoding_candidates(payload: str) -> list[str]:
    """Return encoding rewrites of *payload* in a fixed order."""
    candidates: list[str] = []
    if "<" in payload or ">" in payload:
        candidates.append(payload.replace("<", "%3C").replace(">", "%3E"))
        candidates.append(payload.replace("<", "&lt;").replace(">", "&gt;"))
        hexed = (
            payload.replace("<", "&#x3C;")
            .replace(">", "&#x3E;")
            .replace('"', "&#x22;")
            .replace("'", "&#x27;")
            .replace("(", "&#x28;")
            .replace(")", "&#x29;")
        )
        candidates.append(hexed)
    if "(" in payload or ")" in payload:
        candidates.append(payload.replace("(", "&#40;").replace(")", "&#41;"))
    return candidates


def _whitespace_comment_candidates(payload: str) -> list[str]:
    """Return whitespace/comment-insertion rewrites in a fixed order."""
    candidates: list[str] = []
    if "<" in payload:
        candidates.append(payload.replace("<", "</**/"))
    if " " in payload:
        candidates.append(payload.replace(" ", "/**/"))
        candidates.append(payload.replace(" ", "\n"))
        candidates.append(payload.replace(" ", "\t"))
    if ">" in payload:
        candidates.append(payload.replace(">", "<!--x-->"))
    return candidates


_TAG_EVENT_RULES: tuple[tuple[str, str], ...] = (
    (r"script", "svg"),
    (r"<img", "<svg"),
    (r"onerror", "onload"),
    (r"onclick", "onmouseover"),
    (r"onload", "onerror"),
    (r"alert", "prompt"),
    (r"confirm", "prompt"),
)


def _tag_event_candidates(payload: str) -> list[str]:
    """Return tag/event-handler substitution rewrites in a fixed order."""
    candidates: list[str] = []
    for pattern, replacement in _TAG_EVENT_RULES:
        rewritten, count = re.subn(pattern, replacement, payload, flags=re.IGNORECASE)
        if count and rewritten != payload:
            candidates.append(rewritten)
    return candidates


def _alternating_case(payload: str) -> str:
    """Upper-case even positions and lower-case odd positions."""
    return "".join(
        char.upper() if index % 2 == 0 else char.lower()
        for index, char in enumerate(payload)
    )


def _case_candidates(payload: str) -> list[str]:
    """Return casing rewrites of *payload* in a fixed order."""
    return [
        payload.upper(),
        payload.lower(),
        _alternating_case(payload),
        payload.swapcase(),
    ]


_CATEGORY_FUNCS = {
    ENCODING_CATEGORY: _encoding_candidates,
    WHITESPACE_COMMENT_CATEGORY: _whitespace_comment_candidates,
    TAG_EVENT_CATEGORY: _tag_event_candidates,
    CASE_CATEGORY: _case_candidates,
}


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _check_random_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"seed must be int, got {type(value).__name__}")
    return value


def _check_limit(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int or None, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return value


def _check_categories(categories: Any) -> list[str]:
    if categories is None:
        return list(SUPPORTED_CATEGORIES)
    if isinstance(categories, str) or not isinstance(categories, Sequence):
        raise TypeError(
            "categories must be a sequence of str or None, "
            f"got {type(categories).__name__}"
        )
    selected = list(categories)
    if not selected:
        raise ValueError("categories must not be empty")
    for entry in selected:
        if not isinstance(entry, str):
            raise TypeError(
                f"category names must be str, got {type(entry).__name__}"
            )
        if entry not in SUPPORTED_CATEGORIES:
            raise ValueError(
                f"unsupported mutation category {entry!r}: "
                f"expected one of {list(SUPPORTED_CATEGORIES)}"
            )
    # De-duplicate while preserving the requested order.
    return list(dict.fromkeys(selected))


def _check_allowed_splits(value: Any) -> tuple[str, ...]:
    if value is None:
        return ALLOWED_SEED_SPLITS
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            "allowed_splits must be a sequence of str or None, "
            f"got {type(value).__name__}"
        )
    items = list(value)
    if not items:
        raise ValueError("allowed_splits must not be empty")
    for entry in items:
        if not isinstance(entry, str):
            raise TypeError(
                f"allowed split names must be str, got {type(entry).__name__}"
            )
    return tuple(items)


def _materialize_seeds(seeds: Any) -> list[dict[str, Any]]:
    if seeds is None:
        raise TypeError("seeds must be an iterable of dicts, got None")
    if isinstance(seeds, (dict, str, bytes, bytearray)):
        raise TypeError(
            "seeds must be an iterable of dicts "
            f"(e.g. list), got {type(seeds).__name__}"
        )
    try:
        items = list(seeds)
    except TypeError as exc:
        raise TypeError(
            "seeds must be an iterable of dicts, "
            f"got {type(seeds).__name__}"
        ) from exc
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"seed index {index}: must be a dict, "
                f"got {type(item).__name__}"
            )
        payload = item.get("payload")
        if "payload" not in item:
            raise ValueError(f"seed index {index}: missing required field 'payload'")
        if not isinstance(payload, str):
            raise TypeError(
                f"seed index {index}: field 'payload' must be str, "
                f"got {type(payload).__name__}"
            )
        if not payload.strip():
            raise ValueError(
                f"seed index {index}: 'payload' must be a non-empty string"
            )
        sample_id = item.get("sample_id")
        if "sample_id" not in item:
            raise ValueError(
                f"seed index {index}: missing required field 'sample_id'"
            )
        if not isinstance(sample_id, str):
            raise TypeError(
                f"seed index {index}: field 'sample_id' must be str, "
                f"got {type(sample_id).__name__}"
            )
        if not sample_id.strip():
            raise ValueError(
                f"seed index {index}: 'sample_id' must be a non-empty string"
            )
    return items


def _variant_id(seed_id: str, category: str, payload: str) -> str:
    """Return a deterministic content-based variant ID.

    The digest covers ``seed_id|mutation_category|payload`` only, so IDs are
    stable across runs, orderings, and sampling seeds.
    """
    digest = hashlib.sha256(
        f"{seed_id}|{category}|{payload}".encode("utf-8")
    ).hexdigest()[:_VARIANT_ID_HEX_CHARS]
    return f"{_VARIANT_ID_PREFIX}{digest}"


def generate_variants(
    seeds: Iterable[dict[str, Any]],
    *,
    categories: Sequence[str] | None = None,
    seed: int = 42,
    max_per_seed: int | None = None,
    max_per_category: int | None = None,
    enforce_split_boundary: bool = True,
    allowed_splits: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate deterministic variant records from *seeds*.

    Parameters
    ----------
    seeds:
        Iterable of dicts, each with at least a non-empty ``payload`` and a
        non-empty ``sample_id``. Extra provenance (``label``,
        ``attack_category``, ``source``, ``context_target``, ``split``) is
        preserved in each variant. Inputs are never mutated.
    categories:
        Subset of ``SUPPORTED_CATEGORIES`` to apply, in the requested order.
        ``None`` (the default) applies every supported category in canonical
        order. Unknown names raise ``ValueError``.
    seed:
        Deterministic random state used only when ``max_per_seed`` or
        ``max_per_category`` truncation must choose between candidates.
        Identical inputs with the same ``seed`` always yield identical
        outputs. No LLM or browser is involved.
    max_per_seed:
        Optional cap (``>= 1``) on variants kept per seed.
    max_per_category:
        Optional cap (``>= 1``) on variants kept per mutation category.
    enforce_split_boundary:
        When ``True`` (the default), seeds whose ``split`` is present and
        outside ``allowed_splits`` (e.g. ``validation``, ``clean-test``,
        ``test``) raise ``ValueError`` so held-out records can never leak
        into generated training data. Set to ``False`` only to explicitly
        opt out of that guard.
    allowed_splits:
        Optional override of the accepted seed splits
        (default ``("train", "adv_dev")``).

    Returns
    -------
    list[dict]:
        Variant records in stable generation order (seed input order, then
        requested category order, then transform order). Each record holds
        ``variant_id``, ``seed_id``, ``payload``, ``mutation_category``,
        ``context_target``, ``source``, ``generator``,
        ``generator_version``, ``split`` (``"adv_dev"``), plus ``label`` /
        ``attack_category`` whenever the seed carried them. Variant
        payloads never repeat a seed payload and never repeat each other.
    """
    random_state = _check_random_seed(seed)
    selected_categories = _check_categories(categories)
    per_seed_cap = _check_limit("max_per_seed", max_per_seed)
    per_category_cap = _check_limit("max_per_category", max_per_category)
    accepted_splits = _check_allowed_splits(allowed_splits)

    items = _materialize_seeds(seeds)

    if enforce_split_boundary:
        for index, item in enumerate(items):
            split = item.get("split")
            if split is not None and split not in accepted_splits:
                raise ValueError(
                    f"seed index {index}: split {split!r} is not allowed "
                    f"for generation (allowed: {list(accepted_splits)}); "
                    "refusing to generate variants from held-out records"
                )

    rng = random.Random(random_state)
    seed_payloads = {item["payload"] for item in items}

    # Per-seed candidates in (category, payload) order, de-duplicated and
    # capped before global assembly so limits are deterministic.
    per_seed: list[list[tuple[str, str]]] = []
    for item in items:
        seen: set[str] = set()
        ordered: list[tuple[str, str]] = []
        for category in selected_categories:
            for candidate in _CATEGORY_FUNCS[category](item["payload"]):
                if candidate == item["payload"]:
                    continue
                if candidate in seed_payloads or candidate in seen:
                    continue
                seen.add(candidate)
                ordered.append((category, candidate))
        if per_seed_cap is not None and len(ordered) > per_seed_cap:
            picked = rng.sample(ordered, per_seed_cap)
            keep = set(picked)
            ordered = [entry for entry in ordered if entry in keep]
        per_seed.append(ordered)

    # Global assembly in seed input order with exact-payload deduplication
    # (first occurrence wins).
    assembled: list[tuple[int, str, str]] = []
    seen_global: set[str] = set()
    for seed_index, ordered in enumerate(per_seed):
        for category, candidate in ordered:
            if candidate in seen_global:
                continue
            seen_global.add(candidate)
            assembled.append((seed_index, category, candidate))

    # Global per-category cap, restoring generation order afterwards.
    if per_category_cap is not None:
        kept: list[tuple[int, str, str]] = []
        for category in selected_categories:
            matching = [entry for entry in assembled if entry[1] == category]
            if len(matching) > per_category_cap:
                picked = rng.sample(matching, per_category_cap)
                keep = set(picked)
                matching = [entry for entry in matching if entry in keep]
            kept.extend(matching)
        kept.sort(key=lambda entry: assembled.index(entry))
        assembled = kept

    variants: list[dict[str, Any]] = []
    for seed_index, category, candidate in assembled:
        item = items[seed_index]
        sample_id = item["sample_id"]
        context = item.get("context_target")
        if not isinstance(context, str) or not context:
            context = DEFAULT_CONTEXT_TARGET
        source = item.get("source", "unknown")
        if not isinstance(source, str) or not source:
            source = "unknown"
        record: dict[str, Any] = {
            "variant_id": _variant_id(sample_id, category, candidate),
            "seed_id": sample_id,
            "payload": candidate,
            "mutation_category": category,
            "context_target": context,
            "source": source,
            "seed_source": source,
            "generator": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "split": VARIANT_SPLIT,
        }
        if "split" in item:
            record["seed_split"] = item["split"]
        if "label" in item:
            record["label"] = item["label"]
        if "attack_category" in item:
            record["attack_category"] = item["attack_category"]
        variants.append(record)
    return variants
