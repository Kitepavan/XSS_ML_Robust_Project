"""Deterministic robustness report generation.

Renders a :class:`xssharden.evaluation.HardeningEvaluation` (or its plain
``to_dict()`` JSON form) to Markdown or HTML without inventing results:
every arm metric is serialized from the evaluation with its explicit
denominator, including the valid-only versus raw adversarial rates.

Payload strings from scored records are never rendered, so raw payloads
are not included by default. HTML output escapes untrusted metadata text
with the standard library only. Rendering is deterministic: arms follow
the canonical comparison order, metadata keys are sorted, and numbers
use fixed formatting, so the same input always yields byte-identical
output. Nothing here trains or scores detectors, touches the network,
or runs payloads anywhere.
"""

from __future__ import annotations

import copy
import html as _html
from pathlib import Path
from typing import Any

from xssharden.evaluation import ARM_NAMES as _CANONICAL_ARMS

#: Canonical hardening comparison arms, in report order.
ARM_NAMES: tuple[str, ...] = tuple(_CANONICAL_ARMS)

#: Supported report output suffixes.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".md", ".html")

#: Required clean-test metric keys per arm.
_CLEAN_KEYS: tuple[str, ...] = (
    "total",
    "positives",
    "negatives",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "detector_threshold",
)

#: Required adversarial metric keys per arm.
_ADV_KEYS: tuple[str, ...] = (
    "total_records",
    "valid_records",
    "malicious_count",
    "evasion_count",
    "evasion_rate",
    "valid_malicious_count",
    "valid_evasion_count",
    "valid_malicious_evasion_rate",
    "raw_evasion_rate",
    "detector_threshold",
)

_CLEAN_COUNT_KEYS: tuple[str, ...] = (
    "total",
    "positives",
    "negatives",
    "tp",
    "tn",
    "fp",
    "fn",
)

_ADV_COUNT_KEYS: tuple[str, ...] = (
    "total_records",
    "valid_records",
    "malicious_count",
    "evasion_count",
    "valid_malicious_count",
    "valid_evasion_count",
)

_CAVEAT_TEXT = (
    "Caveat: the current two-source data (http_params_dataset plus "
    "kaggle_xss_dataset, 2 distinct sources) cannot support meaningful "
    "three-way leakage-safe train/validation/clean-test evaluation under "
    "group-atomic splitting. Clean-test and adversarial claims stay "
    "preliminary until a third independent source is added."
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def _check_metrics(metrics: Any, *, keys: tuple[str, ...], label: str) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise TypeError(f"{label} must be a dict, got {type(metrics).__name__}")
    missing = [key for key in keys if key not in metrics]
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")
    checked: dict[str, Any] = {}
    for key in keys:
        value = metrics[key]
        if key in _CLEAN_COUNT_KEYS or key in _ADV_COUNT_KEYS:
            if not _is_int(value) or value < 0:
                raise ValueError(
                    f"{label}[{key!r}] must be a non-negative int, got {value!r}"
                )
            checked[key] = int(value)
        else:
            if not _is_number(value):
                raise TypeError(
                    f"{label}[{key!r}] must be a number, got {value!r}"
                )
            checked[key] = float(value)
    return checked


def _normalize_metadata(metadata: Any) -> dict[str, str]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError(
            f"metadata must be a dict or None, got {type(metadata).__name__}"
        )
    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError(
                f"metadata keys must be str, got {type(key).__name__}"
            )
        if value is None:
            normalized[key] = ""
        elif isinstance(value, bool):
            normalized[key] = str(value)
        elif isinstance(value, (str, int, float)):
            normalized[key] = str(value)
        else:
            raise TypeError(
                f"metadata[{key!r}] must be str, int, float, bool, or None, "
                f"got {type(value).__name__}"
            )
    return normalized


def _normalize_evaluation(evaluation: Any) -> dict[str, Any]:
    """Validate the evaluation shape and return a detached plain dict.

    Accepts a ``HardeningEvaluation`` (anything exposing ``to_dict()``)
    or its plain-dict JSON form. Record lists are dropped so payload
    text can never leak into a report. The input is never mutated.
    """
    if evaluation is None:
        raise TypeError("evaluation must be a HardeningEvaluation or dict, got None")
    to_dict = getattr(evaluation, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
    elif isinstance(evaluation, dict):
        data = copy.deepcopy(evaluation)
    else:
        raise TypeError(
            "evaluation must be a HardeningEvaluation or dict, "
            f"got {type(evaluation).__name__}"
        )
    if not isinstance(data, dict):
        raise TypeError(
            f"evaluation to_dict() must return a dict, got {type(data).__name__}"
        )
    arms = data.get("arms", None)
    if not isinstance(arms, dict):
        raise ValueError("evaluation must contain an 'arms' dict")
    missing = [name for name in ARM_NAMES if name not in arms]
    if missing:
        raise ValueError(f"evaluation arms are missing: {missing}")
    unexpected = [name for name in arms if name not in ARM_NAMES]
    if unexpected:
        raise ValueError(f"evaluation has unexpected arms: {unexpected}")
    normalized_arms: dict[str, Any] = {}
    for name in ARM_NAMES:
        arm = arms[name]
        if not isinstance(arm, dict):
            raise TypeError(
                f"evaluation arms[{name!r}] must be a dict, "
                f"got {type(arm).__name__}"
            )
        clean = _check_metrics(
            arm.get("clean_metrics", None),
            keys=_CLEAN_KEYS,
            label=f"arms[{name!r}].clean_metrics",
        )
        adv = _check_metrics(
            arm.get("adversarial_metrics", None),
            keys=_ADV_KEYS,
            label=f"arms[{name!r}].adversarial_metrics",
        )
        clean_count = arm.get("clean_count", clean["total"])
        adv_count = arm.get("adversarial_count", adv["total_records"])
        if not _is_int(clean_count) or int(clean_count) < 0:
            raise ValueError(
                f"arms[{name!r}].clean_count must be a non-negative int, "
                f"got {clean_count!r}"
            )
        if not _is_int(adv_count) or int(adv_count) < 0:
            raise ValueError(
                f"arms[{name!r}].adversarial_count must be a non-negative int, "
                f"got {adv_count!r}"
            )
        normalized_arms[name] = {
            "arm": name,
            "clean_metrics": clean,
            "adversarial_metrics": adv,
            "clean_count": int(clean_count),
            "adversarial_count": int(adv_count),
        }
    for key in ("clean_test_count", "adversarial_count"):
        value = data.get(key, None)
        if not _is_int(value) or int(value) < 0:
            raise ValueError(
                f"evaluation[{key!r}] must be a non-negative int, got {value!r}"
            )
    clean_split = data.get("clean_split", "clean-test")
    adv_split = data.get("adv_split", "adv_test")
    for label, value in (("clean_split", clean_split), ("adv_split", adv_split)):
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"evaluation[{label!r}] must be a non-empty string, "
                f"got {value!r}"
            )
    return {
        "arms": normalized_arms,
        "clean_test_count": int(data["clean_test_count"]),
        "adversarial_count": int(data["adversarial_count"]),
        "clean_split": clean_split,
        "adv_split": adv_split,
    }


def _fmt_rate(value: float) -> str:
    return f"{float(value):.4f}"


def _fmt_threshold(value: float) -> str:
    return f"{float(value):.4f}"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(
    evaluation: Any, metadata: dict[str, Any] | None = None
) -> str:
    """Render the evaluation as a deterministic Markdown report string.

    Parameters
    ----------
    evaluation:
        A ``HardeningEvaluation`` or its plain-dict ``to_dict()`` form.
        Metrics are serialized as given without inventing results.
    metadata:
        Optional dict of experiment metadata (``str`` keys; ``str``,
        ``int``, ``float``, ``bool``, or ``None`` values). Sorted by
        key so output is deterministic.

    Returns
    -------
    str:
        The Markdown report. Raw payloads are never included.
    """
    data = _normalize_evaluation(evaluation)
    meta = _normalize_metadata(metadata)
    lines: list[str] = []
    lines.append("# XSSHarden Robustness Report")
    lines.append("")
    lines.append("## Experiment metadata")
    lines.append("")
    if meta:
        for key in sorted(meta):
            lines.append(f"- **{key}**: {meta[key]}")
    else:
        lines.append("No experiment metadata was provided.")
    lines.append("")
    lines.append("## Input counts")
    lines.append("")
    lines.append(f"- Clean-test records: {data['clean_test_count']}")
    lines.append(f"- Adversarial records: {data['adversarial_count']}")
    lines.append(f"- Clean split: {data['clean_split']}")
    lines.append(f"- Adversarial split: {data['adv_split']}")
    for name in ARM_NAMES:
        arm = data["arms"][name]
        lines.append(
            f"- Arm `{name}`: clean_count={arm['clean_count']}, "
            f"adversarial_count={arm['adversarial_count']}"
        )
    lines.append("")
    lines.append("## Arm comparison")
    lines.append("")
    lines.append(
        "| Arm | Clean acc | Clean precision | Clean recall | Clean F1 | "
        "Clean FPR | V-ASR valid_malicious_evasion_rate "
        "(valid evasions / valid malicious) | "
        "Raw raw_evasion_rate (evasions / total) | "
        "Evasion evasion_rate (evasions / malicious) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in ARM_NAMES:
        arm = data["arms"][name]
        clean = arm["clean_metrics"]
        adv = arm["adversarial_metrics"]
        lines.append(
            f"| {name} | {_fmt_rate(clean['accuracy'])} "
            f"| {_fmt_rate(clean['precision'])} "
            f"| {_fmt_rate(clean['recall'])} "
            f"| {_fmt_rate(clean['f1'])} "
            f"| {_fmt_rate(clean['fpr'])} "
            f"| {_fmt_rate(adv['valid_malicious_evasion_rate'])} "
            f"({adv['valid_evasion_count']}/{adv['valid_malicious_count']}) "
            f"| {_fmt_rate(adv['raw_evasion_rate'])} "
            f"({adv['evasion_count']}/{adv['total_records']}) "
            f"| {_fmt_rate(adv['evasion_rate'])} "
            f"({adv['evasion_count']}/{adv['malicious_count']}) |"
        )
    lines.append("")
    for name in ARM_NAMES:
        arm = data["arms"][name]
        clean = arm["clean_metrics"]
        adv = arm["adversarial_metrics"]
        lines.append(f"### Arm `{name}`")
        lines.append("")
        lines.append(
            f"- Clean: total={clean['total']}, positives={clean['positives']}, "
            f"negatives={clean['negatives']}, tp={clean['tp']}, "
            f"tn={clean['tn']}, fp={clean['fp']}, fn={clean['fn']}"
        )
        lines.append(
            f"- Clean rates: accuracy={_fmt_rate(clean['accuracy'])}, "
            f"precision={_fmt_rate(clean['precision'])}, "
            f"recall={_fmt_rate(clean['recall'])}, "
            f"f1={_fmt_rate(clean['f1'])}, "
            f"fpr={_fmt_rate(clean['fpr'])} "
            f"(fp={clean['fp']}/(fp+tn)={clean['fp'] + clean['tn']}), "
            f"threshold={_fmt_threshold(clean['detector_threshold'])}"
        )
        lines.append(
            f"- Adversarial: total_records={adv['total_records']}, "
            f"valid_records={adv['valid_records']}, "
            f"malicious_count={adv['malicious_count']}, "
            f"evasion_count={adv['evasion_count']}, "
            f"valid_malicious_count={adv['valid_malicious_count']}, "
            f"valid_evasion_count={adv['valid_evasion_count']}"
        )
        lines.append(
            f"- valid_malicious_evasion_rate (V-ASR)="
            f"{_fmt_rate(adv['valid_malicious_evasion_rate'])} "
            f"({adv['valid_evasion_count']}/{adv['valid_malicious_count']}); "
            f"raw_evasion_rate={_fmt_rate(adv['raw_evasion_rate'])} "
            f"({adv['evasion_count']}/{adv['total_records']}); "
            f"evasion_rate={_fmt_rate(adv['evasion_rate'])} "
            f"({adv['evasion_count']}/{adv['malicious_count']}); "
            f"threshold={_fmt_threshold(adv['detector_threshold'])}"
        )
        lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- Four required arms are compared at a fixed operating threshold: "
        "baseline (original training data only), naive (original plus all "
        "valid generated variants), random_valid (original plus a "
        "budget-matched random subset of valid variants), and selective "
        "(original plus validity-gated detector-impact-selected variants)."
    )
    lines.append(
        "- Validity is a hard gate: only behaviorally valid malicious "
        "variants count toward the valid attack success rate (V-ASR). "
        "Impact ranking uses I(x) = 1 - p_detector(x) over valid variants "
        "only; invalid variants are logged and counted, never selected."
    )
    lines.append(
        "- The realizability gap is reported explicitly: V-ASR "
        "(valid_malicious_evasion_rate over the valid malicious "
        "denominator) versus the raw evasion rate (over all records). "
        "Thresholds are calibrated on clean validation data and held "
        "fixed across arms."
    )
    lines.append(
        "- Evaluation is leakage-controlled: clean-test and adversarial "
        "inputs use held-out splits that never influenced training, "
        "selection, or threshold tuning."
    )
    lines.append("")
    lines.append("## Safety notes")
    lines.append("")
    lines.append(
        "- Payloads are treated as inert strings throughout reporting: "
        "nothing is rendered, executed, fetched, or sent anywhere, and "
        "raw payload text is excluded from this report by default."
    )
    lines.append(
        "- Browser validation, when run elsewhere in the pipeline, uses "
        "only the controlled local sandbox; reporting itself performs no "
        "validation and launches no browser."
    )
    lines.append("")
    lines.append("## Caveat and limitations")
    lines.append("")
    lines.append(f"- {_CAVEAT_TEXT}")
    lines.append(
        "- Negative results are reportable: selective augmentation "
        "matching the random-valid control would indicate that validity "
        "filtering and additional data matter more than the selection "
        "strategy."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML rendering (standard library only)
# ---------------------------------------------------------------------------


def _escape(value: Any) -> str:
    return _html.escape(str(value), quote=True)


def render_html(
    evaluation: Any, metadata: dict[str, Any] | None = None
) -> str:
    """Render the evaluation as a deterministic standalone HTML string.

    Only the standard library is used. Untrusted metadata text is
    escaped, and raw payloads are never included.
    """
    data = _normalize_evaluation(evaluation)
    meta = _normalize_metadata(metadata)
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append("<title>XSSHarden Robustness Report</title>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<h1>XSSHarden Robustness Report</h1>")
    parts.append("<h2>Experiment metadata</h2>")
    if meta:
        parts.append("<ul>")
        for key in sorted(meta):
            parts.append(
                f"<li><strong>{_escape(key)}</strong>: {_escape(meta[key])}</li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p>No experiment metadata was provided.</p>")
    parts.append("<h2>Input counts</h2>")
    parts.append("<ul>")
    parts.append(f"<li>Clean-test records: {data['clean_test_count']}</li>")
    parts.append(f"<li>Adversarial records: {data['adversarial_count']}</li>")
    parts.append(f"<li>Clean split: {_escape(data['clean_split'])}</li>")
    parts.append(
        f"<li>Adversarial split: {_escape(data['adv_split'])}</li>"
    )
    for name in ARM_NAMES:
        arm = data["arms"][name]
        parts.append(
            f"<li>Arm {_escape(name)}: "
            f"clean_count={arm['clean_count']}, "
            f"adversarial_count={arm['adversarial_count']}</li>"
        )
    parts.append("</ul>")
    parts.append("<h2>Arm comparison</h2>")
    parts.append("<table>")
    parts.append(
        "<tr><th>Arm</th><th>Clean acc</th><th>Clean precision</th>"
        "<th>Clean recall</th><th>Clean F1</th><th>Clean FPR</th>"
        "<th>V-ASR valid_malicious_evasion_rate "
        "(valid evasions / valid malicious)</th>"
        "<th>Raw raw_evasion_rate (evasions / total)</th>"
        "<th>Evasion evasion_rate (evasions / malicious)</th></tr>"
    )
    for name in ARM_NAMES:
        arm = data["arms"][name]
        clean = arm["clean_metrics"]
        adv = arm["adversarial_metrics"]
        parts.append(
            f"<tr><td>{_escape(name)}</td>"
            f"<td>{_fmt_rate(clean['accuracy'])}</td>"
            f"<td>{_fmt_rate(clean['precision'])}</td>"
            f"<td>{_fmt_rate(clean['recall'])}</td>"
            f"<td>{_fmt_rate(clean['f1'])}</td>"
            f"<td>{_fmt_rate(clean['fpr'])}</td>"
            f"<td>{_fmt_rate(adv['valid_malicious_evasion_rate'])} "
            f"({adv['valid_evasion_count']}/{adv['valid_malicious_count']})"
            f"</td>"
            f"<td>{_fmt_rate(adv['raw_evasion_rate'])} "
            f"({adv['evasion_count']}/{adv['total_records']})</td>"
            f"<td>{_fmt_rate(adv['evasion_rate'])} "
            f"({adv['evasion_count']}/{adv['malicious_count']})</td></tr>"
        )
    parts.append("</table>")
    for name in ARM_NAMES:
        arm = data["arms"][name]
        clean = arm["clean_metrics"]
        adv = arm["adversarial_metrics"]
        parts.append(f"<h3>Arm {_escape(name)}</h3>")
        parts.append("<ul>")
        parts.append(
            f"<li>Clean: total={clean['total']}, "
            f"positives={clean['positives']}, "
            f"negatives={clean['negatives']}, tp={clean['tp']}, "
            f"tn={clean['tn']}, fp={clean['fp']}, fn={clean['fn']}</li>"
        )
        parts.append(
            f"<li>Clean rates: accuracy={_fmt_rate(clean['accuracy'])}, "
            f"precision={_fmt_rate(clean['precision'])}, "
            f"recall={_fmt_rate(clean['recall'])}, "
            f"f1={_fmt_rate(clean['f1'])}, "
            f"fpr={_fmt_rate(clean['fpr'])}, "
            f"threshold={_fmt_threshold(clean['detector_threshold'])}</li>"
        )
        parts.append(
            f"<li>Adversarial: total_records={adv['total_records']}, "
            f"valid_records={adv['valid_records']}, "
            f"malicious_count={adv['malicious_count']}, "
            f"evasion_count={adv['evasion_count']}, "
            f"valid_malicious_count={adv['valid_malicious_count']}, "
            f"valid_evasion_count={adv['valid_evasion_count']}</li>"
        )
        parts.append(
            f"<li>valid_malicious_evasion_rate (V-ASR)="
            f"{_fmt_rate(adv['valid_malicious_evasion_rate'])} "
            f"({adv['valid_evasion_count']}/{adv['valid_malicious_count']}); "
            f"raw_evasion_rate={_fmt_rate(adv['raw_evasion_rate'])} "
            f"({adv['evasion_count']}/{adv['total_records']}); "
            f"evasion_rate={_fmt_rate(adv['evasion_rate'])} "
            f"({adv['evasion_count']}/{adv['malicious_count']}); "
            f"threshold={_fmt_threshold(adv['detector_threshold'])}</li>"
        )
        parts.append("</ul>")
    parts.append("<h2>Methodology</h2>")
    parts.append(
        "<p>Four required arms are compared at a fixed operating threshold: "
        "baseline, naive all-valid, budget-matched random-valid, and "
        "validity-gated detector-impact-selected augmentation. Validity is "
        "a hard gate; the realizability gap (V-ASR over the valid malicious "
        "denominator versus the raw evasion rate) is reported with explicit "
        "denominators. Held-out splits never influence training, selection, "
        "or threshold tuning.</p>"
    )
    parts.append("<h2>Safety notes</h2>")
    parts.append(
        "<p>Payloads are treated as inert strings throughout reporting: "
        "nothing is rendered, executed, fetched, or sent anywhere, and raw "
        "payload text is excluded from this report by default.</p>"
    )
    parts.append("<h2>Caveat and limitations</h2>")
    parts.append(f"<p>{_escape(_CAVEAT_TEXT)}</p>")
    parts.append(
        "<p>Negative results are reportable: selective augmentation "
        "matching the random-valid control would indicate that validity "
        "filtering and additional data matter more than the selection "
        "strategy.</p>"
    )
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def _check_output_suffix(output_path: Any) -> str:
    if not isinstance(output_path, (str, Path)):
        raise TypeError(
            "output_path must be a str or Path, "
            f"got {type(output_path).__name__}"
        )
    suffix = Path(str(output_path)).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported output suffix {suffix!r}: "
            "expected '.md' or '.html'"
        )
    return suffix


def write_report(
    evaluation: Any,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate, render, and atomically write a robustness report file.

    Parameters
    ----------
    evaluation:
        A ``HardeningEvaluation`` or its plain-dict ``to_dict()`` form.
    output_path:
        Destination path ending in ``.md`` (Markdown) or ``.html``.
        Anything else is refused and no file is created.
    metadata:
        Optional experiment metadata dict (see :func:`render_markdown`).

    Returns
    -------
    dict:
        Summary with ``output``, ``format`` (``"markdown"``/``"html"``),
        ``arms``, ``clean_test_count``, and ``adversarial_count``.

    Validation happens before any file is created, and a failed write
    removes any partial file, so refused or interrupted reports never
    leave partial outputs behind.
    """
    suffix = _check_output_suffix(output_path)
    data = _normalize_evaluation(evaluation)
    meta = _normalize_metadata(metadata)
    if suffix == ".md":
        text = render_markdown(data, metadata=meta)
        kind = "markdown"
    else:
        text = render_html(data, metadata=meta)
        kind = "html"
    target = Path(str(output_path))
    tmp_path = target.with_name(target.name + ".tmp")
    try:
        if target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(target)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return {
        "output": str(output_path),
        "format": kind,
        "arms": list(ARM_NAMES),
        "clean_test_count": data["clean_test_count"],
        "adversarial_count": data["adversarial_count"],
    }
