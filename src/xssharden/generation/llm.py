"""Provider-agnostic cloud-LLM XSS variant generation adapter.

This module talks to any OpenAI-compatible ``/chat/completions`` HTTP
endpoint using only the Python standard library. It never assumes a local
model server: every credential comes from explicit constructor arguments or
from the documented ``XSSHARDEN_LLM_*`` environment variables, and nothing
is sent anywhere unless the caller passes ``allow_payload_submission=True``.

Model output is treated as untrusted text throughout. Responses must match
a strict JSON contract (``{"variants": [...]}``); anything else is rejected
with a typed error and preserved in the auditable ``errors`` list of the
result instead of being silently accepted.

Nothing here executes payloads, renders HTML, launches a browser, or
imports the browser validator. Payloads stay inert strings from request
construction through response parsing and deduplication.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from xssharden.generation.programmatic import SUPPORTED_CATEGORIES

GENERATOR_NAME = "llm"
GENERATOR_VERSION = "llm-v1"

VARIANT_SPLIT = "adv_dev"
DEFAULT_CONTEXT_TARGET = "reflected_html"
ALLOWED_SEED_SPLITS: tuple[str, ...] = ("train", "adv_dev")

ENV_BASE_URL = "XSSHARDEN_LLM_BASE_URL"
ENV_API_KEY = "XSSHARDEN_LLM_API_KEY"
ENV_MODEL = "XSSHARDEN_LLM_MODEL"
ENV_TIMEOUT_S = "XSSHARDEN_LLM_TIMEOUT_S"
ENV_ORGANIZATION = "XSSHARDEN_LLM_ORGANIZATION"
ENV_PROJECT = "XSSHARDEN_LLM_PROJECT"

DEFAULT_TIMEOUT_S = 30.0

_VARIANT_ID_PREFIX = "llm-"
_VARIANT_ID_HEX_CHARS = 16


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for every error raised by this module."""


class LLMConfigurationError(LLMError, ValueError):
    """Raised when endpoint configuration is missing or malformed."""


class LLMPayloadSubmissionNotAllowedError(LLMError, PermissionError):
    """Raised when generation is attempted without explicit opt-in.

    Payload text is never sent to a cloud API unless the caller passes
    ``allow_payload_submission=True``. This error is raised before any
    network request is attempted.
    """


class LLMTransportError(LLMError, RuntimeError):
    """Raised when the HTTP request fails (connection, status, I/O)."""


class LLMTimeoutError(LLMTransportError, TimeoutError):
    """Raised when the HTTP request times out."""


class LLMResponseError(LLMError, ValueError):
    """Raised when a model response violates the strict JSON contract."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _require_non_blank(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LLMConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class LLMConfig:
    """Connection settings for an OpenAI-compatible chat endpoint.

    Attributes
    ----------
    base_url:
        API root such as ``https://api.example.com/v1`` (no path suffix
        assumed; ``/chat/completions`` is appended automatically).
    model:
        Model identifier sent in the request body.
    api_key:
        Secret sent as a ``Bearer`` token. Never hardcoded; pass it
        explicitly or via ``XSSHARDEN_LLM_API_KEY``.
    timeout_s:
        Per-request timeout in seconds (must be positive).
    organization:
        Optional ``OpenAI-Organization`` header value.
    project:
        Optional ``OpenAI-Project`` header value.
    """

    base_url: str
    model: str
    api_key: str
    timeout_s: float = DEFAULT_TIMEOUT_S
    organization: str | None = None
    project: str | None = None

    def __post_init__(self) -> None:
        base_url = _require_non_blank("base_url", self.base_url)
        if not base_url.startswith(("http://", "https://")):
            raise LLMConfigurationError(
                "base_url must start with 'http://' or 'https://', "
                f"got {self.base_url!r}"
            )
        _require_non_blank("model", self.model)
        _require_non_blank("api_key", self.api_key)
        timeout = self.timeout_s
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise LLMConfigurationError(
                f"timeout_s must be a positive number, got {timeout!r}"
            )
        if not timeout > 0:
            raise LLMConfigurationError(
                f"timeout_s must be a positive number, got {timeout!r}"
            )
        for name in ("organization", "project"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise LLMConfigurationError(f"{name} must be a non-empty string or None")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build a config from the documented environment variables.

        Reads ``XSSHARDEN_LLM_BASE_URL``, ``XSSHARDEN_LLM_API_KEY``,
        ``XSSHARDEN_LLM_MODEL``, and the optional
        ``XSSHARDEN_LLM_TIMEOUT_S`` / ``XSSHARDEN_LLM_ORGANIZATION`` /
        ``XSSHARDEN_LLM_PROJECT``. There are no defaults for the URL,
        key, or model (in particular nothing points at a local server);
        missing values raise :class:`LLMConfigurationError`.
        """
        base_url = os.environ.get(ENV_BASE_URL, "").strip()
        api_key = os.environ.get(ENV_API_KEY, "").strip()
        model = os.environ.get(ENV_MODEL, "").strip()
        missing = [
            var
            for var, val in (
                (ENV_BASE_URL, base_url),
                (ENV_API_KEY, api_key),
                (ENV_MODEL, model),
            )
            if not val
        ]
        if missing:
            raise LLMConfigurationError(
                "missing required LLM configuration: "
                + ", ".join(missing)
                + " (set the environment variables or pass explicit arguments)"
            )
        timeout_s: float = DEFAULT_TIMEOUT_S
        raw_timeout = os.environ.get(ENV_TIMEOUT_S, "").strip()
        if raw_timeout:
            try:
                timeout_s = float(raw_timeout)
            except (TypeError, ValueError) as exc:
                raise LLMConfigurationError(
                    f"{ENV_TIMEOUT_S} must be a positive number, got {raw_timeout!r}"
                ) from exc
        organization = os.environ.get(ENV_ORGANIZATION, "").strip() or None
        project = os.environ.get(ENV_PROJECT, "").strip() or None
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            organization=organization,
            project=project,
        )

    @property
    def endpoint_url(self) -> str:
        """Return the full ``/chat/completions`` endpoint URL."""
        return self.base_url.rstrip("/") + "/chat/completions"

    def headers(self) -> dict[str, str]:
        """Return the HTTP headers for an API request."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers


# ---------------------------------------------------------------------------
# Transport (standard library only; injectable for tests)
# ---------------------------------------------------------------------------

#: Callable invoked as ``transport(url, headers, body, timeout_s)`` that
#: must return the parsed JSON response object.
TransportFn = Callable[[str, dict[str, str], dict[str, Any], float], Any]


def _default_transport(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
) -> Any:
    """POST *body* as JSON to *url* with ``urllib`` and return parsed JSON."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = getattr(response, "status", 200)
            raw = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise LLMTimeoutError(f"LLM request timed out after {timeout_s}s: {exc}") from exc
    except urllib.error.HTTPError as exc:
        raise LLMTransportError(
            f"LLM request failed with HTTP status {exc.code}: {exc.reason}"
        ) from exc
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        raise LLMTransportError(f"LLM request failed: {exc}") from exc
    if isinstance(status, int) and not 200 <= status < 300:
        raise LLMTransportError(f"LLM request failed with HTTP status {status}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM endpoint returned invalid JSON: {exc}") from exc


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client over an injectable transport."""

    def __init__(self, config: LLMConfig, transport: TransportFn | None = None) -> None:
        if not isinstance(config, LLMConfig):
            raise LLMConfigurationError(
                f"config must be an LLMConfig, got {type(config).__name__}"
            )
        self._config = config
        self._transport = transport if transport is not None else _default_transport

    @property
    def config(self) -> LLMConfig:
        return self._config

    @property
    def model(self) -> str:
        return self._config.model

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send chat *messages* and return the parsed response object.

        Raises
        ------
        LLMConfigurationError
            If *messages* is not a non-empty list of dicts.
        LLMTimeoutError
            If the request times out.
        LLMTransportError
            If the request fails at the HTTP layer.
        LLMResponseError
            If the transport returns a non-dict response object.
        """
        if not isinstance(messages, list) or not messages or not all(
            isinstance(item, dict) for item in messages
        ):
            raise LLMConfigurationError("messages must be a non-empty list of dicts")
        body = {
            "model": self._config.model,
            "messages": messages,
        }
        # Only request JSON mode if the model might support it (not free tier)
        if "free" not in self._config.model.lower():
            body["response_format"] = {"type": "json_object"}
        try:
            result = self._transport(
                self._config.endpoint_url,
                self._config.headers(),
                body,
                float(self._config.timeout_s),
            )
        except LLMError:
            raise
        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"LLM request timed out after {self._config.timeout_s}s: {exc}"
            ) from exc
        except (ConnectionError, OSError) as exc:
            raise LLMTransportError(f"LLM request failed: {exc}") from exc
        except Exception as exc:
            raise LLMTransportError(f"LLM request failed: {exc}") from exc
        if not isinstance(result, dict):
            raise LLMResponseError(
                "LLM transport must return a parsed JSON object, "
                f"got {type(result).__name__}"
            )
        return result


# ---------------------------------------------------------------------------
# Strict JSON generation contract
# ---------------------------------------------------------------------------


def build_variant_messages(
    seed: dict[str, Any],
    categories: Sequence[str],
    *,
    variants_per_seed: int = 3,
) -> list[dict[str, str]]:
    """Build the chat messages requesting variants for one *seed*.

    The prompt states a strict JSON-only contract: the model must reply
    with ``{"variants": [{...}]}`` where every entry carries the exact
    ``seed_id`` of the requesting seed, a non-empty ``payload`` string, a
    supported ``mutation_category``, and an optional ``context_target``.
    """
    allowed = ", ".join(categories)
    system = (
        "You are an XSS security research assistant. Your task is to rewrite "
        "known XSS payloads into behaviour-preserving variants that would still "
        "execute JavaScript in a browser when reflected into an HTML page via "
        "innerHTML. Focus on bypassing ML-based XSS detectors.\n\n"
        "IMPORTANT RULES:\n"
        "1. Every variant MUST actually execute JavaScript when injected via innerHTML.\n"
        "2. Use raw HTML tags like <script>, <img onerror=...>, <svg onload=...>, "
        "<body onload=...>, <input onfocus=... autofocus>, <details open ontoggle=...>.\n"
        "3. Do NOT use URL encoding (%3C), HTML entities (&lt;), or JavaScript "
        "entities — these do NOT execute via innerHTML.\n"
        "4. Use different obfuscation techniques: mixed case, whitespace tricks, "
        "event handler variations, alternative JS functions (prompt, confirm, "
        "document.cookie, eval, String.fromCharCode).\n"
        "5. Each variant should look structurally different from the seed to "
        "challenge a detector's pattern recognition.\n\n"
        "Reply with JSON ONLY, no prose, matching this exact shape: "
        '{"variants": [{"seed_id": string, "payload": string, '
        '"mutation_category": string, "context_target": string}]}. '
        f'"mutation_category" must be one of: {allowed}. '
        '"seed_id" must equal the requesting seed_id exactly. '
        '"payload" must be a non-empty string that differs from the seed payload.'
    )
    user = json.dumps(
        {
            "seed_id": seed["sample_id"],
            "payload": seed["payload"],
            "context_target": seed.get("context_target", DEFAULT_CONTEXT_TARGET),
            "allowed_mutation_categories": list(categories),
            "max_variants": variants_per_seed,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_content(response: Any) -> str:
    """Extract the assistant content string from an API *response*."""
    if not isinstance(response, dict):
        raise LLMResponseError(
            "LLM response must be a JSON object, "
            f"got {type(response).__name__}"
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("LLM response has no 'choices' list to parse")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseError("LLM response choice must be a JSON object")
    message = first.get("message", first)
    if not isinstance(message, dict):
        raise LLMResponseError("LLM response message must be a JSON object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("LLM response message has no textual 'content'")
    return content


def parse_model_content(content: Any, *, seed_id: str) -> list[dict[str, Any]]:
    """Strictly parse one model *content* string into variant item dicts.

    Every violation (malformed JSON, non-object top level, missing or
    mistyped ``variants`` list, missing/blank/mistyped payload, missing or
    unsupported mutation category, non-dict entry, or a ``seed_id`` that
    does not equal *seed_id*) raises :class:`LLMResponseError` with a
    message naming the problem. Each returned item holds ``payload``,
    ``mutation_category``, ``seed_id``, and ``context_target``.
    """
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("LLM response content must be a non-empty string")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError(
            "LLM response top-level JSON must be an object with a "
            f"'variants' list, got {type(parsed).__name__}"
        )
    if "variants" not in parsed:
        raise LLMResponseError("LLM response object is missing the 'variants' key")
    variants = parsed["variants"]
    if not isinstance(variants, list):
        raise LLMResponseError(
            "LLM response 'variants' must be a list, "
            f"got {type(variants).__name__}"
        )
    items: list[dict[str, Any]] = []
    for index, entry in enumerate(variants):
        where = f"variants[{index}]"
        if not isinstance(entry, dict):
            raise LLMResponseError(
                f"LLM response {where} must be an object, "
                f"got {type(entry).__name__}"
            )
        entry_seed = entry.get("seed_id")
        if not isinstance(entry_seed, str) or not entry_seed:
            raise LLMResponseError(f"LLM response {where} is missing 'seed_id'")
        if entry_seed != seed_id:
            raise LLMResponseError(
                f"LLM response {where} has seed_id {entry_seed!r} "
                f"but the requesting seed is {seed_id!r}"
            )
        payload = entry.get("payload")
        if "payload" not in entry:
            raise LLMResponseError(f"LLM response {where} is missing 'payload'")
        if not isinstance(payload, str):
            raise LLMResponseError(
                f"LLM response {where} 'payload' must be a string, "
                f"got {type(payload).__name__}"
            )
        if not payload.strip():
            raise LLMResponseError(f"LLM response {where} 'payload' must be non-empty")
        category = entry.get("mutation_category")
        if "mutation_category" not in entry:
            raise LLMResponseError(
                f"LLM response {where} is missing 'mutation_category'"
            )
        if not isinstance(category, str) or category not in SUPPORTED_CATEGORIES:
            raise LLMResponseError(
                f"LLM response {where} has unsupported mutation category "
                f"{category!r}: expected one of {list(SUPPORTED_CATEGORIES)}"
            )
        context = entry.get("context_target")
        if context is None:
            context = DEFAULT_CONTEXT_TARGET
        if not isinstance(context, str) or not context.strip():
            raise LLMResponseError(
                f"LLM response {where} 'context_target' must be a non-empty string"
            )
        items.append(
            {
                "seed_id": entry_seed,
                "payload": payload,
                "mutation_category": category,
                "context_target": context,
            }
        )
    return items


# ---------------------------------------------------------------------------
# Seed validation and result assembly
# ---------------------------------------------------------------------------


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
            raise TypeError(f"category names must be str, got {type(entry).__name__}")
        if entry not in SUPPORTED_CATEGORIES:
            raise ValueError(
                f"unsupported mutation category {entry!r}: "
                f"expected one of {list(SUPPORTED_CATEGORIES)}"
            )
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
            f"seeds must be an iterable of dicts, got {type(seeds).__name__}"
        )
    try:
        items = list(seeds)
    except TypeError as exc:
        raise TypeError(
            f"seeds must be an iterable of dicts, got {type(seeds).__name__}"
        ) from exc
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"seed index {index}: must be a dict, got {type(item).__name__}"
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
    digest = hashlib.sha256(
        f"{seed_id}|{category}|{payload}".encode("utf-8")
    ).hexdigest()[:_VARIANT_ID_HEX_CHARS]
    return f"{_VARIANT_ID_PREFIX}{digest}"


@dataclass
class LLMGenerationResult:
    """Outcome of one :func:`generate_llm_variants` call.

    Attributes
    ----------
    variants:
        Accepted variant records in stable generation order (seed input
        order, then model response order). Each record carries
        ``variant_id``, ``seed_id``, ``payload``, ``mutation_category``,
        ``context_target``, ``source``/``seed_source``, ``generator``,
        ``generator_version``, ``model``, and ``split`` (``"adv_dev"``),
        plus ``seed_split``/``label``/``attack_category`` whenever the
        seed carried them. Payloads never repeat a seed payload and never
        repeat each other.
    errors:
        One auditable entry per seed whose model response was rejected
        (malformed JSON, contract violation, or transport failure), each
        holding ``seed_id``, ``reason``, and the preserved ``raw_content``
        and/or ``raw_response``. Invalid model output is never silently
        accepted and never appears in ``variants``.
    """

    variants: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


def generate_llm_variants(
    seeds: Iterable[dict[str, Any]],
    *,
    client: Any,
    allow_payload_submission: bool = False,
    categories: Sequence[str] | None = None,
    variants_per_seed: int = 3,
    max_per_seed: int | None = None,
    enforce_split_boundary: bool = True,
    allowed_splits: Sequence[str] | None = None,
) -> LLMGenerationResult:
    """Generate LLM variant records for *seeds* via an injected *client*.

    Parameters
    ----------
    seeds:
        Iterable of dicts with at least non-empty ``sample_id`` and
        ``payload`` fields. Extra provenance (``label``,
        ``attack_category``, ``source``, ``context_target``, ``split``)
        is preserved in each variant. Inputs are never mutated.
    client:
        Object exposing ``chat(messages)`` returning a parsed
        OpenAI-compatible response dict, plus a ``model`` attribute used
        for generator metadata. Tests inject a fake; production passes
        :class:`OpenAICompatibleClient`. No network is touched except
        through this object.
    allow_payload_submission:
        Explicit opt-in to submit seed payload text to the cloud API.
        Defaults to ``False``, in which case
        :class:`LLMPayloadSubmissionNotAllowedError` is raised before any
        request is attempted.
    categories:
        Subset of ``SUPPORTED_CATEGORIES`` the model may use, in prompt
        order. ``None`` allows every supported category. Unknown names
        raise ``ValueError``.
    variants_per_seed:
        Number of variants requested from the model per seed (``>= 1``).
    max_per_seed:
        Optional cap (``>= 1``) on accepted variants kept per seed.
    enforce_split_boundary:
        When ``True`` (the default), seeds whose ``split`` is present and
        outside ``allowed_splits`` raise ``ValueError`` before any
        request, so held-out records never leak into generated data.
    allowed_splits:
        Optional override of the accepted seed splits
        (default ``("train", "adv_dev")``).

    Returns
    -------
    LLMGenerationResult
        Accepted ``variants`` plus auditable per-seed ``errors``.
        Transport failures for one seed are recorded in ``errors`` and
        do not abort the remaining seeds.
    """
    if client is None or not hasattr(client, "chat") or not callable(client.chat):
        raise TypeError("client must expose a callable 'chat(messages)' method")
    if not isinstance(allow_payload_submission, bool):
        raise TypeError(
            "allow_payload_submission must be bool, "
            f"got {type(allow_payload_submission).__name__}"
        )
    selected_categories = _check_categories(categories)
    if isinstance(variants_per_seed, bool) or not isinstance(variants_per_seed, int):
        raise TypeError(
            f"variants_per_seed must be int, got {type(variants_per_seed).__name__}"
        )
    if variants_per_seed < 1:
        raise ValueError(
            f"variants_per_seed must be >= 1, got {variants_per_seed!r}"
        )
    if max_per_seed is not None:
        if isinstance(max_per_seed, bool) or not isinstance(max_per_seed, int):
            raise TypeError(
                f"max_per_seed must be int or None, got {type(max_per_seed).__name__}"
            )
        if max_per_seed < 1:
            raise ValueError(f"max_per_seed must be >= 1, got {max_per_seed!r}")
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

    # Safety gate: refuse before any payload text could reach the network.
    if not allow_payload_submission:
        raise LLMPayloadSubmissionNotAllowedError(
            "refusing to submit payload text to a cloud API without explicit "
            "opt-in: pass allow_payload_submission=True to proceed"
        )

    model_name = getattr(client, "model", None) or "unknown"
    seed_payloads = {item["payload"] for item in items}
    result = LLMGenerationResult()
    seen_global: set[str] = set()

    import time as _time

    for _seed_idx, item in enumerate(items):
        sample_id = item["sample_id"]
        accepted: list[dict[str, Any]] = []
        response: Any = None

        # Retry loop for rate-limited requests
        _max_retries = 5
        _base_delay = 10.0
        _last_error: Any = None
        for _attempt in range(_max_retries):
            try:
                messages = build_variant_messages(
                    item, selected_categories, variants_per_seed=variants_per_seed
                )
                response = client.chat(messages)
                content = _extract_content(response)
                parsed_items = parse_model_content(content, seed_id=sample_id)
                _last_error = None
                break  # Success
            except LLMTransportError as exc:
                if "429" in str(exc) or "Too Many Requests" in str(exc):
                    _delay = _base_delay * (2 ** _attempt)
                    _time.sleep(_delay)
                    _last_error = exc
                    continue
                # Non-retryable transport error
                _last_error = exc
                break
            except LLMError as exc:
                _last_error = exc
                break

        if _last_error is not None:
            exc = _last_error
            raw_content: Any = _safe_content_fallback(response) if response is not None else None
            if raw_content is None and isinstance(response, str):
                raw_content = response
            result.errors.append(
                {
                    "seed_id": sample_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "raw_content": raw_content,
                    "raw_response": _safe_response_copy(response),
                }
            )
            continue

        # Delay between seeds to avoid rate limits
        if _seed_idx < len(items) - 1:
            _time.sleep(2)

        for parsed in parsed_items:
            candidate = parsed["payload"]
            category = parsed["mutation_category"]
            if candidate in seed_payloads or candidate in seen_global:
                continue
            if max_per_seed is not None and len(accepted) >= max_per_seed:
                break
            seen_global.add(candidate)
            # Prefer an explicit model-provided context; fall back to the
            # seed's own context when the model only echoed the default.
            seed_context = item.get("context_target")
            context = parsed["context_target"]
            if context == DEFAULT_CONTEXT_TARGET and isinstance(
                seed_context, str
            ) and seed_context.strip():
                context = seed_context
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
                "model": model_name,
                "split": VARIANT_SPLIT,
            }
            if "split" in item:
                record["seed_split"] = item["split"]
            if "label" in item:
                record["label"] = item["label"]
            if "attack_category" in item:
                record["attack_category"] = item["attack_category"]
            accepted.append(record)
        result.variants.extend(accepted)
    return result


def _safe_content_fallback(response: Any) -> Any:
    """Best-effort raw content preservation for an invalid envelope."""
    if isinstance(response, dict):
        try:
            return _extract_content(response)
        except LLMError:
            return None
    if isinstance(response, str):
        return response
    return None


def _safe_response_copy(response: Any) -> Any:
    """Return a JSON-round-tripped copy of *response* when possible."""
    if response is None:
        return None
    try:
        return json.loads(json.dumps(response, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(response)
