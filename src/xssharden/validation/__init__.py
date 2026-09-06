"""Controlled local XSS browser-validation skeleton.

Payloads are validated only against a local sandbox URL
(``http(s)://localhost|127.0.0.1|::1``) or a local HTML fixture file.
Non-local/network targets are rejected before any payload is sent, so
payloads never leave the host.

Playwright/Chromium is an *optional* dependency: it is imported lazily
inside :func:`validate_payload` only when no injected runner is given.
If it is unavailable, a clear actionable ``RuntimeError`` is raised.
Tests inject fake runners and never need a browser.
"""

from __future__ import annotations

from pathlib import Path

from xssharden.validation.validator import (
    DEFAULT_TIMEOUT_MS,
    PACKAGED_FIXTURE_PATH,
    SUPPORTED_CONTEXTS,
    ValidationResult,
    is_local_url,
    validate_many,
    validate_payload,
)
from xssharden.validation.variant_runner import validate_variants

__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "PACKAGED_FIXTURE_PATH",
    "SUPPORTED_CONTEXTS",
    "ValidationResult",
    "is_local_url",
    "validate_many",
    "validate_payload",
    "validate_variants",
]
