"""Controlled local XSS browser-validation skeleton.

Only local targets are accepted: ``http(s)://`` URLs whose host is
``localhost``, ``127.0.0.1`` or ``::1``, or a local HTML fixture file
(exposed to the runner as a ``file://`` URL). Anything else raises
``ValueError`` before a payload is touched, so payloads never leave
the host.

Playwright is optional and imported lazily. Pass ``_runner`` (a fake in
tests) to avoid needing a browser.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

DEFAULT_TIMEOUT_MS: float = 50.0
DEFAULT_CONTEXT_TARGET = "reflected_html"
DEFAULT_PROBE = "alert"

SUPPORTED_CONTEXTS = ("reflected_html",)

PACKAGED_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sandbox.html"

# Callable returning either (fired, duration_ms) or a bare fired bool.
RunnerFn = Callable[[str, str, str, float], Any]

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local_url(url: Any) -> bool:
    """Return True only for local http(s) sandbox URLs."""
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urlparse(url.strip())
    except (ValueError, AttributeError):
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    try:
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    return host in _LOCAL_HOSTS


@dataclass(frozen=True)
class ValidationResult:
    """Typed outcome of one controlled validation probe."""

    payload: str
    valid: bool
    context_target: str = DEFAULT_CONTEXT_TARGET
    probe: str = DEFAULT_PROBE
    duration_ms: float = 0.0
    timeout_ms: float = DEFAULT_TIMEOUT_MS
    timed_out: bool = False
    error: str | None = None
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this result."""
        return {
            "payload": self.payload,
            "valid": self.valid,
            "context_target": self.context_target,
            "probe": self.probe,
            "duration_ms": self.duration_ms,
            "timeout_ms": self.timeout_ms,
            "timed_out": self.timed_out,
            "error": self.error,
            "status": self.status,
        }


def _check_timeout(timeout_ms: Any) -> float:
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, (int, float)):
        raise TypeError(
            f"timeout_ms must be a positive number, got {type(timeout_ms).__name__}"
        )
    value = float(timeout_ms)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"timeout_ms must be a positive finite number, got {timeout_ms!r}")
    return value


def _check_probe(probe: Any) -> str:
    if not isinstance(probe, str):
        raise TypeError(f"probe must be str, got {type(probe).__name__}")
    if not probe.strip():
        raise ValueError("probe must be a non-empty probe name (e.g. 'alert')")
    return probe


def _check_context(context_target: Any) -> str:
    if not isinstance(context_target, str):
        raise TypeError(f"context_target must be str, got {type(context_target).__name__}")
    stripped = context_target.strip()
    if not stripped:
        raise ValueError("context_target must be a non-empty string")
    if stripped not in SUPPORTED_CONTEXTS:
        raise ValueError(
            f"unsupported context_target {context_target!r}; "
            f"supported: {', '.join(SUPPORTED_CONTEXTS)}"
        )
    return stripped


def _allowed_fixture_roots() -> list[Path]:
    """Return directories fixture files may live under.

    The packaged fixture lives under the project root; pytest ``tmp_path``
    fixtures live under the system temp dir. Both stay local to this host.
    Anything else (e.g. ``/etc/passwd``) is refused as outside the
    project/workspace.
    """
    roots: list[Path] = []
    try:
        roots.append(Path(__file__).resolve().parents[3])
    except IndexError:
        pass
    try:
        roots.append(Path.cwd().resolve())
    except OSError:
        pass
    candidates = [tempfile.gettempdir(), os.environ.get("TMPDIR", ""), "/tmp"]
    for candidate in candidates:
        if candidate:
            try:
                roots.append(Path(candidate).resolve())
            except OSError:
                continue
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _is_within_roots(resolved: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            if resolved.is_relative_to(root):
                return True
        except (OSError, ValueError):
            continue
    return False


def _resolve_target(
    sandbox_url: Any, fixture_path: Any
) -> str:
    """Validate inputs and return the local target URL. Never touches payloads."""
    if sandbox_url is not None and fixture_path is not None:
        raise ValueError("pass exactly one of sandbox_url or fixture_path, not both")
    if sandbox_url is None and fixture_path is None:
        raise ValueError("one of sandbox_url or fixture_path is required")
    if sandbox_url is not None:
        if not isinstance(sandbox_url, str):
            raise TypeError(
                f"sandbox_url must be str, got {type(sandbox_url).__name__}"
            )
        if not is_local_url(sandbox_url):
            raise ValueError(
                f"sandbox_url must be a local sandbox URL "
                f"(http(s)://localhost|127.0.0.1|::1); refused {sandbox_url!r}"
            )
        return sandbox_url
    # Fixture-path mode: local file only, exposed as file:// URL.
    path = fixture_path if isinstance(fixture_path, Path) else Path(fixture_path)
    if not path.exists():
        raise FileNotFoundError(f"fixture_path does not exist: {fixture_path!r}")
    if not path.is_file():
        raise ValueError(f"fixture_path must be a file, got: {fixture_path!r}")
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise ValueError(f"fixture_path could not be resolved: {fixture_path!r}") from exc
    if not _is_within_roots(resolved, _allowed_fixture_roots()):
        raise ValueError(
            f"fixture_path must be inside the project/workspace or system temp dir; "
            f"refused {fixture_path!r}"
        )
    return resolved.as_uri()


def _is_timeout_exception(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or type(exc).__name__ == "TimeoutError"


# Local-only browser executables used when the Playwright-bundled
# chromium-headless-shell is unavailable (e.g. version skew between the
# installed Playwright package and downloaded browsers). Only existing
# local files are ever used; no network download or remote target is
# involved.
_SYSTEM_CHROME_CANDIDATES = (
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)
_PLAYWRIGHT_EXECUTABLE_ENV_VARS = (
    "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH",
    "PLAYWRIGHT_EXECUTABLE_PATH",
)


def _find_local_chrome_executable() -> str | None:
    """Return the first existing local Chrome executable, if any.

    An explicit ``PLAYWRIGHT_*_EXECUTABLE_PATH`` env var pointing at an
    existing file wins; otherwise the well-known system paths are checked
    in order. Returns ``None`` when no local executable exists.
    """
    for env_var in _PLAYWRIGHT_EXECUTABLE_ENV_VARS:
        configured = os.environ.get(env_var, "").strip()
        if configured and Path(configured).is_file():
            return configured
    for candidate in _SYSTEM_CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def _resolve_chromium_executable(chromium: Any) -> str | None:
    """Return an ``executable_path`` override or ``None`` for default launch.

    Prefers the Playwright-bundled executable when its file exists (caller
    launches without an override). Otherwise returns an existing local
    fallback executable, or ``None`` when no fallback exists (caller then
    attempts the default launch and surfaces Playwright's own error).
    """
    try:
        bundled = chromium.executable_path
    except Exception:
        bundled = None
    if bundled and isinstance(bundled, (str, Path)):
        try:
            if Path(bundled).is_file():
                return None
        except (OSError, ValueError):
            pass
    return _find_local_chrome_executable()


def _is_missing_executable_error(exc: BaseException) -> bool:
    """Return True when Playwright reports its browser executable is missing."""
    try:
        message = str(exc)
    except Exception:
        return False
    return "Executable doesn't exist" in message


def _run_with_playwright(
    payload: str, target: str, probe: str, timeout_ms: float
) -> tuple[bool, float]:
    """Run one payload against a local target with headless Chromium.

    Only called when no injected runner is given. Playwright is imported
    lazily so the package works without it.
    """
    if target.startswith("file://"):
        pass
    elif not is_local_url(target):
        raise ValueError(f"refusing non-local target: {target!r}")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed, so browser validation cannot run. "
            "Install it with 'pip install playwright' and then run "
            "'playwright install chromium' (local sandbox only). "
            "Tests and API callers can inject _runner instead of a browser."
        ) from exc

    started = time.perf_counter()
    fired = False
    with sync_playwright() as playwright:
        # Portability: prefer the Playwright-bundled executable when its
        # file exists; otherwise launch with an existing local system
        # Chrome (/usr/bin/google-chrome, /usr/bin/chromium, ...). Only
        # local headless launches are used -- no proxy, channel, or remote
        # (ws/CDP/server) options -- so payloads stay on this host.
        fallback_executable = _resolve_chromium_executable(playwright.chromium)
        if fallback_executable is None:
            try:
                browser = playwright.chromium.launch(headless=True)
            except Exception as launch_exc:
                if not _is_missing_executable_error(launch_exc):
                    raise
                retry_executable = _find_local_chrome_executable()
                if retry_executable is None:
                    raise
                browser = playwright.chromium.launch(
                    headless=True, executable_path=retry_executable
                )
        else:
            browser = playwright.chromium.launch(
                headless=True, executable_path=fallback_executable
            )
        try:
            page = browser.new_page()
            seen: dict[str, bool] = {}

            def _on_dialog(dialog: Any) -> None:
                seen["fired"] = True
                try:
                    dialog.dismiss()
                except Exception:
                    pass

            page.on("dialog", _on_dialog)
            page.add_init_script(
                """
                window.__probeFired = false;
                for (const name of ['alert', 'confirm', 'prompt']) {
                  try {
                    const original = window[name];
                    window[name] = function (...args) {
                      window.__probeFired = true;
                      window.__probeName = name;
                      try { return original.apply(this, args); }
                      catch (e) { return name === 'prompt' ? null : undefined; }
                    };
                  } catch (e) { /* keep going */ }
                }
                window.__xssProbe = function (name) {
                  window.__probeFired = true;
                  window.__probeName = name || 'custom';
                };
                """
            )
            page.goto(target, wait_until="domcontentloaded", timeout=int(timeout_ms))
            # Deliver the payload to the reflected sink when present.
            sink_count = page.evaluate(
                "() => document.querySelectorAll('#sink').length"
            )
            if sink_count:
                page.evaluate(
                    "(value) => { document.querySelector('#sink').innerHTML = value; }",
                    payload,
                )
            else:
                page.evaluate("(value) => { document.body.innerHTML += value; }", payload)
            try:
                page.wait_for_function(
                    "() => window.__probeFired === true",
                    timeout=int(timeout_ms),
                )
            except Exception as wait_exc:
                if _is_timeout_exception(wait_exc):
                    raise TimeoutError(
                        f"timeout: timed out after {timeout_ms}ms "
                        f"waiting for probe {probe!r}"
                    ) from wait_exc
                raise
            fired = bool(seen.get("fired") or page.evaluate("() => window.__probeFired"))
        finally:
            browser.close()
    duration_ms = (time.perf_counter() - started) * 1000.0
    _ = probe  # probe selects which hook the fixture arms; firing is observed above.
    return fired, duration_ms


def validate_payload(
    payload: str,
    *,
    sandbox_url: str | None = None,
    fixture_path: str | Path | None = None,
    context_target: str = DEFAULT_CONTEXT_TARGET,
    probe: str = DEFAULT_PROBE,
    timeout_ms: float = DEFAULT_TIMEOUT_MS,
    _runner: RunnerFn | None = None,
) -> ValidationResult:
    """Validate one payload in the controlled local sandbox.

    Exactly one of ``sandbox_url`` (local http(s) URL) or ``fixture_path``
    (local HTML file) is required. ``_runner`` is an injectable
    ``(payload, target, probe, timeout_ms) -> (fired, duration_ms)`` hook
    used by tests; when omitted, Playwright/Chromium is used and must be
    installed.
    """
    if not isinstance(payload, str):
        raise TypeError(f"payload must be str, got {type(payload).__name__}")
    if not payload.strip():
        raise ValueError("payload must be a non-empty string")
    checked_probe = _check_probe(probe)
    checked_context = _check_context(context_target)
    checked_timeout = _check_timeout(timeout_ms)
    target = _resolve_target(sandbox_url, fixture_path)

    if _runner is None:
        # Browser path: a missing Playwright install is an environment
        # error and must propagate as RuntimeError, not become a result.
        try:
            fired, duration_ms = _run_with_playwright(
                payload, target, checked_probe, checked_timeout
            )
        except RuntimeError:
            raise
        except Exception as exc:
            if _is_timeout_exception(exc):
                return ValidationResult(
                    payload=payload,
                    valid=False,
                    context_target=checked_context,
                    probe=checked_probe,
                    duration_ms=0.0,
                    timeout_ms=checked_timeout,
                    timed_out=True,
                    error=str(exc),
                    status="timeout",
                )
            return ValidationResult(
                payload=payload,
                valid=False,
                context_target=checked_context,
                probe=checked_probe,
                duration_ms=0.0,
                timeout_ms=checked_timeout,
                timed_out=False,
                error=str(exc),
                status="error",
            )
    else:
        started = time.perf_counter()
        try:
            outcome = _runner(payload, target, checked_probe, checked_timeout)
        except Exception as exc:
            if _is_timeout_exception(exc):
                return ValidationResult(
                    payload=payload,
                    valid=False,
                    context_target=checked_context,
                    probe=checked_probe,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    timeout_ms=checked_timeout,
                    timed_out=True,
                    error=str(exc),
                    status="timeout",
                )
            return ValidationResult(
                payload=payload,
                valid=False,
                context_target=checked_context,
                probe=checked_probe,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                timeout_ms=checked_timeout,
                timed_out=False,
                error=str(exc),
                status="error",
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if isinstance(outcome, tuple) and len(outcome) == 2:
            fired_raw, duration_raw = outcome
            fired = bool(fired_raw)
            try:
                duration_ms = float(duration_raw)
            except (TypeError, ValueError):
                duration_ms = elapsed_ms
        else:
            fired = bool(outcome)
            duration_ms = elapsed_ms

    return ValidationResult(
        payload=payload,
        valid=bool(fired),
        context_target=checked_context,
        probe=checked_probe,
        duration_ms=float(duration_ms),
        timeout_ms=checked_timeout,
        timed_out=False,
        error=None,
        status="ok",
    )


def validate_many(
    payloads: Sequence[str],
    *,
    sandbox_url: str | None = None,
    fixture_path: str | Path | None = None,
    context_target: str = DEFAULT_CONTEXT_TARGET,
    probe: str = DEFAULT_PROBE,
    timeout_ms: float = DEFAULT_TIMEOUT_MS,
    _runner: RunnerFn | None = None,
) -> list[ValidationResult]:
    """Validate several payloads in order against the same local target."""
    if not isinstance(payloads, (list, tuple)):
        raise TypeError(
            f"payloads must be a list/tuple of str, got {type(payloads).__name__}"
        )
    if len(payloads) == 0:
        raise ValueError("payloads must be a non-empty sequence")
    return [
        validate_payload(
            item,
            sandbox_url=sandbox_url,
            fixture_path=fixture_path,
            context_target=context_target,
            probe=probe,
            timeout_ms=timeout_ms,
            _runner=_runner,
        )
        for item in payloads
    ]
