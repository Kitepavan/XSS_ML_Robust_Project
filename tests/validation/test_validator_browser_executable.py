"""Focused TDD tests for Playwright browser-launch portability.

The bundled chromium-headless-shell may be unavailable while a local
system Chrome exists (e.g. /usr/bin/google-chrome). These tests use
mocked Playwright objects only — no browser is launched and no
downloaded dataset payloads are executed (synthetic strings only).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def validator_module():
    from xssharden.validation import validator as mod

    return mod


# --- helpers to build mocked Playwright objects --------------------

class _FakeChromium:
    """Minimal mock of playwright.chromium (executable_path + launch)."""

    def __init__(self, executable_path=None, launch_side_effect=None):
        self._executable_path = executable_path
        self.launch_calls: list[dict] = []
        self._launch_side_effect = launch_side_effect

    @property
    def executable_path(self):
        if isinstance(self._executable_path, Exception):
            raise self._executable_path
        return self._executable_path

    def launch(self, **kwargs):
        self.launch_calls.append(kwargs)
        if self._launch_side_effect is not None:
            return self._launch_side_effect(kwargs, len(self.launch_calls))
        raise AssertionError("launch_side_effect not configured for _FakeChromium")


class _FakeBrowser:
    def close(self):
        pass


def _make_fake_page(fired=False):
    class _FakeDialog:
        def dismiss(self):
            pass

    class _FakePage:
        def __init__(self):
            self.handlers = {}

        def on(self, event, handler):
            self.handlers[event] = handler

        def add_init_script(self, script):
            pass

        def goto(self, target, wait_until=None, timeout=None):
            pass

        def evaluate(self, script, value=None):
            if "querySelectorAll('#sink')" in script:
                return 0
            if "__probeFired" in script:
                return fired
            return None

        def wait_for_function(self, script, timeout=None):
            if fired:
                return None
            raise TimeoutError("timed out waiting for probe")

    return _FakePage()


def _install_fake_playwright(monkeypatch, fake_chromium, fired=False):
    """Install a fake playwright.sync_api module; return the fake browser."""
    fake_browser = _FakeBrowser()
    fake_page = _make_fake_page(fired=fired)

    def _launch_side_effect(kwargs, call_no):
        # Attach page factory to browser for _run_with_playwright flow.
        fake_browser.new_page = lambda: fake_page  # type: ignore[attr-defined]
        return fake_browser

    fake_chromium._launch_side_effect = _launch_side_effect

    fake_playwright_obj = types.SimpleNamespace(chromium=fake_chromium)

    class _FakeSyncPlaywrightCtx:
        def __enter__(self):
            return fake_playwright_obj

        def __exit__(self, *exc):
            return False

    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = lambda: _FakeSyncPlaywrightCtx()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)
    # Ensure `from playwright.sync_api import sync_playwright` resolves to fake.
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    return fake_browser


# --- executable selection ------------------------------------------

class TestExecutableSelection:
    def test_prefers_bundled_playwright_executable_when_available(
        self, validator_module, monkeypatch, tmp_path
    ):
        bundled = tmp_path / "bundled-chrome"
        bundled.write_text("fake", encoding="utf-8")
        fake_chromium = _FakeChromium(executable_path=str(bundled))
        _install_fake_playwright(monkeypatch, fake_chromium, fired=True)
        # Bundled exists -> default launch without executable_path override.
        validator_module._run_with_playwright(
            "synthetic-probe",
            "http://localhost:8000/sandbox.html",
            "alert",
            1500,
        )
        assert len(fake_chromium.launch_calls) == 1
        assert "executable_path" not in fake_chromium.launch_calls[0]
        assert fake_chromium.launch_calls[0].get("headless") is True

    def test_falls_back_to_google_chrome_when_bundled_missing(
        self, validator_module, monkeypatch, tmp_path
    ):
        missing_bundled = str(tmp_path / "does-not-exist-chrome")
        fake_chromium = _FakeChromium(executable_path=missing_bundled)
        _install_fake_playwright(monkeypatch, fake_chromium, fired=True)
        real_is_file = Path.is_file

        def _fake_is_file(self):
            if str(self) == "/usr/bin/google-chrome":
                return True
            if str(self) == missing_bundled:
                return False
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", _fake_is_file)
        validator_module._run_with_playwright(
            "synthetic-probe",
            "http://localhost:8000/sandbox.html",
            "alert",
            1500,
        )
        assert len(fake_chromium.launch_calls) == 1
        assert (
            fake_chromium.launch_calls[0].get("executable_path")
            == "/usr/bin/google-chrome"
        )

    def test_falls_back_to_chromium_variant_when_google_chrome_missing(
        self, validator_module, monkeypatch, tmp_path
    ):
        missing_bundled = str(tmp_path / "does-not-exist-chrome")
        fake_chromium = _FakeChromium(executable_path=missing_bundled)
        _install_fake_playwright(monkeypatch, fake_chromium, fired=True)
        real_is_file = Path.is_file

        def _fake_is_file(self):
            s = str(self)
            if s == "/usr/bin/google-chrome":
                return False
            if s == "/usr/bin/chromium":
                return True
            if s == missing_bundled:
                return False
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", _fake_is_file)
        validator_module._run_with_playwright(
            "synthetic-probe",
            "http://localhost:8000/sandbox.html",
            "alert",
            1500,
        )
        assert (
            fake_chromium.launch_calls[0].get("executable_path")
            == "/usr/bin/chromium"
        )

    def test_prefers_google_chrome_over_chromium(
        self, validator_module, monkeypatch, tmp_path
    ):
        missing_bundled = str(tmp_path / "does-not-exist-chrome")
        fake_chromium = _FakeChromium(executable_path=missing_bundled)
        _install_fake_playwright(monkeypatch, fake_chromium, fired=True)
        real_is_file = Path.is_file

        def _fake_is_file(self):
            s = str(self)
            if s in ("/usr/bin/google-chrome", "/usr/bin/chromium"):
                return True
            if s == missing_bundled:
                return False
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", _fake_is_file)
        validator_module._run_with_playwright(
            "synthetic-probe",
            "http://localhost:8000/sandbox.html",
            "alert",
            1500,
        )
        assert (
            fake_chromium.launch_calls[0].get("executable_path")
            == "/usr/bin/google-chrome"
        )

    def test_retries_with_local_executable_when_default_launch_missing(
        self, validator_module, monkeypatch, tmp_path
    ):
        """Default launch raising 'Executable doesn't exist' retries locally."""
        bundled = tmp_path / "bundled-chrome"
        bundled.write_text("fake", encoding="utf-8")
        calls: list[dict] = []
        fake_browser = _FakeBrowser()
        fake_page = _make_fake_page(fired=True)

        class _RetryChromium:
            executable_path = str(bundled)

            def launch(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise Exception(
                        "BrowserType.launch: Executable doesn't exist at "
                        "/home/pavan/.cache/ms-playwright/chromium_headless_shell-1234/"
                        "chrome-headless-shell-linux64/chrome-headless-shell"
                    )
                fake_browser.new_page = lambda: fake_page  # type: ignore[attr-defined]
                return fake_browser

        fake_chromium = _RetryChromium()
        fake_playwright_obj = types.SimpleNamespace(chromium=fake_chromium)

        class _Ctx:
            def __enter__(self):
                return fake_playwright_obj

            def __exit__(self, *exc):
                return False

        fake_module = types.ModuleType("playwright.sync_api")
        fake_module.sync_playwright = lambda: _Ctx()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)
        monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))

        real_is_file = Path.is_file

        def _fake_is_file(self):
            if str(self) == "/usr/bin/google-chrome":
                return True
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", _fake_is_file)
        validator_module._run_with_playwright(
            "synthetic-probe",
            "http://localhost:8000/sandbox.html",
            "alert",
            1500,
        )
        assert len(calls) == 2
        assert "executable_path" not in calls[0]
        assert calls[1].get("executable_path") == "/usr/bin/google-chrome"

    def test_launch_uses_local_only_options(
        self, validator_module, monkeypatch, tmp_path
    ):
        bundled = tmp_path / "bundled-chrome"
        bundled.write_text("fake", encoding="utf-8")
        fake_chromium = _FakeChromium(executable_path=str(bundled))
        _install_fake_playwright(monkeypatch, fake_chromium, fired=True)
        validator_module._run_with_playwright(
            "synthetic-probe",
            "http://localhost:8000/sandbox.html",
            "alert",
            1500,
        )
        kwargs = fake_chromium.launch_calls[0]
        # Local-only: no network/remote targets, proxy, or CDP endpoints.
        for forbidden in (
            "proxy",
            "ws_endpoint",
            "wsEndpoint",
            "connect_over_cdp",
            "cdp_endpoint",
            "url",
            "server",
            "channel",
        ):
            assert forbidden not in kwargs
        for value in kwargs.values():
            if isinstance(value, str):
                lowered = value.lower()
                assert "http://" not in lowered and "ws://" not in lowered


class TestRunWithPlaywrightLocalOnly:
    def test_rejects_nonlocal_target_before_launch(
        self, validator_module, monkeypatch
    ):
        class _NeverChromium:
            executable_path = "/nonexistent"

            def launch(self, **kwargs):  # pragma: no cover
                raise AssertionError("launch must not be called for remote target")

        fake_playwright_obj = types.SimpleNamespace(chromium=_NeverChromium())

        class _Ctx:
            def __enter__(self):
                return fake_playwright_obj  # pragma: no cover

            def __exit__(self, *exc):
                return False

        fake_module = types.ModuleType("playwright.sync_api")
        fake_module.sync_playwright = lambda: _Ctx()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)
        monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
        with pytest.raises(ValueError):
            validator_module._run_with_playwright(
                "synthetic-probe",
                "https://example.com/xss",
                "alert",
                1500,
            )
