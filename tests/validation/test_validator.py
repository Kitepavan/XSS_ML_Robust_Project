"""Focused TDD tests for the controlled local XSS browser-validation skeleton.

Tests use injected fake runners only — no browser is required and no real
downloaded payloads are executed. Synthetic probe strings only.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def validator_module():
    from xssharden.validation import validator as mod

    return mod


# ===================================================================
# Section 1 — typed result model
# ===================================================================

class TestValidationResultModel:
    def test_result_model_has_required_fields(self, validator_module):
        r = validator_module.ValidationResult(
            payload="<b>synthetic-probe</b>",
            valid=True,
            context_target="reflected_html",
            probe="alert",
            duration_ms=12.5,
            timeout_ms=3000,
            timed_out=False,
            error=None,
            status="ok",
        )
        assert r.payload == "<b>synthetic-probe</b>"
        assert r.valid is True
        assert r.context_target == "reflected_html"
        assert r.probe == "alert"
        assert r.duration_ms == pytest.approx(12.5)
        assert r.timeout_ms == pytest.approx(3000)
        assert r.timed_out is False
        assert r.error is None
        assert r.status == "ok"

    def test_result_model_defaults(self, validator_module):
        r = validator_module.ValidationResult(payload="hello", valid=False)
        assert r.valid is False
        assert r.context_target == "reflected_html"
        assert r.probe == "alert"
        assert r.timeout_ms == pytest.approx(validator_module.DEFAULT_TIMEOUT_MS)
        assert r.timed_out is False
        assert r.error is None
        assert isinstance(r.status, str) and len(r.status) > 0

    def test_result_model_to_dict_round_trip(self, validator_module):
        r = validator_module.ValidationResult(
            payload="hello",
            valid=False,
            context_target="reflected_html",
            probe="confirm",
            duration_ms=3.0,
            timeout_ms=2000,
            timed_out=False,
            error=None,
            status="ok",
        )
        d = r.to_dict()
        for key in (
            "payload",
            "valid",
            "context_target",
            "probe",
            "duration_ms",
            "timeout_ms",
            "timed_out",
            "error",
            "status",
        ):
            assert key in d
        assert d["payload"] == "hello"
        assert d["valid"] is False
        assert d["probe"] == "confirm"


# ===================================================================
# Section 2 — local-only target acceptance (fake runner, no browser)
# ===================================================================

def _fired_runner_factory(fired: bool, duration_ms: float = 5.0):
    calls: list[tuple] = []

    def _runner(payload: str, target: str, probe: str, timeout_ms: float):
        calls.append((payload, target, probe, timeout_ms))
        return fired, duration_ms

    _runner.calls = calls  # type: ignore[attr-defined]
    return _runner


class TestLocalTargetAcceptance:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8000/sandbox.html",
            "http://localhost/sandbox.html",
            "http://127.0.0.1:8000/sandbox.html",
            "http://127.0.0.1/sandbox.html",
            "http://[::1]:8000/sandbox.html",
        ],
    )
    def test_accepts_local_sandbox_urls(self, validator_module, url):
        runner = _fired_runner_factory(True)
        result = validator_module.validate_payload(
            "synthetic-payload-1",
            sandbox_url=url,
            probe="alert",
            timeout_ms=1500,
            _runner=runner,
        )
        assert result.valid is True
        assert result.probe == "alert"
        assert result.timeout_ms == pytest.approx(1500)
        assert len(runner.calls) == 1
        assert runner.calls[0][1] == url

    def test_accepts_local_fixture_path(self, validator_module, tmp_path):
        fixture = tmp_path / "sandbox.html"
        fixture.write_text("<html><body>fixture</body></html>", encoding="utf-8")
        runner = _fired_runner_factory(False)
        result = validator_module.validate_payload(
            "synthetic-payload-2",
            fixture_path=str(fixture),
            _runner=runner,
        )
        assert result.valid is False
        assert result.status == "ok"
        assert len(runner.calls) == 1
        # Fixture mode must stay local: file:// target derived from the path.
        assert runner.calls[0][1].startswith("file://")

    def test_packaged_fixture_exists(self):
        from pathlib import Path

        import xssharden.validation.validator as mod

        assert mod.PACKAGED_FIXTURE_PATH.is_file()
        text = Path(mod.PACKAGED_FIXTURE_PATH).read_text(encoding="utf-8")
        assert "#sink" in text or 'id="sink"' in text


# ===================================================================
# Section 3 — non-local URL rejection (runner must never be called)
# ===================================================================

class TestNonLocalRejection:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/xss",
            "http://example.com/",
            "http://evil.example.com/sandbox.html",
            "http://192.168.1.10/sandbox.html",
            "http://10.0.0.5:8000/",
            "http://172.16.0.9/",
            "http://8.8.8.8/",
            "https://localhost.evil.com/",
            "ftp://localhost/sandbox.html",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
        ],
    )
    def test_rejects_non_local_urls(self, validator_module, url):
        runner = _fired_runner_factory(True)
        with pytest.raises(ValueError):
            validator_module.validate_payload(
                "synthetic-payload",
                sandbox_url=url,
                _runner=runner,
            )
        assert runner.calls == [], "payload must never be sent off-host"

    def test_is_local_url_helper(self, validator_module):
        assert validator_module.is_local_url("http://localhost:8000/a") is True
        assert validator_module.is_local_url("http://127.0.0.1/a") is True
        assert validator_module.is_local_url("https://example.com/") is False
        assert validator_module.is_local_url("not a url") is False
        assert validator_module.is_local_url("") is False


# ===================================================================
# Section 4 — probe fired vs not fired, timeout, runner errors
# ===================================================================

class TestProbeOutcome:
    def test_probe_fired_is_valid(self, validator_module):
        runner = _fired_runner_factory(True, duration_ms=7.0)
        result = validator_module.validate_payload(
            "synthetic-fires",
            sandbox_url="http://localhost:8000/sandbox.html",
            probe="alert",
            _runner=runner,
        )
        assert result.valid is True
        assert result.timed_out is False
        assert result.error is None
        assert result.status == "ok"
        assert result.duration_ms == pytest.approx(7.0)

    def test_probe_not_fired_is_invalid(self, validator_module):
        runner = _fired_runner_factory(False, duration_ms=9.0)
        result = validator_module.validate_payload(
            "just benign text",
            sandbox_url="http://localhost:8000/sandbox.html",
            probe="prompt",
            _runner=runner,
        )
        assert result.valid is False
        assert result.probe == "prompt"
        assert result.timed_out is False
        assert result.status == "ok"

    def test_timeout_maps_to_timeout_status(self, validator_module):
        def _timeout_runner(payload, target, probe, timeout_ms):
            raise TimeoutError("timed out waiting for probe")

        result = validator_module.validate_payload(
            "synthetic-slow",
            sandbox_url="http://localhost:8000/sandbox.html",
            timeout_ms=250,
            _runner=_timeout_runner,
        )
        assert result.valid is False
        assert result.timed_out is True
        assert result.status == "timeout"
        assert result.error is not None
        lowered = result.error.lower()
        assert "timeout" in lowered or "timed out" in lowered

    def test_runner_exception_maps_to_error_status(self, validator_module):
        def _boom_runner(payload, target, probe, timeout_ms):
            raise RuntimeError("fake harness failure")

        result = validator_module.validate_payload(
            "synthetic-err",
            sandbox_url="http://localhost:8000/sandbox.html",
            _runner=_boom_runner,
        )
        assert result.valid is False
        assert result.status == "error"
        assert result.error is not None and "fake harness failure" in result.error

    def test_timeout_value_forwarded_to_runner(self, validator_module):
        seen: dict = {}

        def _runner(payload, target, probe, timeout_ms):
            seen["timeout_ms"] = timeout_ms
            return False, 1.0

        validator_module.validate_payload(
            "x",
            sandbox_url="http://localhost:8000/sandbox.html",
            timeout_ms=2345,
            _runner=_runner,
        )
        assert seen["timeout_ms"] == pytest.approx(2345)


# ===================================================================
# Section 5 — malformed inputs
# ===================================================================

class TestMalformedInputs:
    def test_none_payload_rejected(self, validator_module):
        with pytest.raises(TypeError):
            validator_module.validate_payload(
                None,  # type: ignore[arg-type]
                sandbox_url="http://localhost:8000/sandbox.html",
                _runner=_fired_runner_factory(True),
            )

    def test_non_string_payload_rejected(self, validator_module):
        with pytest.raises(TypeError):
            validator_module.validate_payload(
                12345,  # type: ignore[arg-type]
                sandbox_url="http://localhost:8000/sandbox.html",
                _runner=_fired_runner_factory(True),
            )

    def test_missing_target_rejected(self, validator_module):
        with pytest.raises(ValueError):
            validator_module.validate_payload(
                "x", _runner=_fired_runner_factory(True)
            )

    def test_both_targets_rejected(self, validator_module, tmp_path):
        fixture = tmp_path / "s.html"
        fixture.write_text("<html></html>", encoding="utf-8")
        with pytest.raises(ValueError):
            validator_module.validate_payload(
                "x",
                sandbox_url="http://localhost:8000/sandbox.html",
                fixture_path=str(fixture),
                _runner=_fired_runner_factory(True),
            )

    def test_missing_fixture_file_rejected(self, validator_module, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            validator_module.validate_payload(
                "x",
                fixture_path=str(tmp_path / "does-not-exist.html"),
                _runner=_fired_runner_factory(True),
            )

    def test_directory_fixture_rejected(self, validator_module, tmp_path):
        with pytest.raises(ValueError):
            validator_module.validate_payload(
                "x",
                fixture_path=str(tmp_path),
                _runner=_fired_runner_factory(True),
            )

    @pytest.mark.parametrize("bad_timeout", [0, -1, -500, "fast", None])
    def test_bad_timeout_rejected(self, validator_module, bad_timeout):
        with pytest.raises((TypeError, ValueError)):
            validator_module.validate_payload(
                "x",
                sandbox_url="http://localhost:8000/sandbox.html",
                timeout_ms=bad_timeout,  # type: ignore[arg-type]
                _runner=_fired_runner_factory(True),
            )

    @pytest.mark.parametrize("bad_probe", ["", "   ", None, 123])
    def test_bad_probe_rejected(self, validator_module, bad_probe):
        with pytest.raises((TypeError, ValueError)):
            validator_module.validate_payload(
                "x",
                sandbox_url="http://localhost:8000/sandbox.html",
                probe=bad_probe,  # type: ignore[arg-type]
                _runner=_fired_runner_factory(True),
            )


# ===================================================================
# Section 6 — Playwright is optional and lazy
# ===================================================================

class TestPlaywrightOptional:
    def test_import_does_not_require_playwright(self, validator_module):
        # The skeleton must import without a browser installed.
        assert "playwright" not in sys.modules or sys.modules["playwright"] is None
        assert hasattr(validator_module, "validate_payload")

    def test_missing_playwright_raises_actionable_error(
        self, validator_module, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        with pytest.raises(RuntimeError) as excinfo:
            validator_module.validate_payload(
                "synthetic",
                sandbox_url="http://localhost:8000/sandbox.html",
            )
        message = str(excinfo.value).lower()
        assert "playwright" in message
        assert "pip install" in message or "install" in message


# ===================================================================
# Section 7 — batch helper
# ===================================================================

class TestValidateMany:
    def test_validate_many_preserves_order(self, validator_module):
        def _runner(payload, target, probe, timeout_ms):
            return payload.startswith("fires"), 1.0

        results = validator_module.validate_many(
            ["fires-1", "quiet-1", "fires-2"],
            sandbox_url="http://localhost:8000/sandbox.html",
            _runner=_runner,
        )
        assert [r.valid for r in results] == [True, False, True]
        assert [r.payload for r in results] == ["fires-1", "quiet-1", "fires-2"]

    def test_validate_many_rejects_empty(self, validator_module):
        with pytest.raises(ValueError):
            validator_module.validate_many(
                [],
                sandbox_url="http://localhost:8000/sandbox.html",
                _runner=_fired_runner_factory(True),
            )
