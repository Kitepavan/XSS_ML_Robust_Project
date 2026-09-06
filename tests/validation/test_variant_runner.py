"""Focused TDD tests for the variant-runner batch API.

All browser interaction is faked via injected ``_runner`` callables.
Only synthetic probe strings are used: no downloaded payloads are
executed and no network traffic is sent.
"""

from __future__ import annotations

import pytest


def _fired_runner_factory(fired: bool = True, duration_ms: float = 5.0):
    calls: list[tuple] = []

    def _runner(payload, target, probe, timeout_ms):
        calls.append((payload, target, probe, timeout_ms))
        return fired, duration_ms

    _runner.calls = calls  # type: ignore[attr-defined]
    return _runner


LOCAL_URL = "http://localhost:8000/sandbox.html"


def _record(**overrides):
    base = {
        "sample_id": "S001",
        "seed_id": "SEED-1",
        "mutation_category": "encoding",
        "source": "adv_dev",
        "context_target": "reflected_html",
        "payload": "synthetic-probe-<img>",
    }
    base.update(overrides)
    return base


class TestBatchShape:
    def test_one_result_per_input_in_order(self):
        from xssharden.validation.variant_runner import validate_variants

        def _runner(payload, target, probe, timeout_ms):
            return payload.startswith("fires"), 1.0

        records = [
            _record(sample_id="S001", payload="fires-1"),
            _record(sample_id="S002", payload="quiet-1"),
            _record(sample_id="S003", payload="fires-2"),
        ]
        results = validate_variants(records, sandbox_url=LOCAL_URL, _runner=_runner)
        assert len(results) == 3
        assert [r["sample_id"] for r in results] == ["S001", "S002", "S003"]
        assert [r["valid"] for r in results] == [True, False, True]
        assert [r["payload"] for r in results] == ["fires-1", "quiet-1", "fires-2"]

    def test_accepts_any_iterable(self):
        from xssharden.validation.variant_runner import validate_variants

        runner = _fired_runner_factory(True)
        gen = (_record(sample_id=f"S{i:03d}", payload=f"p-{i}") for i in range(3))
        results = validate_variants(gen, sandbox_url=LOCAL_URL, _runner=runner)
        assert len(results) == 3
        assert [r["sample_id"] for r in results] == ["S000", "S001", "S002"]

    def test_empty_iterable_returns_empty(self):
        from xssharden.validation.variant_runner import validate_variants

        runner = _fired_runner_factory(True)
        assert validate_variants([], sandbox_url=LOCAL_URL, _runner=runner) == []
        assert runner.calls == []

    def test_preserves_provenance_fields(self):
        from xssharden.validation.variant_runner import validate_variants

        runner = _fired_runner_factory(True, duration_ms=7.0)
        record = _record(
            sample_id="S009",
            seed_id="SEED-9",
            mutation_category="whitespace",
            source="adv_dev",
            context_target="reflected_html",
            payload="fires-x",
            generator="programmatic",
        )
        (result,) = validate_variants(
            [record], sandbox_url=LOCAL_URL, _runner=runner
        )
        assert result["sample_id"] == "S009"
        assert result["seed_id"] == "SEED-9"
        assert result["mutation_category"] == "whitespace"
        assert result["source"] == "adv_dev"
        assert result["generator"] == "programmatic"
        assert result["context_target"] == "reflected_html"
        assert result["payload"] == "fires-x"
        assert result["valid"] is True
        assert result["status"] == "ok"
        assert result["timed_out"] is False
        assert result["error"] is None


class TestRequiredFieldValidation:
    def test_non_dict_record_rejected(self):
        from xssharden.validation.variant_runner import validate_variants

        with pytest.raises(TypeError):
            validate_variants(
                ["not-a-dict"],  # type: ignore[list-item]
                sandbox_url=LOCAL_URL,
                _runner=_fired_runner_factory(True),
            )

    def test_missing_payload_rejected(self):
        from xssharden.validation.variant_runner import validate_variants

        record = _record()
        del record["payload"]
        with pytest.raises((ValueError, TypeError)) as excinfo:
            validate_variants(
                [record], sandbox_url=LOCAL_URL, _runner=_fired_runner_factory(True)
            )
        assert "payload" in str(excinfo.value).lower()

    def test_empty_payload_rejected(self):
        from xssharden.validation.variant_runner import validate_variants

        with pytest.raises(ValueError):
            validate_variants(
                [_record(payload="   ")],
                sandbox_url=LOCAL_URL,
                _runner=_fired_runner_factory(True),
            )

    def test_non_string_payload_rejected(self):
        from xssharden.validation.variant_runner import validate_variants

        with pytest.raises(TypeError):
            validate_variants(
                [_record(payload=12345)],  # type: ignore[dict-item]
                sandbox_url=LOCAL_URL,
                _runner=_fired_runner_factory(True),
            )

    def test_error_names_record_index(self):
        from xssharden.validation.variant_runner import validate_variants

        records = [_record(sample_id="S001"), _record(sample_id="S002", payload=999)]
        with pytest.raises(TypeError) as excinfo:
            validate_variants(
                records,  # type: ignore[list-item]
                sandbox_url=LOCAL_URL,
                _runner=_fired_runner_factory(True),
            )
        assert "1" in str(excinfo.value)

    def test_records_container_must_be_iterable_of_dicts(self):
        from xssharden.validation.variant_runner import validate_variants

        with pytest.raises(TypeError):
            validate_variants(
                {"sample_id": "S001"},  # type: ignore[arg-type]
                sandbox_url=LOCAL_URL,
                _runner=_fired_runner_factory(True),
            )


class TestLocalOnlyAndPassthrough:
    def test_rejects_non_local_url_without_calling_runner(self):
        from xssharden.validation.variant_runner import validate_variants

        runner = _fired_runner_factory(True)
        with pytest.raises(ValueError):
            validate_variants(
                [_record()],
                sandbox_url="https://example.com/xss",
                _runner=runner,
            )
        assert runner.calls == []

    def test_rejects_missing_and_dual_targets(self):
        from xssharden.validation.variant_runner import validate_variants

        with pytest.raises(ValueError):
            validate_variants([_record()], _runner=_fired_runner_factory(True))
        with pytest.raises(ValueError):
            validate_variants(
                [_record()],
                sandbox_url=LOCAL_URL,
                fixture_path="some/file.html",
                _runner=_fired_runner_factory(True),
            )

    def test_passes_timeout_probe_context_and_target_through(self, tmp_path):
        from xssharden.validation.variant_runner import validate_variants

        fixture = tmp_path / "sandbox.html"
        fixture.write_text("<html><body>fixture</body></html>", encoding="utf-8")
        seen: dict = {}

        def _runner(payload, target, probe, timeout_ms):
            seen.update(
                payload=payload, target=target, probe=probe, timeout_ms=timeout_ms
            )
            return True, 2.0

        results = validate_variants(
            [_record(payload="fires-1")],
            fixture_path=str(fixture),
            timeout_ms=2345,
            probe="confirm",
            context_target="reflected_html",
            _runner=_runner,
        )
        assert seen["payload"] == "fires-1"
        assert seen["target"].startswith("file://")
        assert seen["probe"] == "confirm"
        assert seen["timeout_ms"] == pytest.approx(2345)
        assert results[0]["probe"] == "confirm"
        assert results[0]["timeout_ms"] == pytest.approx(2345)
        assert results[0]["context_target"] == "reflected_html"

    def test_per_record_probe_and_context_override(self):
        from xssharden.validation.variant_runner import validate_variants

        seen: list[tuple] = []

        def _runner(payload, target, probe, timeout_ms):
            seen.append((payload, probe))
            return True, 1.0

        records = [_record(payload="a", probe="confirm")]
        results = validate_variants(
            records, sandbox_url=LOCAL_URL, probe="alert", _runner=_runner
        )
        assert seen[0][1] == "confirm"
        assert results[0]["probe"] == "confirm"


class TestOutcomePreservationOrderingAndLimit:
    def test_preserves_invalid_timeout_error_outcomes(self):
        from xssharden.validation.variant_runner import validate_variants

        def _runner(payload, target, probe, timeout_ms):
            if payload == "ok-off":
                return False, 3.0
            if payload == "slow":
                raise TimeoutError("timed out waiting for probe")
            raise RuntimeError("fake harness failure")

        results = validate_variants(
            [
                _record(sample_id="S001", payload="ok-off"),
                _record(sample_id="S002", payload="slow"),
                _record(sample_id="S003", payload="boom"),
            ],
            sandbox_url=LOCAL_URL,
            _runner=_runner,
        )
        assert [r["sample_id"] for r in results] == ["S001", "S002", "S003"]
        assert results[0]["valid"] is False
        assert results[0]["status"] == "ok"
        assert results[1]["valid"] is False
        assert results[1]["status"] == "timeout"
        assert results[1]["timed_out"] is True
        assert results[2]["status"] == "error"
        assert "fake harness failure" in (results[2]["error"] or "")

    def test_limit_processes_first_n_only(self):
        from xssharden.validation.variant_runner import validate_variants

        runner = _fired_runner_factory(True)
        records = [_record(sample_id=f"S{i:03d}", payload=f"p-{i}") for i in range(5)]
        results = validate_variants(
            records, sandbox_url=LOCAL_URL, limit=2, _runner=runner
        )
        assert [r["sample_id"] for r in results] == ["S000", "S001"]
        assert len(runner.calls) == 2

    def test_limit_zero_returns_empty_without_calling_runner(self):
        from xssharden.validation.variant_runner import validate_variants

        runner = _fired_runner_factory(True)
        records = [_record(sample_id="S001")]
        assert (
            validate_variants(records, sandbox_url=LOCAL_URL, limit=0, _runner=runner)
            == []
        )
        assert runner.calls == []

    def test_bad_limit_rejected(self):
        from xssharden.validation.variant_runner import validate_variants

        with pytest.raises((TypeError, ValueError)):
            validate_variants([_record()], sandbox_url=LOCAL_URL, limit=-1)
        with pytest.raises(TypeError):
            validate_variants(
                [_record()], sandbox_url=LOCAL_URL, limit="2"  # type: ignore[arg-type]
            )

    def test_does_not_mutate_inputs(self):
        from xssharden.validation.variant_runner import validate_variants

        record = _record(sample_id="S001", payload="fires-1")
        snapshot = dict(record)
        validate_variants([record], sandbox_url=LOCAL_URL, _runner=_fired_runner_factory(True))
        assert record == snapshot


class TestValidateVariantsCli:
    """CLI batch command uses only local targets and injected fakes."""

    def _write_input(self, path, records):
        import json

        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return str(path)

    def test_cli_rejects_dual_targets_without_reading_payloads(self, tmp_path):
        from xssharden.cli import main

        inp = self._write_input(tmp_path / "in.jsonl", [_record()])
        out = str(tmp_path / "out.jsonl")
        rc = main(
            [
                "validate-variants",
                "--input", inp,
                "--output", out,
                "--sandbox-url", LOCAL_URL,
                "--fixture", str(tmp_path / "sandbox.html"),
            ]
        )
        assert rc == 2

    def test_cli_writes_one_result_per_input_in_order(self, tmp_path, monkeypatch):
        import json

        import xssharden.validation.variant_runner as runner_mod
        from xssharden.cli import main

        seen: dict = {}

        def _fake_validate(records, **kwargs):
            seen["kwargs"] = kwargs
            return [
                {**r, "valid": r["payload"].startswith("fires"), "status": "ok"}
                for r in records
            ]

        monkeypatch.setattr(runner_mod, "validate_variants", _fake_validate)
        inp = self._write_input(
            tmp_path / "in.jsonl",
            [_record(sample_id="S001", payload="fires-1"), _record(sample_id="S002", payload="quiet-1")],
        )
        out = str(tmp_path / "out.jsonl")
        rc = main(
            [
                "validate-variants",
                "--input", inp,
                "--output", out,
                "--sandbox-url", LOCAL_URL,
                "--limit", "2",
            ]
        )
        assert rc == 0
        assert seen["kwargs"]["sandbox_url"] == LOCAL_URL
        assert seen["kwargs"]["limit"] == 2
        with open(out, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        assert [r["sample_id"] for r in rows] == ["S001", "S002"]
        assert [r["valid"] for r in rows] == [True, False]

    def test_cli_rejects_invalid_json(self, tmp_path):
        from xssharden.cli import main

        inp = tmp_path / "bad.jsonl"
        inp.write_text("{not-json\n", encoding="utf-8")
        rc = main(
            [
                "validate-variants",
                "--input", str(inp),
                "--output", str(tmp_path / "out.jsonl"),
                "--sandbox-url", LOCAL_URL,
            ]
        )
        assert rc == 2
