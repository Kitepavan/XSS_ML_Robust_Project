"""Focused tests for deterministic robustness report generation (Task 20).

Uses fake detectors only: no network, no Ollama, no browser, no payload
execution. Every test is deterministic.
"""

from __future__ import annotations

import copy
import json

import pytest


class PayloadFakeDetector:
    """Minimal read-only fake detector keyed by payload string."""

    def __init__(self, table, threshold=0.5):
        self._table = dict(table)
        self.threshold_ = float(threshold)

    def predict_proba(self, records):
        import numpy as np

        rows = []
        for record in records:
            score, _ = self._table[record["payload"]]
            rows.append([1.0 - float(score), float(score)])
        return np.asarray(rows, dtype=float)

    def predict(self, records):
        import numpy as np

        return np.asarray(
            [self._table[record["payload"]][1] for record in records], dtype=int
        )


def _clean_records():
    return [
        {"sample_id": "c1", "payload": "hello world", "label": 0,
         "source": "s1", "attack_category": "benign", "split": "clean-test"},
        {"sample_id": "c2", "payload": "good day friend", "label": 0,
         "source": "s1", "attack_category": "benign", "split": "clean-test"},
        {"sample_id": "c3", "payload": "<script>alert(1)</script>", "label": 1,
         "source": "s2", "attack_category": "xss", "split": "clean-test"},
        {"sample_id": "c4", "payload": "<img src=x onerror=alert(1)>", "label": 1,
         "source": "s2", "attack_category": "xss", "split": "clean-test"},
    ]


def _adv_records():
    return [
        {"variant_id": "a1", "seed_id": "s",
         "payload": "<svg onload=alert(1)>UNIQUESTRING123",
         "label": 1, "valid": True, "mutation_category": "encoding",
         "split": "adv_test"},
        {"variant_id": "a2", "seed_id": "s", "payload": "<b>not here</b>",
         "label": 1, "valid": True, "mutation_category": "case_variation",
         "split": "adv_test"},
        {"variant_id": "a3", "seed_id": "s", "payload": "<div>broken-payload",
         "label": 1, "valid": False, "mutation_category": "encoding",
         "split": "adv_test"},
        {"variant_id": "a4", "seed_id": "s", "payload": "just text benign",
         "label": 0, "valid": True, "mutation_category": "encoding",
         "split": "adv_test"},
    ]


def _mixed_table():
    return {
        "hello world": (0.1, 0),
        "good day friend": (0.9, 1),
        "<script>alert(1)</script>": (0.8, 1),
        "<img src=x onerror=alert(1)>": (0.2, 0),
        "<svg onload=alert(1)>UNIQUESTRING123": (0.1, 0),
        "<b>not here</b>": (0.9, 1),
        "<div>broken-payload": (0.05, 0),
        "just text benign": (0.9, 1),
    }


def _evaluation():
    from xssharden.evaluation import evaluate_arms

    table = _mixed_table()
    arms = {
        "baseline": PayloadFakeDetector(dict(table)),
        "naive": PayloadFakeDetector(dict(table)),
        "random_valid": PayloadFakeDetector(dict(table)),
        "selective": PayloadFakeDetector(dict(table)),
    }
    return evaluate_arms(arms, _clean_records(), _adv_records())


class TestMarkdownSections:
    def test_markdown_has_required_sections(self):
        from xssharden.reporting import render_markdown

        text = render_markdown(_evaluation())
        lowered = text.lower()
        for section in (
            "metadata",
            "counts",
            "arm comparison",
            "methodology",
            "safety",
            "caveat",
            "limitation",
        ):
            assert section in lowered, f"missing section {section!r}"

    def test_markdown_covers_all_four_arms(self):
        from xssharden.reporting import render_markdown

        text = render_markdown(_evaluation())
        for arm in ("baseline", "naive", "random_valid", "selective"):
            assert arm in text

    def test_markdown_preserves_denominators_and_valid_vs_raw(self):
        from xssharden.reporting import render_markdown

        text = render_markdown(_evaluation())
        assert "valid_malicious" in text or "V-ASR" in text or "valid malicious" in text.lower()
        assert "raw_evasion_rate" in text or "raw evasion" in text.lower()
        # Explicit denominators must appear (e.g. "1/2").
        assert "1/2" in text
        assert "2/3" in text or "2/4" in text or "/" in text

    def test_markdown_has_two_source_caveat(self):
        from xssharden.reporting import render_markdown

        text = render_markdown(_evaluation()).lower()
        assert "third" in text and "independent" in text and "source" in text
        assert "two" in text or "2" in text

    def test_markdown_includes_metadata(self):
        from xssharden.reporting import render_markdown

        text = render_markdown(
            _evaluation(),
            metadata={"experiment_id": "exp-001", "seed": 42},
        )
        assert "exp-001" in text
        assert "42" in text

    def test_markdown_does_not_include_raw_payloads(self):
        from xssharden.reporting import render_markdown

        text = render_markdown(_evaluation())
        assert "UNIQUESTRING123" not in text

    def test_markdown_deterministic(self):
        from xssharden.reporting import render_markdown

        first = render_markdown(_evaluation(), metadata={"b": 2, "a": 1})
        second = render_markdown(_evaluation(), metadata={"a": 1, "b": 2})
        assert first == second

    def test_markdown_does_not_mutate_inputs(self):
        from xssharden.reporting import render_markdown

        evaluation = _evaluation()
        before = copy.deepcopy(evaluation.to_dict())
        metadata = {"experiment_id": "exp-001"}
        before_meta = copy.deepcopy(metadata)
        render_markdown(evaluation, metadata=metadata)
        render_markdown(evaluation.to_dict(), metadata=metadata)
        assert evaluation.to_dict() == before
        assert metadata == before_meta


class TestHtmlReport:
    def test_html_has_sections_and_arms(self):
        from xssharden.reporting import render_html

        text = render_html(_evaluation())
        lowered = text.lower()
        assert "<html" in lowered
        for arm in ("baseline", "naive", "random_valid", "selective"):
            assert arm in text
        assert "v-asr" in lowered or "valid" in lowered

    def test_html_escapes_untrusted_metadata(self):
        from xssharden.reporting import render_html

        text = render_html(
            _evaluation(), metadata={"note": "<script>alert(1)</script>"}
        )
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;" in text

    def test_html_does_not_include_raw_payloads(self):
        from xssharden.reporting import render_html

        text = render_html(_evaluation())
        assert "UNIQUESTRING123" not in text

    def test_html_deterministic(self):
        from xssharden.reporting import render_html

        assert render_html(_evaluation()) == render_html(_evaluation())

    def test_html_does_not_mutate_inputs(self):
        from xssharden.reporting import render_html

        evaluation = _evaluation()
        before = copy.deepcopy(evaluation.to_dict())
        render_html(evaluation)
        assert evaluation.to_dict() == before


class TestWriteReport:
    def test_write_markdown_file(self, tmp_path):
        from xssharden.reporting import write_report

        out = tmp_path / "report.md"
        summary = write_report(_evaluation(), str(out))
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "baseline" in text
        assert summary["output"] == str(out)

    def test_write_html_file(self, tmp_path):
        from xssharden.reporting import write_report

        out = tmp_path / "report.html"
        write_report(_evaluation(), str(out))
        text = out.read_text(encoding="utf-8")
        assert "<html" in text.lower()

    def test_unsupported_suffix_refused_without_partial_file(self, tmp_path):
        from xssharden.reporting import write_report

        out = tmp_path / "report.pdf"
        with pytest.raises(ValueError):
            write_report(_evaluation(), str(out))
        assert not out.exists()

    def test_invalid_evaluation_shape_refused(self, tmp_path):
        from xssharden.reporting import write_report

        out = tmp_path / "report.md"
        with pytest.raises((TypeError, ValueError)):
            write_report({"arms": {}}, str(out))
        assert not out.exists()

    def test_invalid_evaluation_shape_leaves_no_partial_file(self, tmp_path):
        from xssharden.reporting import write_report

        out = tmp_path / "report.md"
        with pytest.raises((TypeError, ValueError)):
            write_report({"bogus": 1}, str(out))
        assert not out.exists()


class TestNoExecution:
    def test_reporting_module_never_executes_or_networks(self):
        import pathlib

        package = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src"
            / "xssharden"
            / "reporting"
        )
        sources = [path.read_text(encoding="utf-8") for path in package.glob("*.py")]
        assert sources, "reporting package has no modules"
        for source in sources:
            lowered = source.lower()
            for marker in (
                "playwright",
                "chromium",
                "ollama",
                "urllib",
                "requests",
                "socket",
                "subprocess",
                "os.system",
                "selenium",
                "validate_payload",
                "validate_variants",
                "predict_proba",
                ".fit(",
            ):
                assert marker not in lowered, f"forbidden marker {marker!r}"
            assert "eval(" not in lowered
            assert "exec(" not in lowered

    def test_reporting_does_not_import_validator(self, tmp_path):
        import sys

        for mod in [m for m in sys.modules if m.startswith("xssharden.validation")]:
            del sys.modules[mod]
        from xssharden.reporting import render_markdown

        render_markdown(_evaluation())
        assert not any(
            m.startswith("xssharden.validation") for m in sys.modules
        ), "reporting must not import the browser validator"


class TestReportCli:
    def test_report_cli_writes_markdown(self, tmp_path):
        import subprocess
        import sys

        evaluation = _evaluation().to_dict()
        src = tmp_path / "evaluation.json"
        src.write_text(json.dumps(evaluation), encoding="utf-8")
        out = tmp_path / "report.md"
        proc = subprocess.run(
            [sys.executable, "-m", "xssharden", "report",
             "--input", str(src), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        assert "baseline" in out.read_text(encoding="utf-8")

    def test_report_cli_rejects_bad_suffix(self, tmp_path):
        import subprocess
        import sys

        evaluation = _evaluation().to_dict()
        src = tmp_path / "evaluation.json"
        src.write_text(json.dumps(evaluation), encoding="utf-8")
        out = tmp_path / "report.pdf"
        proc = subprocess.run(
            [sys.executable, "-m", "xssharden", "report",
             "--input", str(src), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2
        assert "error" in proc.stderr.lower()
        assert not out.exists()

    def test_report_cli_missing_input_is_clear_error(self, tmp_path):
        import subprocess
        import sys

        out = tmp_path / "report.md"
        proc = subprocess.run(
            [sys.executable, "-m", "xssharden", "report",
             "--input", str(tmp_path / "nope.json"),
             "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2
        assert "error" in proc.stderr.lower()
        assert not out.exists()
