"""Focused tests for the safe multi-source dataset merger (TDD).

Covers ``xssharden.dataset.merge.merge_prepared_datasets`` and the
``dataset merge`` CLI: three-source success, two-source refusal (the two
current local artifacts must stay refused), duplicate input paths,
cross-file duplicates via the shared cleaner, group-atomic splitting,
deterministic byte-identical output, provenance preservation, CLI wiring,
and never executing payloads (inert text only, no network/browser).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _rec(sample_id, payload, label, source, category="xss", split="train", **extra):
    record = {
        "sample_id": sample_id,
        "payload": payload,
        "label": label,
        "source": source,
        "attack_category": category,
        "split": split,
    }
    record.update(extra)
    return record


def _write_records(path: Path, records: list[dict]) -> None:
    from xssharden.dataset.io import write_records

    write_records(records, str(path))


def _read_out(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _three_inputs(tmp_path: Path) -> list[Path]:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    c = tmp_path / "c.jsonl"
    _write_records(
        a,
        [
            _rec("a-1", "<script>alert(1)</script>", 1, "source_a"),
            _rec("a-2", "hello benign", 0, "source_a", "benign"),
        ],
    )
    _write_records(
        b,
        [
            _rec("b-1", "<svg onload=alert(1)>", 1, "source_b"),
            _rec("b-2", "plain text", 0, "source_b", "benign"),
        ],
    )
    _write_records(
        c,
        [
            _rec("c-1", "\"><img src=x onerror=alert(1)>", 1, "source_c"),
            _rec("c-2", "another benign", 0, "source_c", "benign"),
        ],
    )
    return [a, b, c]


class TestThreeSourceSuccess:
    def test_merge_three_sources_writes_summary(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets

        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        summary = merge_prepared_datasets(
            [str(p) for p in inputs], str(out), seed=42
        )
        assert summary["written"] == 6
        assert summary["duplicates_removed"] == 0
        assert sorted(summary["sources"]) == ["source_a", "source_b", "source_c"]
        assert summary["seed"] == 42
        assert sum(summary["splits"].values()) == 6
        records = _read_out(out)
        assert len(records) == 6
        assert sorted({r["source"] for r in records}) == [
            "source_a",
            "source_b",
            "source_c",
        ]

    def test_merge_reports_counts_sources_splits(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets

        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        summary = merge_prepared_datasets(
            [str(p) for p in inputs], str(out), seed=42
        )
        assert summary["total_rows"] == 6
        assert "per_source_counts" in summary
        assert summary["per_source_counts"] == {
            "source_a": 2,
            "source_b": 2,
            "source_c": 2,
        }
        assert "per_input_counts" in summary
        assert summary["per_input_counts"] == [2, 2, 2]


class TestRefusals:
    def test_two_sources_refused_before_writing(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets
        import pytest

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _write_records(a, [_rec("a-1", "<script>alert(1)</script>", 1, "source_a")])
        _write_records(b, [_rec("b-1", "<svg onload=alert(1)>", 1, "source_b")])
        out = tmp_path / "merged.jsonl"
        with pytest.raises(ValueError, match="at least 3 distinct"):
            merge_prepared_datasets([str(a), str(b)], str(out))
        assert not out.exists()

    def test_single_source_refused_before_writing(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets
        import pytest

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _write_records(a, [_rec("a-1", "payload one", 1, "only_src")])
        _write_records(b, [_rec("b-1", "payload two", 0, "only_src", "benign")])
        out = tmp_path / "merged.jsonl"
        with pytest.raises(ValueError, match="at least 3 distinct"):
            merge_prepared_datasets([str(a), str(b)], str(out))
        assert not out.exists()

    def test_duplicate_input_paths_refused(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets
        import pytest

        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        with pytest.raises(ValueError, match="[Dd]uplicate input"):
            merge_prepared_datasets(
                [str(inputs[0]), str(inputs[0]), str(inputs[1]), str(inputs[2])],
                str(out),
            )
        assert not out.exists()

    def test_duplicate_paths_by_spelling_refused(self, tmp_path, monkeypatch):
        from xssharden.dataset.merge import merge_prepared_datasets
        import pytest

        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="[Dd]uplicate input"):
            merge_prepared_datasets(
                ["a.jsonl", str(inputs[0]), str(inputs[1]), str(inputs[2])],
                str(out),
            )
        assert not out.exists()

    def test_output_path_cannot_overwrite_an_input(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets
        import pytest

        inputs = _three_inputs(tmp_path)
        with pytest.raises(ValueError, match="must not overwrite"):
            merge_prepared_datasets(
                [str(p) for p in inputs], str(inputs[0])
            )

    def test_missing_input_refused(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets
        import pytest

        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        with pytest.raises(FileNotFoundError):
            merge_prepared_datasets(
                [str(inputs[0]), str(tmp_path / "nope.jsonl"), str(inputs[2])],
                str(out),
            )
        assert not out.exists()

    def test_invalid_schema_input_refused(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets
        import pytest

        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"payload": "x"}\n', encoding="utf-8")
        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        with pytest.raises((ValueError, TypeError)):
            merge_prepared_datasets(
                [str(bad), str(inputs[1]), str(inputs[2])], str(out)
            )
        assert not out.exists()


class TestDedupAndSplit:
    def test_cross_file_duplicates_removed(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        c = tmp_path / "c.jsonl"
        dup = _rec("a-1", "<script>alert(1)</script>", 1, "source_a")
        _write_records(a, [dup, _rec("a-2", "hello benign", 0, "source_a", "benign")])
        # Same normalized record (same source/payload/label/category/split).
        _write_records(
            b,
            [
                dict(dup, sample_id="b-dup"),
                _rec("b-2", "plain text", 0, "source_b", "benign"),
            ],
        )
        _write_records(
            c, [_rec("c-1", "<svg onload=alert(2)>", 1, "source_c")]
        )
        out = tmp_path / "merged.jsonl"
        summary = merge_prepared_datasets(
            [str(a), str(b), str(c)], str(out), seed=42
        )
        assert summary["total_rows"] == 5
        assert summary["duplicates_removed"] == 1
        assert summary["written"] == 4
        records = _read_out(out)
        assert len(records) == 4
        # First occurrence wins.
        kept = [r for r in records if r["payload"] == "<script>alert(1)</script>"]
        assert len(kept) == 1
        assert kept[0]["sample_id"] == "a-1"

    def test_group_atomic_split_keeps_sources_together(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets

        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        merge_prepared_datasets([str(p) for p in inputs], str(out), seed=7)
        records = _read_out(out)
        by_source: dict[str, set[str]] = {}
        for record in records:
            by_source.setdefault(record["source"], set()).add(record["split"])
        for source, splits in by_source.items():
            assert len(splits) == 1, f"{source} split across {splits}"


class TestDeterminismAndProvenance:
    def test_deterministic_byte_identical_rerun(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets

        inputs = _three_inputs(tmp_path)
        out1 = tmp_path / "m1.jsonl"
        out2 = tmp_path / "m2.jsonl"
        merge_prepared_datasets([str(p) for p in inputs], str(out1), seed=42)
        merge_prepared_datasets([str(p) for p in inputs], str(out2), seed=42)
        assert out1.read_bytes() == out2.read_bytes()

    def test_provenance_preserved(self, tmp_path):
        from xssharden.dataset.merge import merge_prepared_datasets

        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        c = tmp_path / "c.jsonl"
        _write_records(
            a,
            [
                _rec(
                    "a-1",
                    "<script>alert(1)</script>",
                    1,
                    "source_a",
                    raw_label="1",
                    raw_row_number=3,
                    context_target="unknown",
                )
            ],
        )
        _write_records(
            b, [_rec("b-1", "<svg onload=alert(1)>", 1, "source_b", raw_label="1")]
        )
        _write_records(
            c, [_rec("c-1", "benign text", 0, "source_c", "benign", custom="kept")]
        )
        out = tmp_path / "merged.jsonl"
        merge_prepared_datasets([str(a), str(b), str(c)], str(out), seed=42)
        records = _read_out(out)
        by_id = {r["sample_id"]: r for r in records}
        assert by_id["a-1"]["raw_label"] == "1"
        assert by_id["a-1"]["raw_row_number"] == 3
        assert by_id["a-1"]["context_target"] == "unknown"
        assert by_id["b-1"]["raw_label"] == "1"
        assert by_id["c-1"]["custom"] == "kept"


class TestCLI:
    def test_cli_merge_success(self, tmp_path):
        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        cmd = (
            [sys.executable, "-m", "xssharden", "dataset", "merge"]
            + [item for p in inputs for item in ("--input", str(p))]
            + ["--output", str(out), "--seed", "42"]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        records = _read_out(out)
        assert len(records) == 6

    def test_cli_two_sources_refused(self, tmp_path):
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _write_records(a, [_rec("a-1", "<script>alert(1)</script>", 1, "source_a")])
        _write_records(b, [_rec("b-1", "<svg onload=alert(1)>", 1, "source_b")])
        out = tmp_path / "merged.jsonl"
        cmd = [
            sys.executable,
            "-m",
            "xssharden",
            "dataset",
            "merge",
            "--input",
            str(a),
            "--input",
            str(b),
            "--output",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 2
        assert "at least 3 distinct" in proc.stderr.lower()
        assert not out.exists()

    def test_cli_duplicate_paths_refused(self, tmp_path):
        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        cmd = [
            sys.executable,
            "-m",
            "xssharden",
            "dataset",
            "merge",
            "--input",
            str(inputs[0]),
            "--input",
            str(inputs[0]),
            "--input",
            str(inputs[1]),
            "--input",
            str(inputs[2]),
            "--output",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 2
        assert "duplicate" in proc.stderr.lower()
        assert not out.exists()


class TestNoExecution:
    def test_merge_module_never_executes_or_networks(self):
        text = Path("src/xssharden/dataset/merge.py").read_text(encoding="utf-8")
        lowered = text.lower()
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
        ):
            assert marker not in lowered, f"forbidden marker {marker!r}"
        assert "exec(" not in lowered
        assert "eval(" not in lowered

    def test_merge_does_not_import_validator(self, tmp_path):
        import sys

        for mod in [m for m in sys.modules if m.startswith("xssharden.validation")]:
            del sys.modules[mod]
        from xssharden.dataset.merge import merge_prepared_datasets

        inputs = _three_inputs(tmp_path)
        out = tmp_path / "merged.jsonl"
        merge_prepared_datasets([str(p) for p in inputs], str(out), seed=42)
        assert not any(
            m.startswith("xssharden.validation") for m in sys.modules
        ), "merge must not import the browser validator"
