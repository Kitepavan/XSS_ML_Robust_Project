"""Tests for dataset JSONL/CSV I/O and the `dataset build` command (Task 5)."""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> dict:
    """Return a fully-formed record, allowing selective overrides."""
    base = {
        "sample_id": "S001",
        "payload": "<script>alert(1)</script>",
        "label": 1,
        "source": "github",
        "attack_category": "reflected_html",
        "split": "train",
    }
    base.update(overrides)
    return base


def _make_records(n: int) -> list[dict]:
    """Return *n* distinct valid records with sequential sample_ids."""
    return [
        _make_record(
            sample_id=f"S{i:04d}",
            payload=f"<script>alert({i})</script>",
            label=i % 2,
        )
        for i in range(n)
    ]


# ===================================================================
# Section 1 – JSONL round-tripping
# ===================================================================

class TestJsonlRoundTrip:
    """read_records/write_records must round-trip JSONL losslessly."""

    def test_jsonl_round_trip_preserves_records(self, tmp_path):
        from xssharden.dataset.io import read_records, write_records

        records = _make_records(5)
        path = tmp_path / "data.jsonl"
        write_records(records, str(path))
        loaded = read_records(str(path))
        assert loaded == records

    def test_jsonl_preserves_record_order(self, tmp_path):
        from xssharden.dataset.io import read_records, write_records

        records = _make_records(10)
        path = tmp_path / "ordered.jsonl"
        write_records(records, str(path))
        loaded = read_records(str(path))
        assert [r["sample_id"] for r in loaded] == [r["sample_id"] for r in records]

    def test_jsonl_skips_blank_lines(self, tmp_path):
        from xssharden.dataset.io import read_records

        path = tmp_path / "blank.jsonl"
        path.write_text(
            json.dumps(_make_record(sample_id="S001")) + "\n"
            + "\n"
            + json.dumps(_make_record(sample_id="S002")) + "\n",
            encoding="utf-8",
        )
        loaded = read_records(str(path))
        assert [r["sample_id"] for r in loaded] == ["S001", "S002"]

    def test_read_jsonl_validates_required_fields(self, tmp_path):
        from xssharden.dataset.io import read_records

        bad = {"sample_id": "S001", "payload": "<script>alert(1)</script>"}
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises((ValueError, TypeError)):
            read_records(str(path))


# ===================================================================
# Section 2 – CSV round-tripping
# ===================================================================

class TestCsvRoundTrip:
    """read_records/write_records must round-trip CSV losslessly."""

    def test_csv_round_trip_preserves_records(self, tmp_path):
        from xssharden.dataset.io import read_records, write_records

        records = _make_records(5)
        path = tmp_path / "data.csv"
        write_records(records, str(path))
        loaded = read_records(str(path))
        assert loaded == records

    def test_csv_preserves_record_order(self, tmp_path):
        from xssharden.dataset.io import read_records, write_records

        records = _make_records(10)
        path = tmp_path / "ordered.csv"
        write_records(records, str(path))
        loaded = read_records(str(path))
        assert [r["sample_id"] for r in loaded] == [r["sample_id"] for r in records]

    def test_csv_label_read_back_as_int(self, tmp_path):
        from xssharden.dataset.io import read_records, write_records

        records = _make_records(3)
        path = tmp_path / "labels.csv"
        write_records(records, str(path))
        loaded = read_records(str(path))
        for record in loaded:
            assert isinstance(record["label"], int)

    def test_read_csv_validates_required_fields(self, tmp_path):
        from xssharden.dataset.io import read_records

        path = tmp_path / "bad.csv"
        path.write_text("sample_id,payload\nS001,<script>alert(1)</script>\n", encoding="utf-8")
        with pytest.raises((ValueError, TypeError)):
            read_records(str(path))


# ===================================================================
# Section 3 – Validation and error handling
# ===================================================================

class TestIoValidation:
    """I/O must validate records and reject unsupported input."""

    def test_write_records_validates_before_writing(self, tmp_path):
        from xssharden.dataset.io import write_records

        bad = _make_record()
        del bad["payload"]
        path = tmp_path / "out.jsonl"
        with pytest.raises((ValueError, TypeError)):
            write_records([bad], str(path))
        assert not path.exists()

    def test_write_records_rejects_non_dict(self, tmp_path):
        from xssharden.dataset.io import write_records

        path = tmp_path / "out.jsonl"
        with pytest.raises(TypeError):
            write_records(["not-a-dict"], str(path))  # type: ignore[list-item]

    def test_read_records_rejects_unsupported_extension(self, tmp_path):
        from xssharden.dataset.io import read_records

        path = tmp_path / "data.txt"
        path.write_text("hello\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            read_records(str(path))

    def test_write_records_rejects_unsupported_extension(self, tmp_path):
        from xssharden.dataset.io import write_records

        path = tmp_path / "data.txt"
        with pytest.raises(ValueError, match="Unsupported"):
            write_records(_make_records(2), str(path))

    def test_read_records_missing_file_raises(self, tmp_path):
        from xssharden.dataset.io import read_records

        with pytest.raises(FileNotFoundError):
            read_records(str(tmp_path / "does-not-exist.jsonl"))

    def test_write_then_read_is_deterministic(self, tmp_path):
        from xssharden.dataset.io import read_records, write_records

        records = _make_records(5)
        first = tmp_path / "first.jsonl"
        second = tmp_path / "second.jsonl"
        write_records(records, str(first))
        write_records(records, str(second))
        assert first.read_bytes() == second.read_bytes()
        assert read_records(str(first)) == read_records(str(second))


# ===================================================================
# Section 4 – `dataset build` command output
# ===================================================================

class TestDatasetBuildCommand:
    """The `dataset build` subcommand must produce a readable split output."""

    def test_dataset_build_writes_split_output(self, tmp_path, capsys):
        from xssharden.cli import main
        from xssharden.dataset.io import read_records, write_records

        src = tmp_path / "input.jsonl"
        dst = tmp_path / "built.jsonl"
        write_records(_make_records(30), str(src))

        rc = main(["dataset", "build", "--input", str(src), "--output", str(dst)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "30" in out

        built = read_records(str(dst))
        assert len(built) > 0
        assert {r["split"] for r in built} <= {"train", "validation", "clean-test"}

    def test_dataset_build_is_deterministic(self, tmp_path, capsys):
        from xssharden.cli import main
        from xssharden.dataset.io import write_records

        src = tmp_path / "input.jsonl"
        first = tmp_path / "first.jsonl"
        second = tmp_path / "second.jsonl"
        write_records(_make_records(30), str(src))

        assert main(["dataset", "build", "--input", str(src), "--output", str(first)]) == 0
        capsys.readouterr()
        assert main(["dataset", "build", "--input", str(src), "--output", str(second)]) == 0
        capsys.readouterr()
        assert first.read_bytes() == second.read_bytes()

    def test_dataset_build_deduplicates_input(self, tmp_path, capsys):
        from xssharden.cli import main
        from xssharden.dataset.io import read_records, write_records

        record = _make_record()
        src = tmp_path / "dupes.jsonl"
        dst = tmp_path / "deduped.jsonl"
        write_records([record, dict(record)], str(src))

        assert main(["dataset", "build", "--input", str(src), "--output", str(dst)]) == 0
        capsys.readouterr()
        built = read_records(str(dst))
        assert len(built) == 1
