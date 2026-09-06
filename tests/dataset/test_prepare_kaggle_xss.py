"""Focused tests for the Kaggle XSS dataset adapter (TDD).

Covers ``xssharden.dataset.prepare_kaggle_xss.prepare_kaggle_xss``:
``XSS_dataset.csv`` (``Sentence`` -> payload, ``Label`` 0/1 -> benign/xss),
blank-payload skip/count, invalid-label and missing-column rejection,
provenance (``source=kaggle_xss_dataset``, ``raw_label``,
``raw_row_number``, original source ID, ``context_target=unknown``,
``license_status=unknown``), deterministic content IDs, deduplication via
shared cleaning, group-atomic splitting, file writing, CLI wiring, and
never executing payloads (inert text only).
"""

from __future__ import annotations

import csv
import hashlib
import json


SOURCE = "kaggle_xss_dataset"


def _write_kaggle_csv(path, rows: list[dict], fieldnames=None) -> None:
    fieldnames = fieldnames or ["", "Sentence", "Label"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _kaggle_row(idx: str, sentence: str, label: str) -> dict:
    return {"": idx, "Sentence": sentence, "Label": label}


def _read_out(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _expected_id(source: str, label: int, payload: str) -> str:
    digest = hashlib.sha256(
        f"{source}|{label}|{payload}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{source}-{digest}"


def _mixed_rows():
    return [
        _kaggle_row("0", "hello world", "0"),
        _kaggle_row("1", "<script>alert(1)</script>", "1"),
        _kaggle_row("2", "plain text", "0"),
        _kaggle_row("3", "<svg onload=alert(1)>", "1"),
    ]


class TestBasicMapping:
    def test_csv_basic_flow_and_labels(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, _mixed_rows())
        summary = prepare_kaggle_xss(str(src), str(dst))
        assert summary["written"] == 4
        records = _read_out(dst)
        assert len(records) == 4
        by_payload = {r["payload"]: r for r in records}
        assert by_payload["hello world"]["label"] == 0
        assert by_payload["hello world"]["attack_category"] == "benign"
        assert by_payload["<script>alert(1)</script>"]["label"] == 1
        assert by_payload["<script>alert(1)</script>"]["attack_category"] == "xss"
        assert by_payload["plain text"]["label"] == 0
        assert by_payload["<svg onload=alert(1)>"]["label"] == 1

    def test_sentence_maps_to_payload_verbatim(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [_kaggle_row("7", "  spaced <b>hi</b>  ", "0"),
                                _kaggle_row("8", "<img src=x onerror=alert(1)>", "1")])
        prepare_kaggle_xss(str(src), str(dst))
        by_payload = {r["payload"]: r for r in _read_out(dst)}
        assert "  spaced <b>hi</b>  " in by_payload

    def test_int_labels_accepted(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        with open(src, "w", encoding="utf-8", newline="") as handle:
            handle.write(",Sentence,Label\n")
            handle.write("0,hello,0\n")
            handle.write("1,<script>alert(1)</script>,1\n")
        summary = prepare_kaggle_xss(str(src), str(dst))
        assert summary["kept_benign"] == 1
        assert summary["kept_xss"] == 1

    def test_output_is_valid_project_schema(self, tmp_path):
        from xssharden.dataset.io import read_records
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, _mixed_rows())
        prepare_kaggle_xss(str(src), str(dst))
        loaded = read_records(str(dst))
        assert len(loaded) == 4
        for rec in loaded:
            assert isinstance(rec["sample_id"], str) and rec["sample_id"]
            assert isinstance(rec["payload"], str) and rec["payload"].strip()
            assert rec["label"] in (0, 1)
            assert rec["source"] == SOURCE
            assert rec["attack_category"] in ("benign", "xss")
            assert rec["split"] in ("train", "validation", "clean-test")

    def test_never_executes_payloads(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        evil = "__import__('os').system('echo PWNED')"
        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [_kaggle_row("0", evil, "1"),
                                _kaggle_row("1", "harmless", "0")])
        prepare_kaggle_xss(str(src), str(dst))
        records = _read_out(dst)
        assert any(r["payload"] == evil for r in records)


class TestBlankPayloadsSkipped:
    def test_blank_payloads_skipped_and_counted(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [
            _kaggle_row("0", "hello", "0"),
            _kaggle_row("1", "", "1"),
            _kaggle_row("2", "   ", "0"),
            _kaggle_row("3", "\t\n ", "1"),
            _kaggle_row("4", "<script>alert(1)</script>", "1"),
        ])
        summary = prepare_kaggle_xss(str(src), str(dst))
        assert summary["total_rows"] == 5
        assert summary["skipped_blank"] == 3
        assert summary["kept_total"] == 2
        assert summary["written"] == 2
        records = _read_out(dst)
        assert {r["payload"] for r in records} == {"hello", "<script>alert(1)</script>"}

    def test_blank_rows_do_not_break_row_numbering(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [
            _kaggle_row("0", "hello", "0"),
            _kaggle_row("1", "", "1"),
            _kaggle_row("2", "<svg onload=alert(1)>", "1"),
        ])
        prepare_kaggle_xss(str(src), str(dst))
        by_payload = {r["payload"]: r for r in _read_out(dst)}
        assert by_payload["hello"]["raw_row_number"] == 1
        assert by_payload["<svg onload=alert(1)>"]["raw_row_number"] == 3


class TestInvalidLabelsRejected:
    def test_unsupported_label_rejected(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [_kaggle_row("0", "hello", "0"),
                                _kaggle_row("1", "<script>alert(1)</script>", "maybe")])
        try:
            prepare_kaggle_xss(str(src), str(dst))
        except ValueError as exc:
            assert "label" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for unsupported label")

    def test_numeric_label_out_of_range_rejected(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [_kaggle_row("0", "hello", "0"),
                                _kaggle_row("1", "evil", "2")])
        try:
            prepare_kaggle_xss(str(src), str(dst))
        except ValueError as exc:
            assert "label" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for label 2")

    def test_empty_label_rejected(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [_kaggle_row("0", "hello", "0"),
                                _kaggle_row("1", "evil", "")])
        try:
            prepare_kaggle_xss(str(src), str(dst))
        except ValueError as exc:
            assert "label" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for empty label")


class TestMissingColumnsRejected:
    def test_missing_sentence_column_rejected(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [{"": "0", "Label": "0"}],
                          fieldnames=["", "Label"])
        try:
            prepare_kaggle_xss(str(src), str(dst))
        except ValueError as exc:
            assert "sentence" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for missing Sentence")

    def test_missing_label_column_rejected(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [{"": "0", "Sentence": "hi"}],
                          fieldnames=["", "Sentence"])
        try:
            prepare_kaggle_xss(str(src), str(dst))
        except ValueError as exc:
            assert "label" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for missing Label")

    def test_unsupported_suffix_rejected(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.txt"
        src.write_text("Sentence,Label\nhello,0\n", encoding="utf-8")
        try:
            prepare_kaggle_xss(str(src), str(tmp_path / "o.jsonl"))
        except ValueError as exc:
            assert "support" in str(exc).lower() or "csv" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for unsupported suffix")

    def test_missing_input_raises(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        try:
            prepare_kaggle_xss(str(tmp_path / "nope.csv"), str(tmp_path / "o.jsonl"))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError")


class TestProvenance:
    def test_source_and_audit_fields(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, _mixed_rows())
        prepare_kaggle_xss(str(src), str(dst))
        records = _read_out(dst)
        by_payload = {r["payload"]: r for r in records}
        for rec in records:
            assert rec["source"] == SOURCE
            assert rec["context_target"] == "unknown"
            assert rec["license_status"] == "unknown"
            assert isinstance(rec["raw_label"], str) and rec["raw_label"] in ("0", "1")
            assert isinstance(rec["raw_row_number"], int) and rec["raw_row_number"] >= 1
            assert "raw_source_id" in rec
        assert by_payload["hello world"]["raw_row_number"] == 1
        assert by_payload["hello world"]["raw_label"] == "0"
        assert by_payload["hello world"]["raw_source_id"] == "0"
        assert by_payload["<svg onload=alert(1)>"]["raw_row_number"] == 4
        assert by_payload["<svg onload=alert(1)>"]["raw_source_id"] == "3"

    def test_raw_label_preserves_original_spelling(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        with open(src, "w", encoding="utf-8", newline="") as handle:
            handle.write(",Sentence,Label\n")
            handle.write("10,hello, 0 \n")
            handle.write("11,<script>alert(1)</script>, 1 \n")
        prepare_kaggle_xss(str(src), str(dst))
        by_payload = {r["payload"]: r for r in _read_out(dst)}
        assert by_payload["hello"]["label"] == 0
        assert by_payload["hello"]["raw_source_id"] == "10"


class TestDeterministicIds:
    def test_ids_content_based_and_reorder_stable(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        rows = _mixed_rows()
        src_a = tmp_path / "a.csv"
        src_b = tmp_path / "b.csv"
        dst_a = tmp_path / "a.jsonl"
        dst_b = tmp_path / "b.jsonl"
        _write_kaggle_csv(src_a, rows)
        _write_kaggle_csv(src_b, list(reversed(rows)))
        prepare_kaggle_xss(str(src_a), str(dst_a), seed=42)
        prepare_kaggle_xss(str(src_b), str(dst_b), seed=42)
        recs_a = {r["payload"]: r for r in _read_out(dst_a)}
        recs_b = {r["payload"]: r for r in _read_out(dst_b)}
        assert set(recs_a) == set(recs_b)
        for payload in recs_a:
            assert recs_a[payload]["sample_id"] == recs_b[payload]["sample_id"]
            assert recs_a[payload]["sample_id"] == _expected_id(
                SOURCE, recs_a[payload]["label"], payload
            )

    def test_ids_include_source_and_differ_from_row_position(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, _mixed_rows())
        prepare_kaggle_xss(str(src), str(dst))
        for rec in _read_out(dst):
            assert rec["sample_id"].startswith(f"{SOURCE}-")
            assert len(rec["sample_id"]) == len(SOURCE) + 1 + 16


class TestDedupAndSplit:
    def test_normalized_duplicate_removed(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, [
            _kaggle_row("0", "<SCRIPT>alert(1)</SCRIPT>", "1"),
            _kaggle_row("1", "<script>alert(1)</script>", "1"),
            _kaggle_row("2", "clean row", "0"),
        ])
        summary = prepare_kaggle_xss(str(src), str(dst))
        assert summary["duplicates_removed"] == 1
        assert summary["written"] == 2

    def test_group_atomic_single_source(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        rows = [_kaggle_row(str(i), f"benign-{i}", "0") for i in range(15)]
        rows += [_kaggle_row(str(15 + i), f"<script>alert({i})</script>", "1")
                 for i in range(15)]
        _write_kaggle_csv(src, rows)
        prepare_kaggle_xss(str(src), str(dst), seed=42)
        records = _read_out(dst)
        assert {r["split"] for r in records} <= {"train", "validation", "clean-test"}
        assert len({r["split"] for r in records}) == 1

    def test_split_deterministic_for_seed(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        first = tmp_path / "first.jsonl"
        second = tmp_path / "second.jsonl"
        _write_kaggle_csv(src, _mixed_rows())
        prepare_kaggle_xss(str(src), str(first), seed=7)
        prepare_kaggle_xss(str(src), str(second), seed=7)
        assert first.read_bytes() == second.read_bytes()

    def test_summary_counts(self, tmp_path):
        from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, _mixed_rows())
        summary = prepare_kaggle_xss(str(src), str(dst), seed=42)
        assert summary["total_rows"] == 4
        assert summary["kept_benign"] == 2
        assert summary["kept_xss"] == 2
        assert summary["kept_total"] == 4
        assert summary["skipped_blank"] == 0
        assert summary["duplicates_removed"] == 0
        assert summary["written"] == 4
        assert summary["seed"] == 42
        assert summary["source"] == SOURCE
        assert summary["input"] == str(src)
        assert summary["output"] == str(dst)
        assert sum(summary["splits"].values()) == 4


class TestCli:
    def test_prepare_kaggle_xss_cli(self, tmp_path):
        from xssharden.cli import main

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_kaggle_csv(src, _mixed_rows())
        rc = main(["dataset", "prepare-kaggle-xss",
                   "--input", str(src), "--output", str(dst)])
        assert rc == 0
        records = _read_out(dst)
        assert len(records) == 4
        assert all(r["source"] == SOURCE for r in records)
