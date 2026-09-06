"""Focused tests for the generic labeled-payload source adapter (TDD).

Covers ``xssharden.dataset.prepare_labeled_payloads.prepare_labeled_payloads``:
CSV/JSONL inputs, 0/1 + benign/xss (case-insensitive) labels, audit
summary, deduplication via existing cleaning, deterministic content-based
IDs including source_name, rejection cases, and group-atomic split
boundaries. Payloads are treated as inert text and never executed.
"""

from __future__ import annotations

import csv
import hashlib
import json

import pytest


def _write_csv(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["payload", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _expected_id(source: str, label: int, payload: str) -> str:
    digest = hashlib.sha256(
        f"{source}|{label}|{payload}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{source}-{digest}"


def _read_out(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mixed_rows():
    return [
        {"payload": "hello world", "label": "benign"},
        {"payload": "<script>alert(1)</script>", "label": "xss"},
        {"payload": "plain text", "label": "0"},
        {"payload": "<svg onload=alert(1)>", "label": "1"},
    ]


class TestCsvAndJsonlInputs:
    def test_csv_basic_flow_and_provenance(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(src, _mixed_rows())
        summary = prepare_labeled_payloads(str(src), str(dst), "second_src")
        assert summary["written"] == 4
        records = _read_out(dst)
        assert len(records) == 4
        by_payload = {r["payload"]: r for r in records}
        assert by_payload["hello world"]["label"] == 0
        assert by_payload["hello world"]["attack_category"] == "benign"
        assert by_payload["<script>alert(1)</script>"]["label"] == 1
        assert by_payload["<script>alert(1)</script>"]["attack_category"] == "xss"
        for rec in records:
            assert rec["source"] == "second_src"
            assert rec["context_target"] == "unknown"
            assert isinstance(rec["raw_label"], str) and rec["raw_label"]
            assert isinstance(rec["raw_row_number"], int) and rec["raw_row_number"] >= 1
        # raw_row_number tracks input order starting at 1.
        assert by_payload["hello world"]["raw_row_number"] == 1
        assert by_payload["<svg onload=alert(1)>"]["raw_row_number"] == 4
        # raw_label preserves the original spelling.
        assert by_payload["hello world"]["raw_label"] == "benign"
        assert by_payload["plain text"]["raw_label"] == "0"

    def test_jsonl_basic_flow(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.jsonl"
        dst = tmp_path / "out.jsonl"
        _write_jsonl(src, _mixed_rows())
        summary = prepare_labeled_payloads(str(src), str(dst), "second_src")
        assert summary["written"] == 4
        records = _read_out(dst)
        assert {r["label"] for r in records} == {0, 1}
        for rec in records:
            assert rec["source"] == "second_src"
            assert rec["context_target"] == "unknown"

    def test_jsonl_int_labels_accepted(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.jsonl"
        dst = tmp_path / "out.jsonl"
        _write_jsonl(
            src,
            [
                {"payload": "a", "label": 0},
                {"payload": "b", "label": 1},
            ],
        )
        prepare_labeled_payloads(str(src), str(dst), "second_src")
        records = _read_out(dst)
        assert {r["payload"]: r["label"] for r in records} == {"a": 0, "b": 1}

    def test_csv_numeric_string_labels(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(
            src,
            [
                {"payload": "a", "label": "0"},
                {"payload": "b", "label": "1"},
            ],
        )
        prepare_labeled_payloads(str(src), str(dst), "second_src")
        records = _read_out(dst)
        assert {r["payload"]: r["label"] for r in records} == {"a": 0, "b": 1}

    def test_output_is_valid_project_schema(self, tmp_path):
        from xssharden.dataset.io import read_records
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(src, _mixed_rows())
        prepare_labeled_payloads(str(src), str(dst), "second_src")
        loaded = read_records(str(dst))
        assert len(loaded) == 4
        for rec in loaded:
            assert isinstance(rec["sample_id"], str) and rec["sample_id"]
            assert isinstance(rec["payload"], str) and rec["payload"].strip()
            assert rec["label"] in (0, 1)
            assert rec["source"] == "second_src"
            assert rec["attack_category"] in ("benign", "xss")
            assert rec["split"] in ("train", "validation", "clean-test")

    def test_never_executes_payloads(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        evil = "__import__('os').system('echo PWNED')"
        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(
            src,
            [
                {"payload": evil, "label": "xss"},
                {"payload": "harmless", "label": "benign"},
            ],
        )
        prepare_labeled_payloads(str(src), str(dst), "second_src")
        records = _read_out(dst)
        assert any(r["payload"] == evil for r in records)


class TestLabelVariants:
    @pytest.mark.parametrize(
        "label,expected",
        [
            (0, 0),
            (1, 1),
            ("0", 0),
            ("1", 1),
            ("benign", 0),
            ("xss", 1),
            ("Benign", 0),
            ("XSS", 1),
            ("BENIGN", 0),
            ("XsS", 1),
            ("  benign  ", 0),
            ("  1 ", 1),
        ],
    )
    def test_label_variants_accepted(self, tmp_path, label, expected):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.jsonl"
        dst = tmp_path / "out.jsonl"
        other = 1 if expected == 0 else 0
        other_raw = "xss" if other == 1 else "benign"
        _write_jsonl(
            src,
            [
                {"payload": "probe-payload", "label": label},
                {"payload": "other-payload", "label": other_raw},
            ],
        )
        prepare_labeled_payloads(str(src), str(dst), "second_src")
        records = _read_out(dst)
        by_payload = {r["payload"]: r["label"] for r in records}
        assert by_payload["probe-payload"] == expected
        assert by_payload["other-payload"] == other


class TestAuditSummary:
    def test_summary_counts(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(src, _mixed_rows())
        summary = prepare_labeled_payloads(str(src), str(dst), "second_src", seed=42)
        assert summary["total_rows"] == 4
        assert summary["kept_benign"] == 2
        assert summary["kept_xss"] == 2
        assert summary["kept_total"] == 4
        assert summary["duplicates_removed"] == 0
        assert summary["written"] == 4
        assert summary["seed"] == 42
        assert summary["source"] == "second_src"
        assert summary["input"] == str(src)
        assert summary["output"] == str(dst)
        assert sum(summary["splits"].values()) == 4


class TestDeduplication:
    def test_normalized_duplicate_removed(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(
            src,
            [
                {"payload": "<SCRIPT>alert(1)</SCRIPT>", "label": "xss"},
                {"payload": "<script>alert(1)</script>", "label": "xss"},
                {"payload": "clean row", "label": "benign"},
            ],
        )
        summary = prepare_labeled_payloads(str(src), str(dst), "second_src")
        assert summary["duplicates_removed"] == 1
        assert summary["written"] == 2

    def test_distinct_payloads_kept(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(
            src,
            [
                {"payload": "<script>alert(1)</script>", "label": "xss"},
                {"payload": "<script>alert(2)</script>", "label": "xss"},
                {"payload": "hello", "label": "benign"},
            ],
        )
        summary = prepare_labeled_payloads(str(src), str(dst), "second_src")
        assert summary["duplicates_removed"] == 0
        assert summary["written"] == 3


class TestDeterministicIds:
    def test_ids_content_based_and_reorder_stable(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        rows = _mixed_rows()
        src_a = tmp_path / "a.csv"
        src_b = tmp_path / "b.csv"
        dst_a = tmp_path / "a.jsonl"
        dst_b = tmp_path / "b.jsonl"
        _write_csv(src_a, rows)
        _write_csv(src_b, list(reversed(rows)))
        prepare_labeled_payloads(str(src_a), str(dst_a), "second_src", seed=42)
        prepare_labeled_payloads(str(src_b), str(dst_b), "second_src", seed=42)
        recs_a = {r["payload"]: r for r in _read_out(dst_a)}
        recs_b = {r["payload"]: r for r in _read_out(dst_b)}
        assert set(recs_a) == set(recs_b)
        for payload in recs_a:
            assert recs_a[payload]["sample_id"] == recs_b[payload]["sample_id"]
        for payload, rec in recs_a.items():
            assert rec["sample_id"] == _expected_id(
                "second_src", rec["label"], payload
            )
            assert "second_src" in rec["sample_id"]

    def test_ids_include_source_name(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        rows = [
            {"payload": "same payload", "label": "benign"},
            {"payload": "<script>alert(1)</script>", "label": "xss"},
        ]
        src = tmp_path / "in.csv"
        dst_one = tmp_path / "one.jsonl"
        dst_two = tmp_path / "two.jsonl"
        _write_csv(src, rows)
        prepare_labeled_payloads(str(src), str(dst_one), "source_one", seed=42)
        prepare_labeled_payloads(str(src), str(dst_two), "source_two", seed=42)
        ids_one = {r["sample_id"] for r in _read_out(dst_one)}
        ids_two = {r["sample_id"] for r in _read_out(dst_two)}
        assert ids_one.isdisjoint(ids_two)
        for sid in ids_one:
            assert "source_one" in sid
        for sid in ids_two:
            assert "source_two" in sid

    def test_rerun_is_byte_identical(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        first = tmp_path / "first.jsonl"
        second = tmp_path / "second.jsonl"
        _write_csv(src, _mixed_rows())
        prepare_labeled_payloads(str(src), str(first), "second_src", seed=42)
        prepare_labeled_payloads(str(src), str(second), "second_src", seed=42)
        assert first.read_bytes() == second.read_bytes()


class TestRejections:
    def test_blank_payload_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(
            src,
            [
                {"payload": "   ", "label": "benign"},
                {"payload": "<script>alert(1)</script>", "label": "xss"},
            ],
        )
        with pytest.raises(ValueError, match="payload"):
            prepare_labeled_payloads(str(src), str(dst), "second_src")

    def test_missing_payload_column_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        with open(src, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerow({"text": "hi", "label": "benign"})
        with pytest.raises(ValueError, match="payload"):
            prepare_labeled_payloads(str(src), str(dst), "second_src")

    def test_missing_label_column_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        with open(src, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["payload", "tag"])
            writer.writeheader()
            writer.writerow({"payload": "hi", "tag": "benign"})
        with pytest.raises(ValueError, match="label"):
            prepare_labeled_payloads(str(src), str(dst), "second_src")

    def test_unsupported_label_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(
            src,
            [
                {"payload": "hello", "label": "benign"},
                {"payload": "<script>alert(1)</script>", "label": "maybe"},
            ],
        )
        with pytest.raises(ValueError, match="label"):
            prepare_labeled_payloads(str(src), str(dst), "second_src")

    def test_numeric_label_out_of_range_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.jsonl"
        dst = tmp_path / "out.jsonl"
        _write_jsonl(
            src,
            [
                {"payload": "hello", "label": 0},
                {"payload": "evil", "label": 2},
            ],
        )
        with pytest.raises(ValueError, match="label"):
            prepare_labeled_payloads(str(src), str(dst), "second_src")

    def test_single_class_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(
            src,
            [
                {"payload": "a", "label": "benign"},
                {"payload": "b", "label": "0"},
            ],
        )
        with pytest.raises(ValueError, match="both classes"):
            prepare_labeled_payloads(str(src), str(dst), "second_src")

    def test_reserved_source_name_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(src, _mixed_rows())
        with pytest.raises(ValueError, match="http_params_dataset"):
            prepare_labeled_payloads(
                str(src), str(dst), "http_params_dataset"
            )

    def test_blank_source_name_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        _write_csv(src, _mixed_rows())
        with pytest.raises(ValueError, match="source_name"):
            prepare_labeled_payloads(str(src), str(dst), "   ")

    def test_unsupported_suffix_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.txt"
        src.write_text("payload,label\nhello,benign\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            prepare_labeled_payloads(str(src), str(tmp_path / "o.jsonl"), "second_src")

    def test_missing_input_raises(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        with pytest.raises(FileNotFoundError):
            prepare_labeled_payloads(
                str(tmp_path / "nope.csv"),
                str(tmp_path / "o.jsonl"),
                "second_src",
            )


class TestSplitBoundaries:
    def test_group_atomic_single_source(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        dst = tmp_path / "out.jsonl"
        rows = [{"payload": f"benign-{i}", "label": "benign"} for i in range(15)]
        rows += [
            {"payload": f"<script>alert({i})</script>", "label": "xss"}
            for i in range(15)
        ]
        _write_csv(src, rows)
        prepare_labeled_payloads(str(src), str(dst), "second_src", seed=42)
        records = _read_out(dst)
        assert {r["split"] for r in records} <= {"train", "validation", "clean-test"}
        # Group-atomic split_records keeps one source together.
        assert len({r["split"] for r in records}) == 1

    def test_split_deterministic_for_seed(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.csv"
        first = tmp_path / "first.jsonl"
        second = tmp_path / "second.jsonl"
        _write_csv(src, _mixed_rows())
        prepare_labeled_payloads(str(src), str(first), "second_src", seed=7)
        prepare_labeled_payloads(str(src), str(second), "second_src", seed=7)
        assert first.read_bytes() == second.read_bytes()


def _write_json_array(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False)


class TestJsonArrayInput:
    """Standard ``.json`` array-of-objects support (TDD regression).

    A ``.json`` file must accept either a top-level JSON array of
    ``{"payload", "label"}`` objects or the existing JSONL-style
    one-object-per-line layout. ``.jsonl`` behavior is unchanged.
    """

    def test_json_array_basic_flow_and_provenance(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.json"
        dst = tmp_path / "out.jsonl"
        _write_json_array(src, _mixed_rows())
        summary = prepare_labeled_payloads(str(src), str(dst), "second_src", seed=42)
        assert summary["total_rows"] == 4
        assert summary["kept_benign"] == 2
        assert summary["kept_xss"] == 2
        assert summary["written"] == 4
        records = _read_out(dst)
        by_payload = {r["payload"]: r for r in records}
        assert by_payload["hello world"]["label"] == 0
        assert by_payload["hello world"]["attack_category"] == "benign"
        assert by_payload["<script>alert(1)</script>"]["label"] == 1
        assert by_payload["<script>alert(1)</script>"]["attack_category"] == "xss"
        for rec in records:
            assert rec["source"] == "second_src"
            assert rec["context_target"] == "unknown"
            assert rec["sample_id"] == _expected_id(
                "second_src", rec["label"], rec["payload"]
            )
        assert by_payload["hello world"]["raw_row_number"] == 1
        assert by_payload["<svg onload=alert(1)>"]["raw_row_number"] == 4
        assert by_payload["hello world"]["raw_label"] == "benign"

    def test_json_array_pretty_printed_multiline(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.json"
        dst = tmp_path / "out.jsonl"
        src.write_text(
            json.dumps(_mixed_rows(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = prepare_labeled_payloads(str(src), str(dst), "second_src")
        assert summary["written"] == 4
        assert {r["label"] for r in _read_out(dst)} == {0, 1}

    def test_json_array_int_labels_accepted(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.json"
        dst = tmp_path / "out.jsonl"
        _write_json_array(
            src,
            [
                {"payload": "a", "label": 0},
                {"payload": "b", "label": 1},
            ],
        )
        prepare_labeled_payloads(str(src), str(dst), "second_src")
        records = _read_out(dst)
        assert {r["payload"]: r["label"] for r in records} == {"a": 0, "b": 1}

    def test_dotjson_still_accepts_jsonl_lines(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.json"
        dst = tmp_path / "out.jsonl"
        _write_jsonl(src, _mixed_rows())
        summary = prepare_labeled_payloads(str(src), str(dst), "second_src")
        assert summary["written"] == 4
        assert {r["label"] for r in _read_out(dst)} == {0, 1}

    def test_json_array_missing_field_error_is_record_specific(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.json"
        dst = tmp_path / "out.jsonl"
        _write_json_array(
            src,
            [
                {"payload": "hello", "label": "benign"},
                {"payload": "<script>alert(1)</script>"},
            ],
        )
        with pytest.raises(ValueError, match="record 2.*label"):
            prepare_labeled_payloads(str(src), str(dst), "second_src")

    def test_json_array_unsupported_label_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.json"
        dst = tmp_path / "out.jsonl"
        _write_json_array(
            src,
            [
                {"payload": "hello", "label": "benign"},
                {"payload": "evil", "label": "maybe"},
            ],
        )
        with pytest.raises(ValueError, match="label"):
            prepare_labeled_payloads(str(src), str(dst), "second_src")

    def test_json_array_non_object_element_rejected(self, tmp_path):
        from xssharden.dataset.prepare_labeled_payloads import (
            prepare_labeled_payloads,
        )

        src = tmp_path / "in.json"
        dst = tmp_path / "out.jsonl"
        _write_json_array(
            src,
            [
                {"payload": "hello", "label": "benign"},
                "not-an-object",
            ],
        )
        with pytest.raises(TypeError, match="record 2"):
            prepare_labeled_payloads(str(src), str(dst), "second_src")
