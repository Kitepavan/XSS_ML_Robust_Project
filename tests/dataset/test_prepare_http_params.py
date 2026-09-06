"""Focused tests for http_params preparation (TDD — written before implementation)."""

from __future__ import annotations

import csv
import json


def _write_raw_csv(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["payload", "length", "attack_type", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _raw(payload: str, attack_type: str, label: str) -> dict:
    return {
        "payload": payload,
        "length": str(len(payload)),
        "attack_type": attack_type,
        "label": label,
    }


class TestPrepareKeepsOnlyNormAndXss:
    def test_keep_norm_and_xss_exclude_other_anomaly_types(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        _write_raw_csv(src, [
            _raw("hello", "norm", "norm"),
            _raw("<script>alert(1)</script>", "xss", "anom"),
            _raw("' OR '1'='1", "sqli", "anom"),
            _raw("; cat /etc/passwd", "cmdi", "anom"),
            _raw("../../etc/passwd", "path-traversal", "anom"),
        ])
        summary = prepare_http_params(str(src), str(dst))
        assert summary["kept_norm"] == 1
        assert summary["kept_xss"] == 1
        assert summary["excluded_total"] == 3
        assert summary["excluded_by_type"] == {"sqli": 1, "cmdi": 1, "path-traversal": 1}
        records = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 2
        by_payload = {r["payload"]: r for r in records}
        assert by_payload["hello"]["label"] == 0
        assert by_payload["hello"]["attack_category"] == "benign"
        assert by_payload["hello"]["context_target"] == "unknown"
        assert by_payload["<script>alert(1)</script>"]["label"] == 1
        # The raw source provides no injection context, so raw xss must
        # stay a generic xss category and carry context_target=unknown.
        assert by_payload["<script>alert(1)</script>"]["attack_category"] == "xss"
        assert by_payload["<script>alert(1)</script>"]["context_target"] == "unknown"

    def test_never_executes_payloads_only_reads_text(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        evil = "__import__('os').system('echo PWNED')"
        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        _write_raw_csv(src, [_raw(evil, "xss", "anom")])
        summary = prepare_http_params(str(src), str(dst))
        assert summary["written"] == 1
        records = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
        assert records[0]["payload"] == evil


class TestPrepareLabelConsistency:
    def test_norm_with_anom_label_raises(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        row = _raw("hello", "norm", "anom")
        _write_raw_csv(src, [row])
        try:
            prepare_http_params(str(src), str(dst))
        except ValueError as exc:
            assert "norm" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for norm/anom mismatch")

    def test_xss_with_norm_label_raises(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        row = _raw("<script>alert(1)</script>", "xss", "norm")
        _write_raw_csv(src, [row])
        try:
            prepare_http_params(str(src), str(dst))
        except ValueError as exc:
            assert "xss" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for xss/norm mismatch")

    def test_xss_with_unexpected_label_raises(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        row = _raw("<script>alert(1)</script>", "xss", "weird")
        _write_raw_csv(src, [row])
        try:
            prepare_http_params(str(src), str(dst))
        except ValueError as exc:
            assert "xss" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError for xss/unexpected mismatch")


class TestPrepareContentIds:
    def test_sample_ids_content_based_and_reorder_stable(self, tmp_path):
        import hashlib

        from xssharden.dataset.prepare_http_params import prepare_http_params

        rows = [
            _raw("alpha", "norm", "norm"),
            _raw("<svg onload=alert(1)>", "xss", "anom"),
            _raw("beta", "norm", "norm"),
        ]
        src_a = tmp_path / "a.csv"
        src_b = tmp_path / "b.csv"
        dst_a = tmp_path / "a.jsonl"
        dst_b = tmp_path / "b.jsonl"
        _write_raw_csv(src_a, rows)
        _write_raw_csv(src_b, list(reversed(rows)))
        prepare_http_params(str(src_a), str(dst_a), seed=42)
        prepare_http_params(str(src_b), str(dst_b), seed=42)
        recs_a = {
            r["payload"]: r
            for r in (
                json.loads(line)
                for line in dst_a.read_text(encoding="utf-8").splitlines()
            )
        }
        recs_b = {
            r["payload"]: r
            for r in (
                json.loads(line)
                for line in dst_b.read_text(encoding="utf-8").splitlines()
            )
        }
        assert set(recs_a) == set(recs_b)
        # Same content -> same ID regardless of raw-row order.
        for payload in recs_a:
            assert recs_a[payload]["sample_id"] == recs_b[payload]["sample_id"]
        # IDs are SHA-256 content hashes, not filtered-row positions.
        for payload, rec in recs_a.items():
            assert rec["sample_id"].startswith("httpp-")
            assert rec["sample_id"] != "httpp-000001"
            digest = hashlib.sha256(
                f"http_params_dataset|{rec['raw_attack_type']}|{payload}".encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            assert rec["sample_id"] == f"httpp-{digest}"


class TestPrepareIdsAndMetadata:
    def test_sample_ids_deterministic_and_stable(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        first = tmp_path / "first.jsonl"
        second = tmp_path / "second.jsonl"
        _write_raw_csv(src, [
            _raw("alpha", "norm", "norm"),
            _raw("<svg onload=alert(1)>", "xss", "anom"),
        ])
        prepare_http_params(str(src), str(first), seed=42)
        prepare_http_params(str(src), str(second), seed=42)
        assert first.read_bytes() == second.read_bytes()
        records = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
        ids = [r["sample_id"] for r in records]
        assert len(set(ids)) == 2
        assert all(isinstance(i, str) and i for i in ids)

    def test_length_mismatch_preserved_as_extra_not_dropped(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        row = _raw("<script>alert(1)</script>", "xss", "anom")
        row["length"] = "5"  # deliberately wrong declared length
        _write_raw_csv(src, [row])
        summary = prepare_http_params(str(src), str(dst))
        assert summary["length_mismatches"] == 1
        records = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 1
        rec = records[0]
        assert rec["payload"] == "<script>alert(1)</script>"
        assert rec["extras_declared_length"] == 5 if "extras_declared_length" in rec else True
        # Generic check: mismatch metadata must be present under some key.
        blob = json.dumps(rec)
        assert "mismatch" in blob.lower() or "declared" in blob.lower()

    def test_source_and_raw_audit_fields_preserved(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        _write_raw_csv(src, [_raw("hello", "norm", "norm")])
        prepare_http_params(str(src), str(dst))
        rec = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
        assert rec["source"]  # non-empty project source
        blob = json.dumps(rec)
        assert "norm" in blob  # raw attack_type/label audit info retained


class TestPrepareDedupAndSplit:
    def test_normalized_duplicate_removed_via_existing_dedup(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        _write_raw_csv(src, [
            _raw("<SCRIPT>alert(1)</SCRIPT>", "xss", "anom"),
            _raw("<script>alert(1)</script>", "xss", "anom"),
        ])
        summary = prepare_http_params(str(src), str(dst))
        assert summary["duplicates_removed"] == 1
        assert summary["written"] == 1

    def test_output_has_valid_splits_and_schema(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        _write_raw_csv(src, [_raw(f"benign-{i}", "norm", "norm") for i in range(30)])
        prepare_http_params(str(src), str(dst), seed=42)
        records = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
        assert {r["split"] for r in records} <= {"train", "validation", "clean-test"}
        for rec in records:
            assert isinstance(rec["sample_id"], str) and rec["sample_id"]
            assert isinstance(rec["payload"], str) and rec["payload"].strip()
            assert rec["label"] in (0, 1)
            assert isinstance(rec["source"], str) and rec["source"]
            assert isinstance(rec["attack_category"], str) and rec["attack_category"]

    def test_summary_reports_kept_excluded_mismatch(self, tmp_path):
        from xssharden.dataset.prepare_http_params import prepare_http_params

        src = tmp_path / "raw.csv"
        dst = tmp_path / "out.jsonl"
        bad = _raw("x", "xss", "anom")
        bad["length"] = "999"
        _write_raw_csv(src, [
            _raw("a", "norm", "norm"),
            bad,
            _raw("' OR 1=1--", "sqli", "anom"),
        ])
        summary = prepare_http_params(str(src), str(dst))
        assert summary["kept_norm"] == 1
        assert summary["kept_xss"] == 1
        assert summary["excluded_total"] == 1
        assert summary["length_mismatches"] == 1
        assert summary["written"] == 2
