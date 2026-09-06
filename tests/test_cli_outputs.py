"""Regression tests for generation output serialization."""

from __future__ import annotations

import json

import pytest


def _variants() -> list[dict]:
    return [{"variant_id": "v-1", "payload": "<svg>", "label": 1}]


def test_variant_writer_emits_json_array_for_json(tmp_path):
    from xssharden.cli import _write_variant_records

    output = tmp_path / "variants.json"
    _write_variant_records(str(output), _variants())

    assert json.loads(output.read_text(encoding="utf-8")) == _variants()


def test_variant_writer_emits_one_object_per_line_for_jsonl(tmp_path):
    from xssharden.cli import _write_variant_records

    output = tmp_path / "variants.jsonl"
    _write_variant_records(str(output), _variants())

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == _variants()[0]


def test_variant_writer_rejects_unknown_suffix(tmp_path):
    from xssharden.cli import _write_variant_records

    with pytest.raises(ValueError, match="expected '.jsonl' or '.json'"):
        _write_variant_records(str(tmp_path / "variants.txt"), _variants())
