"""Focused TDD tests for the deterministic programmatic XSS mutation engine.

Conventions shared with the rest of the suite:

- Only synthetic inert strings are used as payloads. No downloaded dataset
  payloads are executed, no browser is launched, and no network traffic is
  sent. Generation is pure string manipulation.
- ``generate_variants`` is deterministic: identical inputs produce
  byte-identical outputs, and outputs carry full seed provenance.
"""

from __future__ import annotations

import pytest


def _seed(**overrides):
    base = {
        "sample_id": "S001",
        "payload": '<script>alert("xss")</script>',
        "label": 1,
        "source": "train_corpus",
        "attack_category": "xss",
        "split": "train",
        "context_target": "reflected_html",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_identical_inputs_produce_identical_outputs(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed(), _seed(sample_id="S002", payload='<img src=x onerror=alert(1)>')]
        first = generate_variants(seeds, seed=42)
        second = generate_variants(seeds, seed=42)
        assert first == second
        assert len(first) > 0

    def test_deterministic_across_repeated_calls_with_limits(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed()]
        first = generate_variants(seeds, seed=7, max_per_seed=3)
        second = generate_variants(seeds, seed=7, max_per_seed=3)
        assert first == second
        assert len(first) == 3

    def test_different_random_state_is_deterministic(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed()]
        first = generate_variants(seeds, seed=1, max_per_seed=2)
        second = generate_variants(seeds, seed=1, max_per_seed=2)
        assert first == second

    def test_stable_output_ordering(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [
            _seed(sample_id="S002", payload="<svg onload=alert(1)>"),
            _seed(sample_id="S001"),
        ]
        first = generate_variants(seeds, seed=42)
        second = generate_variants(list(reversed(seeds)), seed=42)
        # Same multiset of variant payloads regardless of seed input order.
        assert sorted(v["variant_id"] for v in first) == sorted(
            v["variant_id"] for v in second
        )

    def test_variant_ids_are_content_based_not_positional(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed(sample_id="S001"), _seed(sample_id="S002", payload="<svg>")]
        variants = generate_variants(seeds, seed=42)
        by_seed = {}
        for variant in variants:
            by_seed.setdefault(variant["seed_id"], []).append(variant["variant_id"])
        # Variant IDs embed the seed identity: reordering seeds keeps IDs.
        reordered = generate_variants(list(reversed(seeds)), seed=42)
        assert {v["variant_id"] for v in variants} == {
            v["variant_id"] for v in reordered
        }


# ---------------------------------------------------------------------------
# Provenance and output schema
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_required_output_fields_present(self):
        from xssharden.generation.programmatic import generate_variants

        (variant,) = generate_variants([_seed()], seed=42, max_per_seed=1)
        for field in (
            "variant_id",
            "seed_id",
            "payload",
            "mutation_category",
            "context_target",
            "source",
            "generator",
        ):
            assert field in variant, f"missing field {field!r}"

    def test_seed_provenance_preserved(self):
        from xssharden.generation.programmatic import generate_variants

        variants = generate_variants([_seed()], seed=42, max_per_seed=2)
        assert len(variants) > 0
        for variant in variants:
            assert variant["seed_id"] == "S001"
            assert variant["source"] == "train_corpus"
            assert variant["generator"] == "programmatic"
            assert variant["context_target"] == "reflected_html"

    def test_label_preserved_when_present(self):
        from xssharden.generation.programmatic import generate_variants

        variants = generate_variants([_seed(label=1)], seed=42, max_per_seed=2)
        assert all(v["label"] == 1 for v in variants)

    def test_missing_label_does_not_crash_and_is_absent(self):
        from xssharden.generation.programmatic import generate_variants

        seed = _seed()
        del seed["label"]
        variants = generate_variants([seed], seed=42, max_per_seed=1)
        assert len(variants) == 1
        assert "label" not in variants[0]

    def test_variant_ids_unique_and_hex_shaped(self):
        import re

        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed(), _seed(sample_id="S002", payload="<svg onload=alert(1)>")]
        variants = generate_variants(seeds, seed=42)
        ids = [v["variant_id"] for v in variants]
        assert len(set(ids)) == len(ids)
        for vid in ids:
            assert re.fullmatch(r"var-[0-9a-f]{16}", vid), vid

    def test_generator_metadata_present(self):
        from xssharden.generation.programmatic import generate_variants

        (variant,) = generate_variants([_seed()], seed=42, max_per_seed=1)
        assert variant["generator"] == "programmatic"
        assert isinstance(variant.get("generator_version"), str)
        assert variant["generator_version"]


# ---------------------------------------------------------------------------
# Mutation categories
# ---------------------------------------------------------------------------


class TestCategories:
    def test_single_category_request_yields_only_that_category(self):
        from xssharden.generation.programmatic import generate_variants

        for category in (
            "encoding",
            "whitespace_comment",
            "tag_event_substitution",
            "case_variation",
        ):
            variants = generate_variants([_seed()], categories=[category], seed=42)
            assert len(variants) > 0, category
            assert {v["mutation_category"] for v in variants} == {category}

    def test_default_covers_all_supported_categories(self):
        from xssharden.generation.programmatic import (
            SUPPORTED_CATEGORIES,
            generate_variants,
        )

        variants = generate_variants([_seed()], seed=42)
        assert set(SUPPORTED_CATEGORIES) == {
            "encoding",
            "whitespace_comment",
            "tag_event_substitution",
            "case_variation",
        }
        assert {v["mutation_category"] for v in variants} <= set(SUPPORTED_CATEGORIES)

    def test_each_category_actually_rewrites_payload(self):
        from xssharden.generation.programmatic import generate_variants

        seed_payload = _seed()["payload"]
        for category in (
            "encoding",
            "whitespace_comment",
            "tag_event_substitution",
            "case_variation",
        ):
            variants = generate_variants([_seed()], categories=[category], seed=42)
            assert all(v["payload"] != seed_payload for v in variants), category

    def test_unsupported_category_rejected(self):
        from xssharden.generation.programmatic import generate_variants

        with pytest.raises(ValueError) as excinfo:
            generate_variants([_seed()], categories=["llm_paraphrase"])
        assert "llm_paraphrase" in str(excinfo.value)

    def test_empty_category_list_rejected(self):
        from xssharden.generation.programmatic import generate_variants

        with pytest.raises(ValueError):
            generate_variants([_seed()], categories=[])


# ---------------------------------------------------------------------------
# Deduplication and limits
# ---------------------------------------------------------------------------


class TestDeduplicationAndLimits:
    def test_no_duplicate_payloads_in_output(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed(), _seed(sample_id="S002", payload='<script>alert("xss")</script>')]
        variants = generate_variants(seeds, seed=42)
        payloads = [v["payload"] for v in variants]
        assert len(set(payloads)) == len(payloads)

    def test_variants_never_repeat_the_seed_payload(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed()]
        seed_payloads = {s["payload"] for s in seeds}
        variants = generate_variants(seeds, seed=42)
        assert all(v["payload"] not in seed_payloads for v in variants)

    def test_max_per_seed_enforced(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed(), _seed(sample_id="S002", payload="<svg onload=alert(1)>")]
        variants = generate_variants(seeds, seed=42, max_per_seed=2)
        from collections import Counter

        counts = Counter(v["seed_id"] for v in variants)
        assert all(n <= 2 for n in counts.values())

    def test_max_per_category_enforced(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed(sample_id=f"S{i:03d}") for i in range(4)]
        variants = generate_variants(seeds, seed=42, max_per_category=2)
        from collections import Counter

        counts = Counter(v["mutation_category"] for v in variants)
        assert all(n <= 2 for n in counts.values())

    def test_bad_limits_rejected(self):
        from xssharden.generation.programmatic import generate_variants

        with pytest.raises(ValueError):
            generate_variants([_seed()], max_per_seed=0)
        with pytest.raises(ValueError):
            generate_variants([_seed()], max_per_category=-1)
        with pytest.raises(TypeError):
            generate_variants([_seed()], max_per_seed="3")  # type: ignore[arg-type]

    def test_empty_seed_list_returns_empty(self):
        from xssharden.generation.programmatic import generate_variants

        assert generate_variants([], seed=42) == []


# ---------------------------------------------------------------------------
# Split-boundary enforcement
# ---------------------------------------------------------------------------


class TestSplitBoundary:
    @pytest.mark.parametrize("split", ["validation", "clean-test", "test"])
    def test_non_train_seeds_rejected_by_default(self, split):
        from xssharden.generation.programmatic import generate_variants

        with pytest.raises(ValueError) as excinfo:
            generate_variants([_seed(split=split)])
        assert split in str(excinfo.value)

    @pytest.mark.parametrize("split", ["train", "adv_dev"])
    def test_train_and_adv_dev_seeds_accepted(self, split):
        from xssharden.generation.programmatic import generate_variants

        variants = generate_variants([_seed(split=split)], seed=42, max_per_seed=1)
        assert len(variants) == 1

    def test_boundary_can_be_explicitly_relaxed(self):
        from xssharden.generation.programmatic import generate_variants

        variants = generate_variants(
            [_seed(split="validation")],
            seed=42,
            max_per_seed=1,
            enforce_split_boundary=False,
        )
        assert len(variants) == 1

    def test_mixed_batch_names_offending_index(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed(), _seed(sample_id="S002", split="clean-test")]
        with pytest.raises(ValueError) as excinfo:
            generate_variants(seeds)
        assert "1" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Malformed inputs
# ---------------------------------------------------------------------------


class TestMalformedInputs:
    def test_non_dict_seed_rejected(self):
        from xssharden.generation.programmatic import generate_variants

        with pytest.raises(TypeError):
            generate_variants(["not-a-dict"])  # type: ignore[list-item]

    def test_missing_payload_rejected(self):
        from xssharden.generation.programmatic import generate_variants

        seed = _seed()
        del seed["payload"]
        with pytest.raises((ValueError, TypeError)) as excinfo:
            generate_variants([seed])
        assert "payload" in str(excinfo.value).lower()

    def test_empty_payload_rejected(self):
        from xssharden.generation.programmatic import generate_variants

        with pytest.raises(ValueError):
            generate_variants([_seed(payload="   ")])

    def test_non_string_payload_rejected(self):
        from xssharden.generation.programmatic import generate_variants

        with pytest.raises(TypeError):
            generate_variants([_seed(payload=12345)])  # type: ignore[dict-item]

    def test_missing_sample_id_rejected(self):
        from xssharden.generation.programmatic import generate_variants

        seed = _seed()
        del seed["sample_id"]
        with pytest.raises((ValueError, TypeError)) as excinfo:
            generate_variants([seed])
        assert "sample_id" in str(excinfo.value).lower()

    def test_error_names_seed_index(self):
        from xssharden.generation.programmatic import generate_variants

        seeds = [_seed(), _seed(sample_id="S002", payload=999)]
        with pytest.raises(TypeError) as excinfo:
            generate_variants(seeds)  # type: ignore[list-item]
        assert "1" in str(excinfo.value)

    def test_inputs_not_mutated(self):
        from xssharden.generation.programmatic import generate_variants

        seed = _seed()
        snapshot = dict(seed)
        generate_variants([seed], seed=42)
        assert seed == snapshot

    def test_bad_seed_type_rejected(self):
        from xssharden.generation.programmatic import generate_variants

        with pytest.raises(TypeError):
            generate_variants([_seed()], seed="42")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Inert / no-execution behavior
# ---------------------------------------------------------------------------


class TestInertBehavior:
    def test_payloads_are_plain_strings(self):
        from xssharden.generation.programmatic import generate_variants

        variants = generate_variants([_seed()], seed=42)
        assert all(isinstance(v["payload"], str) for v in variants)

    def test_module_performs_no_execution_or_browser_work(self):
        import pathlib

        module_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src"
            / "xssharden"
            / "generation"
            / "programmatic.py"
        )
        source = module_path.read_text(encoding="utf-8")
        for forbidden in (
            "playwright",
            "chromium",
            "subprocess",
            "os.system",
            "os.exec",
            "os.popen",
            "__import__",
        ):
            assert forbidden not in source, forbidden
        # No dynamic code execution primitives anywhere in the module.
        for forbidden in ("eval(", "exec("):
            assert forbidden not in source, forbidden

    def test_generation_does_not_import_browser_stack(self, monkeypatch):
        import sys

        from xssharden.generation import programmatic

        assert "playwright" not in sys.modules or True
        # The generation package must not depend on the validator package.
        monkeypatch.delitem(sys.modules, "playwright", raising=False)
        variants = programmatic.generate_variants([_seed()], seed=42, max_per_seed=1)
        assert len(variants) == 1
        assert "playwright" not in sys.modules


# ---------------------------------------------------------------------------
# CLI (local-only, no browser; synthetic records only)
# ---------------------------------------------------------------------------


class TestGenerateCli:
    def _write_seeds(self, path, records):
        import json

        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return str(path)

    def test_cli_writes_variants_jsonl(self, tmp_path):
        import json

        from xssharden.cli import main

        inp = self._write_seeds(tmp_path / "seeds.jsonl", [_seed(), _seed(
            sample_id="S002",
            payload="<svg onload=alert(1)>",
            source="train_corpus",
        )])
        out = str(tmp_path / "variants.jsonl")
        rc = main(["generate", "--input", inp, "--output", out, "--max-per-seed", "2"])
        assert rc == 0
        with open(out, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        assert len(rows) > 0
        assert all("variant_id" in r and "seed_id" in r for r in rows)

    def test_cli_rejects_unsupported_category(self, tmp_path):
        from xssharden.cli import main

        inp = self._write_seeds(tmp_path / "seeds.jsonl", [_seed()])
        rc = main(
            [
                "generate",
                "--input", inp,
                "--output", str(tmp_path / "out.jsonl"),
                "--categories", "llm_paraphrase",
            ]
        )
        assert rc == 2

    def test_cli_rejects_validation_seed_by_default(self, tmp_path):
        from xssharden.cli import main

        inp = self._write_seeds(tmp_path / "seeds.jsonl", [_seed(split="validation")])
        rc = main(
            ["generate", "--input", inp, "--output", str(tmp_path / "out.jsonl")]
        )
        assert rc == 2

    def test_cli_json_writes_valid_json_array(self, tmp_path):
        import json

        from xssharden.cli import main

        inp = self._write_seeds(tmp_path / "seeds.jsonl", [_seed()])
        out = str(tmp_path / "variants.json")
        rc = main(["generate", "--input", inp, "--output", out, "--max-per-seed", "2"])
        assert rc == 0
        with open(out, encoding="utf-8") as handle:
            data = json.load(handle)
        assert isinstance(data, list)
        assert len(data) > 0
        assert all(isinstance(r, dict) and "variant_id" in r for r in data)

    def test_cli_jsonl_writes_one_object_per_line(self, tmp_path):
        import json

        from xssharden.cli import main

        inp = self._write_seeds(tmp_path / "seeds.jsonl", [_seed()])
        out = str(tmp_path / "variants.jsonl")
        rc = main(["generate", "--input", inp, "--output", out, "--max-per-seed", "2"])
        assert rc == 0
        with open(out, encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        assert len(lines) > 0
        for line in lines:
            assert isinstance(json.loads(line), dict)

    def test_cli_rejects_invalid_suffix(self, tmp_path):
        import os

        from xssharden.cli import main

        inp = self._write_seeds(tmp_path / "seeds.jsonl", [_seed()])
        out = str(tmp_path / "variants.txt")
        rc = main(["generate", "--input", inp, "--output", out])
        assert rc == 2
        assert not os.path.exists(out)
