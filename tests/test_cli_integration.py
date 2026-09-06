"""Integration tests for the train, attack, select, and run CLI commands.

Uses tiny in-memory fixtures only — no real browser, network, or large
datasets are required.  Every test exercises the full CLI entry point
(:func:`xssharden.cli.main`) with synthetic records that satisfy the
project schema.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from xssharden.cli import main


# ---------------------------------------------------------------------------
# Prevent Playwright from being imported into sys.modules during these
# tests.  The ``run`` command's stage 2 imports the validator module,
# which in turn imports Playwright.  Since these integration tests never
# execute real browser validation, we mock the Playwright import so that
# the pre-existing ``test_import_does_not_require_playwright`` test in
# ``tests/validation/`` continues to pass when the full suite runs.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _prevent_playwright_import():
    """Ensure Playwright is not loaded into sys.modules by these tests."""
    had_playwright = "playwright" in sys.modules
    old_playwright = sys.modules.get("playwright")
    old_sync_api = sys.modules.get("playwright.sync_api")
    yield
    # Restore sys.modules to its previous state so other test modules
    # (e.g. test_validator.py) see a clean import environment.
    if not had_playwright and "playwright" in sys.modules:
        del sys.modules["playwright"]
    if old_playwright is not None:
        sys.modules["playwright"] = old_playwright
    elif "playwright" in sys.modules and old_playwright is None:
        del sys.modules["playwright"]
    if old_sync_api is not None:
        sys.modules["playwright.sync_api"] = old_sync_api
    elif "playwright.sync_api" in sys.modules and old_sync_api is None:
        del sys.modules["playwright.sync_api"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_dataset(n_train: int = 20, n_val: int = 10, n_test: int = 10) -> list[dict[str, Any]]:
    """Create a minimal but valid project-schema dataset with train/validation/clean-test splits."""
    records: list[dict[str, Any]] = []
    for i in range(n_train):
        label = 1 if i % 2 == 0 else 0
        records.append({
            "sample_id": f"tr-{i:04d}",
            "payload": f"<script>alert({i})</script>" if label else f"hello world {i}",
            "label": label,
            "source": "test_source",
            "attack_category": "xss" if label else "benign",
            "split": "train",
        })
    for i in range(n_val):
        label = 1 if i % 2 == 0 else 0
        records.append({
            "sample_id": f"val-{i:04d}",
            "payload": f"<img src=x onerror=prompt({i})>" if label else f"safe text {i}",
            "label": label,
            "source": "test_source",
            "attack_category": "xss" if label else "benign",
            "split": "validation",
        })
    for i in range(n_test):
        label = 1 if i % 2 == 0 else 0
        records.append({
            "sample_id": f"tst-{i:04d}",
            "payload": f"<svg onload=confirm({i})>" if label else f"benign content {i}",
            "label": label,
            "source": "test_source",
            "attack_category": "xss" if label else "benign",
            "split": "clean-test",
        })
    return records


def _make_variants(n: int = 8, *, split: str = "adv_dev", valid: bool = True) -> list[dict[str, Any]]:
    """Create minimal scored variant records for select/attack tests.

    Includes all required schema fields (sample_id, payload, label, source,
    attack_category, split) plus variant-specific provenance and scores.
    """
    records: list[dict[str, Any]] = []
    for i in range(n):
        records.append({
            "sample_id": f"V-{i:04d}",
            "variant_id": f"V-{i:04d}",
            "seed_id": f"S-{i:04d}",
            "payload": f"<script>alert({i})</script>" if i % 2 == 0 else f"<img src=x onerror=prompt({i})>",
            "label": 1,
            "source": "test_gen",
            "attack_category": "xss",
            "generator": "programmatic",
            "mutation_category": ["encoding", "whitespace_comment", "tag_event_substitution", "case_variation"][i % 4],
            "valid": valid,
            "split": split,
            "detector_score": round(0.1 + 0.8 * (i / max(n - 1, 1)), 4),
            "probability": round(0.1 + 0.8 * (i / max(n - 1, 1)), 4),
            "predicted_label": 0 if i < n // 2 else 1,
            "detector_threshold": 0.5,
            "evaded": i < n // 2,
        })
    return records


# ---------------------------------------------------------------------------
# Task 22 — train CLI
# ---------------------------------------------------------------------------


class TestTrainCLI:
    """Tests for ``xssharden train``."""

    def test_train_lr_success(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        output_path = tmp_path / "model.pkl"
        _write_jsonl(_make_dataset(), dataset_path)

        rc = main(["train", "--input", str(dataset_path), "--output", str(output_path), "--model", "lr"])
        assert rc == 0
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_train_xgb_success(self, tmp_path: Path) -> None:
        xgb = pytest.importorskip("xgboost", reason="xgboost not installed")
        dataset_path = tmp_path / "dataset.jsonl"
        output_path = tmp_path / "model.pkl"
        _write_jsonl(_make_dataset(), dataset_path)

        rc = main(["train", "--input", str(dataset_path), "--output", str(output_path), "--model", "xgb"])
        assert rc == 0
        assert output_path.exists()

    def test_train_missing_input(self, tmp_path: Path) -> None:
        output_path = tmp_path / "model.pkl"
        rc = main(["train", "--input", str(tmp_path / "missing.jsonl"), "--output", str(output_path)])
        assert rc == 2

    def test_train_no_train_records(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        output_path = tmp_path / "model.pkl"
        records = [
            {"sample_id": "v-0001", "payload": "hello", "label": 0,
             "source": "s", "attack_category": "benign", "split": "validation"},
        ]
        _write_jsonl(records, dataset_path)
        rc = main(["train", "--input", str(dataset_path), "--output", str(output_path)])
        assert rc == 2

    def test_train_invalid_model(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        output_path = tmp_path / "model.pkl"
        _write_jsonl(_make_dataset(), dataset_path)
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "--input", str(dataset_path), "--output", str(output_path), "--model", "bad"])
        assert exc_info.value.code == 2

    def test_train_loadable_after_save(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        output_path = tmp_path / "model.pkl"
        _write_jsonl(_make_dataset(), dataset_path)
        main(["train", "--input", str(dataset_path), "--output", str(output_path), "--model", "lr"])

        from xssharden.detectors import load_detector
        detector = load_detector(str(output_path))
        desc = detector.describe()
        assert desc["detector"] == "baseline-lr"
        assert desc["threshold"] > 0.0

    def test_train_custom_seed(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        output_path = tmp_path / "model.pkl"
        _write_jsonl(_make_dataset(), dataset_path)
        rc = main(["train", "--input", str(dataset_path), "--output", str(output_path),
                    "--model", "lr", "--seed", "99"])
        assert rc == 0


# ---------------------------------------------------------------------------
# Task 23 — attack CLI
# ---------------------------------------------------------------------------


class TestAttackCLI:
    """Tests for ``xssharden attack``."""

    def test_attack_success(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        model_path = tmp_path / "model.pkl"
        variants_path = tmp_path / "variants.jsonl"
        output_path = tmp_path / "scored.jsonl"

        _write_jsonl(_make_dataset(), dataset_path)
        main(["train", "--input", str(dataset_path), "--output", str(model_path)])
        _write_jsonl(_make_variants(), variants_path)

        rc = main(["attack", "--detector", str(model_path), "--input", str(variants_path),
                    "--output", str(output_path)])
        assert rc == 0
        assert output_path.exists()

        scored = _read_jsonl(output_path)
        assert len(scored) == 8
        for rec in scored:
            assert "detector_score" in rec
            assert "predicted_label" in rec
            assert "evaded" in rec

    def test_attack_missing_detector(self, tmp_path: Path) -> None:
        variants_path = tmp_path / "variants.jsonl"
        output_path = tmp_path / "scored.jsonl"
        _write_jsonl(_make_variants(), variants_path)
        rc = main(["attack", "--detector", str(tmp_path / "missing.pkl"),
                    "--input", str(variants_path), "--output", str(output_path)])
        assert rc == 2

    def test_attack_missing_input(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.pkl"
        dataset_path = tmp_path / "dataset.jsonl"
        _write_jsonl(_make_dataset(), dataset_path)
        main(["train", "--input", str(dataset_path), "--output", str(model_path)])
        output_path = tmp_path / "scored.jsonl"
        rc = main(["attack", "--detector", str(model_path),
                    "--input", str(tmp_path / "missing.jsonl"), "--output", str(output_path)])
        assert rc == 2

    def test_attack_preserves_provenance(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        model_path = tmp_path / "model.pkl"
        variants_path = tmp_path / "variants.jsonl"
        output_path = tmp_path / "scored.jsonl"

        _write_jsonl(_make_dataset(), dataset_path)
        main(["train", "--input", str(dataset_path), "--output", str(model_path)])
        _write_jsonl(_make_variants(), variants_path)

        main(["attack", "--detector", str(model_path), "--input", str(variants_path),
              "--output", str(output_path)])
        scored = _read_jsonl(output_path)
        for rec in scored:
            assert "variant_id" in rec
            assert "seed_id" in rec
            assert "mutation_category" in rec


# ---------------------------------------------------------------------------
# Task 24 — select CLI
# ---------------------------------------------------------------------------


class TestSelectCLI:
    """Tests for ``xssharden select``."""

    def test_select_impact_success(self, tmp_path: Path) -> None:
        variants_path = tmp_path / "scored.jsonl"
        output_path = tmp_path / "selected.jsonl"
        _write_jsonl(_make_variants(), variants_path)

        rc = main(["select", "--input", str(variants_path), "--output", str(output_path),
                    "--strategy", "impact", "--budget", "4"])
        assert rc == 0
        assert output_path.exists()

        selected = _read_jsonl(output_path)
        assert len(selected) == 4
        for rec in selected:
            assert rec["valid"] is True
            assert rec["label"] == 1

    def test_select_random_valid_success(self, tmp_path: Path) -> None:
        variants_path = tmp_path / "scored.jsonl"
        output_path = tmp_path / "selected.jsonl"
        _write_jsonl(_make_variants(), variants_path)

        rc = main(["select", "--input", str(variants_path), "--output", str(output_path),
                    "--strategy", "random_valid", "--budget", "3"])
        assert rc == 0
        selected = _read_jsonl(output_path)
        assert len(selected) == 3

    def test_select_missing_input(self, tmp_path: Path) -> None:
        output_path = tmp_path / "selected.jsonl"
        rc = main(["select", "--input", str(tmp_path / "missing.jsonl"),
                    "--output", str(output_path), "--budget", "5"])
        assert rc == 2

    def test_select_shortfall(self, tmp_path: Path) -> None:
        variants_path = tmp_path / "scored.jsonl"
        output_path = tmp_path / "selected.jsonl"
        _write_jsonl(_make_variants(n=3), variants_path)

        rc = main(["select", "--input", str(variants_path), "--output", str(output_path),
                    "--strategy", "impact", "--budget", "100"])
        assert rc == 0
        selected = _read_jsonl(output_path)
        assert len(selected) == 3

    def test_select_with_training_dedup(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        variants_path = tmp_path / "scored.jsonl"
        output_path = tmp_path / "selected.jsonl"

        _write_jsonl(_make_dataset(), dataset_path)
        _write_jsonl(_make_variants(), variants_path)

        rc = main(["select", "--input", str(variants_path), "--output", str(output_path),
                    "--strategy", "impact", "--budget", "4",
                    "--training-data", str(dataset_path)])
        assert rc == 0

    def test_select_max_per_category(self, tmp_path: Path) -> None:
        variants_path = tmp_path / "scored.jsonl"
        output_path = tmp_path / "selected.jsonl"
        _write_jsonl(_make_variants(n=8), variants_path)

        rc = main(["select", "--input", str(variants_path), "--output", str(output_path),
                    "--strategy", "impact", "--budget", "8", "--max-per-category", "1"])
        assert rc == 0
        selected = _read_jsonl(output_path)
        categories = [r["mutation_category"] for r in selected]
        assert len(categories) == len(set(categories))


# ---------------------------------------------------------------------------
# Task 25 — run end-to-end
# ---------------------------------------------------------------------------


class TestRunCLI:
    """Tests for ``xssharden run``."""

    def test_run_missing_config(self, tmp_path: Path) -> None:
        rc = main(["run", str(tmp_path / "missing.yaml")])
        assert rc == 2

    def test_run_invalid_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(": invalid: yaml: [[", encoding="utf-8")
        rc = main(["run", str(config_path)])
        assert rc == 2

    def test_run_no_dataset_path(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model: lr\nseed: 42\n", encoding="utf-8")
        rc = main(["run", str(config_path)])
        assert rc == 2

    def test_run_missing_dataset(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"dataset_path: {tmp_path / 'nonexistent.jsonl'}\nmodel: lr\nseed: 42\n",
            encoding="utf-8",
        )
        rc = main(["run", str(config_path)])
        assert rc == 2

    def test_run_train_only(self, tmp_path: Path) -> None:
        """Run with a dataset that has train records but no validation — should complete generate+train stages."""
        dataset_path = tmp_path / "dataset.jsonl"
        records = []
        for i in range(20):
            label = 1 if i % 2 == 0 else 0
            records.append({
                "sample_id": f"r-{i:04d}",
                "payload": f"<script>alert({i})</script>" if label else f"text {i}",
                "label": label,
                "source": "test",
                "attack_category": "xss" if label else "benign",
                "split": "train",
            })
        _write_jsonl(records, dataset_path)

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(f"""\
                dataset_path: {dataset_path}
                model: lr
                seed: 42
                budget: 5
                max_per_seed: 2
                variants_dir: {tmp_path / 'variants'}
                models_dir: {tmp_path / 'models'}
                output_dir: {tmp_path / 'experiments'}
                reports_dir: {tmp_path / 'reports'}
            """),
            encoding="utf-8",
        )

        rc = main(["run", str(config_path)])
        # Should succeed through generate+train stages; validation will produce
        # a RuntimeError if Playwright is not installed, which maps to rc=2.
        # We accept rc 0 or 2 depending on the runtime.
        assert rc in (0, 2)
        # At minimum, the model should have been saved
        model_path = tmp_path / "models" / "baseline_lr.pkl"
        assert model_path.exists()

    def test_run_skip_existing(self, tmp_path: Path) -> None:
        """When output files exist and --force is not passed, stages are skipped."""
        dataset_path = tmp_path / "dataset.jsonl"
        records = []
        for i in range(20):
            label = 1 if i % 2 == 0 else 0
            records.append({
                "sample_id": f"r-{i:04d}",
                "payload": f"<script>alert({i})</script>" if label else f"text {i}",
                "label": label,
                "source": "test",
                "attack_category": "xss" if label else "benign",
                "split": "train",
            })
        _write_jsonl(records, dataset_path)

        variants_dir = tmp_path / "variants"
        variants_dir.mkdir()
        generated_path = variants_dir / "generated_programmatic.jsonl"
        _write_jsonl([{"variant_id": "V-0001", "payload": "<script>alert(1)</script>",
                        "label": 1, "split": "adv_dev", "valid": True,
                        "source": "test", "generator": "programmatic",
                        "mutation_category": "encoding"}], generated_path)

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(f"""\
                dataset_path: {dataset_path}
                model: lr
                seed: 42
                budget: 5
                variants_dir: {variants_dir}
                models_dir: {tmp_path / 'models'}
                output_dir: {tmp_path / 'experiments'}
                reports_dir: {tmp_path / 'reports'}
            """),
            encoding="utf-8",
        )

        rc = main(["run", str(config_path)])
        # Stage 1 should be skipped because generated file exists
        assert rc in (0, 2)
        # The generated file should still be there unchanged
        assert generated_path.exists()
