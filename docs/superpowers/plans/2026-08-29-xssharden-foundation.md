# XSSHarden Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first reproducible foundation of the XSSHarden research pipeline.

**Architecture:** A small Python package will separate dataset processing, schema validation, leakage-safe splitting, and shared character-level feature extraction. A lightweight CLI will expose dataset-building and feature smoke checks without pretending the later training/validation stages exist yet.

**Tech Stack:** Python 3.10+, pandas, NumPy, scikit-learn, pytest, JSONL/CSV, standard-library hashing and configuration utilities.

**Spec:** `XSS_ML_Robustness_handoff_doc.md`, `xss-robustness-project-proposal.md`

## Global Constraints

- Keep XSS as the only scope and use reflected HTML as the primary context.
- Preserve `sample_id`, `payload`, `label`, `source`, `attack_category`, and `split` fields.
- Use exact and normalized deduplication before splitting.
- Never use naive random splitting when source/cluster-aware grouping is available.
- Keep deterministic seeds and explicit train/validation/clean-test boundaries.
- Use one shared `featurize()` implementation for every detector.
- Keep generated or invalid variants auditable rather than silently deleting records.

---

### Task 1: Repository and package scaffolding

**Files:**
- Create: `pyproject.toml`, `README.md`, `.gitignore`
- Create: `src/xssharden/__init__.py`, `src/xssharden/cli.py`
- Create: `src/xssharden/dataset/__init__.py`, `src/xssharden/features/__init__.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Produces an installable package and a `python -m xssharden` entry point.

- [ ] Write a failing smoke test asserting the package exposes `__version__` and the CLI help exits successfully.
- [ ] Run `pytest tests/test_smoke.py -q` and verify it fails because the package is absent.
- [ ] Add minimal package metadata, CLI help, and a test configuration.
- [ ] Run the smoke test and verify it passes.

### Task 2: Dataset schema, cleaning, and deduplication

**Files:**
- Create: `src/xssharden/dataset/schema.py`, `src/xssharden/dataset/clean.py`
- Create: `tests/dataset/test_clean.py`

**Interfaces:**
- `validate_records(records) -> list[dict]`
- `normalize_payload(payload) -> str`
- `deduplicate_records(records) -> list[dict]`

- [ ] Write failing tests for required fields, stable normalization, exact duplicate removal, and normalized duplicate removal.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement typed validation and deterministic cleaning with clear errors.
- [ ] Run the focused tests and verify they pass.

### Task 3: Leakage-safe splitting and reproducibility

**Files:**
- Create: `src/xssharden/dataset/split.py`, `src/xssharden/config.py`
- Create: `tests/dataset/test_split.py`, `tests/test_config.py`

**Interfaces:**
- `split_records(records, seed, ratios=(0.70, 0.15, 0.15)) -> list[dict]`
- `make_experiment_id(config, seed) -> str`

- [ ] Write failing tests for deterministic assignments, disjoint split membership, ratio validation, and source/cluster grouping where metadata exists.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement deterministic grouped splitting with explicit ratio checks and stable experiment IDs.
- [ ] Run the focused tests and verify they pass.

### Task 4: Shared character-level feature pipeline

**Files:**
- Create: `src/xssharden/features/text.py`
- Create: `tests/features/test_text.py`

**Interfaces:**
- `featurize(payloads, vectorizer=None) -> tuple[object, object]`

- [ ] Write failing tests for deterministic character n-gram features, fit/transform reuse, and empty-input handling.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement a shared scikit-learn character TF-IDF pipeline with a reusable fitted vectorizer.
- [ ] Run the focused tests and verify they pass.

### Task 5: Dataset CLI and documentation

**Files:**
- Modify: `src/xssharden/cli.py`
- Create: `src/xssharden/dataset/io.py`, `tests/dataset/test_io.py`
- Modify: `README.md`

**Interfaces:**
- `read_records(path) -> list[dict]`
- `write_records(records, path) -> None`

- [ ] Write failing tests for JSONL round-tripping and the `dataset build` command’s output.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement JSONL I/O, dataset build wiring, and documented setup/test/smoke commands.
- [ ] Run `pytest -q` and a CLI smoke command; verify both pass.

### Verification

- [ ] Run the full test suite with `pytest -q`.
- [ ] Run `python -m xssharden --help`.
- [ ] Check that no test imports a detector-specific feature implementation.
- [ ] Review the diff for accidental credentials, real external payload execution, or committed generated data.
