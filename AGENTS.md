# Repository Guidelines

## Required Context Review

Before making plans, changing code, or interpreting results, always read [XSS_ML_Robustness_handoff_doc.md](/home/pavan/XSS_ML_Robust_Project/XSS_ML_Robustness_handoff_doc.md), [xss-robustness-project-proposal.md](/home/pavan/XSS_ML_Robust_Project/xss-robustness-project-proposal.md), and [progress.md](/home/pavan/XSS_ML_Robust_Project/progress.md). The handoff governs decisions, scope, architecture, and workflow; the proposal governs methodology, stack, schedule, and success criteria; progress records the current implementation state. Do not reopen settled decisions without concrete evidence.

## Project Overview

XSSHarden is a runnable Python research pipeline—not a production WAF—that generates XSS variants through deterministic mutations and an LLM, verifies browser execution in a controlled sandbox, measures detector evasion, and selectively hardens ML-based XSS detectors. The core experiment compares baseline, all-valid (naive), budget-matched random-valid, and validity-gated detector-impact-selected augmentation. Preserve leakage-safe splits, fresh category-holdout adversarial tests, and the mandatory random-valid control.

## Project Structure

- `src/`: Python modules for dataset processing, detectors, generation, validation, attack simulation, selection, hardening, evaluation, and reporting.
- `data/`: raw, processed, seed, and split datasets. Keep train, validation, clean-test, and final adversarial-test data separated.
- `sandbox/`: controlled vulnerable application and Playwright validator.
- `models/`, `variants/`, `experiments/`, and `reports/`: model artifacts, JSONL variant records, experiment outputs, and generated reports.
- `tests/`: pytest unit, integration, and validator regression tests; `configs/`: reproducible experiment configuration.

## Build, Test, and Development Commands

No executable commands are committed yet. The planned CLI includes `xssharden run experiment.yaml` for a reproducible end-to-end run, `pytest` for sanity/regression tests, and Playwright’s test command for browser validation. Document final setup and entry points in `README.md` when implementing them.

## Agent Orchestration & Model Routing

For implementation tasks, act as the orchestrator: break work into focused tasks, delegate implementation to Pi agents, review their changes and test results, and integrate only verified work. Do not implement project features directly when a Pi agent can execute the task. Route normal implementation tasks through the Xiaomi provider using the exact `mimo-v2.5` model. Route complex, large, or difficult tasks through the Xiaomi provider using the exact `mimo-v2.5-pro` model. If the required provider or model is unavailable, stop and report the blocker rather than silently substituting another model.

## Coding Style & Naming

Use Python 3.10+, 4-space indentation, type hints on public functions, small modules, and deterministic seeds. Prefer `snake_case` for files/functions/variables, `PascalCase` for classes, and descriptive experiment IDs. Keep the shared `featurize()` pipeline identical across detectors.

## Testing Guidelines

Use pytest with behavior-focused names such as `test_validator_rejects_non_executable_payload`. Cover valid/invalid payloads, split leakage, provenance, deduplication, and threshold calibration. Never use final adversarial-test data for tuning or training.

## Security & Research Integrity

Run payloads only inside the controlled local sandbox. Treat LLM output as untrusted: browser validation determines realizability. Preserve provenance for every generated variant, include the mandatory random-valid control arm, and report negative results honestly.

## Commits & Pull Requests

There is no existing Git history to establish a commit convention. Use imperative, focused messages (for example, `Add Playwright validator regression tests`). Pull requests should explain the research or code change, identify affected experiment arms and data boundaries, include test commands and results, and attach report screenshots when output presentation changes.
