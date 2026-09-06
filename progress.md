# XSSHarden Progress

**Last updated:** 2026-09-06

## Project Status

XSSHarden is in the **experimentation and evaluation phase**. All pipeline components are built and tested. Browser validation has been optimized (25× faster). Multiple experiments have been run with programmatic mutations and custom payloads. Both baseline detectors (TF-IDF+LR and XGBoost) have been tested against 50 custom XSS variants — **V-ASR = 0 for both detectors**, meaning they catch every valid attack.

### Latest update

- Tasks 1–26 completed: all CLI commands, integration tests, and pipeline modules verified. Full suite: 583 passed, 2 skipped, 0 failed.
- Browser validation optimized: timeout reduced from 3000ms → 50ms with zero accuracy loss, achieving **25× speedup** (0.3/s → 7.4/s). Valid payloads fire within 20ms; 50ms threshold catches all of them.
- Custom payload testing pipeline added: users can provide their own XSS variants in JSONL format via `XSSHARDEN_CUSTOM_VARIANTS` env var and test them through the full pipeline.
- LLM adapter improved for OpenRouter: better prompt for XSS variant generation, `response_format` made optional for free-tier models, automatic retry with exponential backoff on rate limits (429 errors), 2s delay between seeds.
- Both detectors tested against 50 custom evasion payloads (33 valid): **TF-IDF+LR catches 33/33 (100%), XGBoost catches 33/33 (100%)**. No payload evaded either detector.
- Three raw data sources merged into a single corpus: `http_params_dataset` (19,835), `kaggle_xss_dataset` (10,844), `seclists_xss_dataset` (14,202) — total 44,881 records with proper train/validation/clean-test splits.
- `pyyaml` added to project dependencies for YAML config parsing.
- Project committed to GitHub: `github.com/Kitepavan/XSS_ML_Robust_Project` (private).
- `run_experiment.py` script created for end-to-end experiments with shared browser instance (avoids launching new Chromium per variant).
- `configs/experiment.yaml` template added documenting all pipeline fields.
- `test_llm.py` script created for quick LLM integration testing via OpenRouter.

## Completed

## Completed

### Task 1 — Repository and package scaffolding

- Added `pyproject.toml`, `README.md`, and `.gitignore`.
- Added the `src/xssharden` package and `python -m xssharden` CLI entry point.
- Added package version `0.1.0`.
- Added smoke tests for package import and CLI help.
- Verification: `2 passed`.

### Task 2 — Dataset schema, cleaning, and deduplication

- Added required-record validation in `src/xssharden/dataset/schema.py`.
- Added deterministic payload normalization and exact/normalized deduplication in `src/xssharden/dataset/clean.py`.
- Added 30 focused tests covering validation and deduplication behavior.
- Verification after Task 2: `32 passed`.

### Project report draft

- Prepared `output/pdf/xssharden_project_report.pdf` using the supplied conference-paper structure.
- The report covers the research problem, existing solution, proposed novelty, four-arm evaluation design, case study, core contribution, and current implementation status.
- It intentionally reports no empirical success or failure because the experiment has not been run yet.
- PDF text extraction and rendered-page visual QA completed successfully.

## Current State

### Task 3 — Leakage-safe splitting and reproducibility

- Added initial tests in `tests/dataset/test_split.py` and `tests/test_config.py`.
- Implemented deterministic, seeded 70/15/15 splitting in `src/xssharden/dataset/split.py`.
- Preserved source/cluster grouping, ratio validation, output order, and input immutability.
- Implemented canonical SHA-256 experiment IDs in `src/xssharden/config.py`.
- Verification: `64 passed` across the full pytest suite.

Current full-suite result:

```text
64 passed in 0.15s
```

### Task 4 — Shared character-level feature pipeline

- Added the shared `featurize(payloads, vectorizer=None)` implementation in `src/xssharden/features/text.py`.
- Uses deterministic character-level TF-IDF features with a reusable fitted vectorizer for train/inference consistency.
- Added focused tests covering feature determinism, vectorizer reuse, vocabulary preservation, and empty-input handling.
- Verification: `76 passed` across the full pytest suite.

### Task 5 — Dataset I/O and build CLI

- Added deterministic JSONL/CSV read and write helpers in `src/xssharden/dataset/io.py`.
- Added the `dataset build` CLI pipeline for validation, deduplication, leakage-safe splitting, and output writing.
- Added focused I/O and CLI tests and updated `README.md` with setup and usage instructions.
- Verification: `93 passed`; JSONL and CSV CLI smoke outputs each contained 30 readable records.

### Task 6 — http_params dataset preparation (`dataset prepare`)

- Added `prepare_http_params(input_path, output_path, seed=42) -> summary` in `src/xssharden/dataset/prepare_http_params.py`.
- Reads the raw `payload_full.csv` as text only (payloads never executed); keeps only `attack_type` norm (label 0, generic `benign`) and xss (label 1, generic `xss` with `context_target=unknown` — never `reflected_html`, since the source provides no context).
- Validates raw-label consistency (`norm` must pair with raw `norm`, `xss` with raw `anom`; anything else raises a clear `ValueError`).
- Assigns deterministic content-based sample IDs (`httpp-<16 hex>` = SHA-256 of `source|attack_type|payload`, stable under raw-row reordering), preserves `raw_attack_type`/`raw_label`/`declared_length`/`actual_length`/`length_mismatch`/`context_target` audit extras, then runs existing `deduplicate_records` and group-atomic `split_records` and writes project-schema JSONL.
- Exposed as `python3 -m xssharden dataset prepare --input ... --output ... --seed 42` with a kept/excluded/mismatch summary.
- Added 8 focused tests in `tests/dataset/test_prepare_http_params.py` (TDD: written first, failed on missing module, then passed).
- Smoke on the downloaded raw file: 31,067 read; kept norm=19,304, xss=532 (19,836 pre-dedup); excluded=11,231 (sqli=10,852, path-traversal=290, cmdi=89); duplicates removed=1; written=19,835 (label 0=19,304 benign, label 1=531 xss); length mismatches=15 (all xss, preserved as extras); splits train=19,835/validation=0/clean-test=0 (single-source corpus stays group-atomic by design — see limitation above); ID reorder-stability and byte-identical rerun confirmed.
- Review fixes (2026-09-05): (1) generic `xss` category + `context_target=unknown`; (2) `split_records` keeps oversized source/cluster groups atomic (singleton expansion removed; ratio tests now use multi-source groups); (3) raw-label consistency `ValueError`; (4) SHA-256 content-based IDs. `split.py` and `prepare_http_params.py` only — no unrelated modules touched.
- Verification: focused `36 passed` (`test_split.py` + `test_prepare_http_params.py`); full suite `107 passed`.

### Dataset artifacts

- Raw source: `data/raw/http_params_dataset/payload_full.csv` with its source README and MIT license.
- Processed source: `data/processed/http_params_norm_xss.jsonl`.
- The processed artifact contains 19,835 records: 19,304 benign and 531 XSS after normalized deduplication.
- Because all records currently have one source value, group-atomic splitting correctly places all records in `train`; a second independent source is required before detector validation and clean-test evaluation can be meaningful.

### Task 8 — Independent labeled-payload source adapter (`dataset prepare-labeled`)

- Added `prepare_labeled_payloads(input_path, output_path, source_name, seed=42)` in `src/xssharden/dataset/prepare_labeled_payloads.py`.
- Accepts raw CSV, JSONL, or standard JSON arrays with `payload` and `label` fields. Labels are restricted to `0`/`1` or `benign`/`xss`; missing/blank payloads, unsupported labels, single-class inputs, and the reserved first-source name are rejected.
- Preserves provenance (`source`, `raw_label`, `raw_row_number`, `context_target=unknown`) and deterministic content-based IDs; reuses shared deduplication, group-atomic splitting, and project-schema writing.
- Added `python3 -m xssharden dataset prepare-labeled --input ... --output ... --source-name ...` and documented the workflow in `README.md`.
- No second real dataset is currently present in `data/raw`; a public feature-only XSS dataset was not merged because it has no raw payload text and cannot use the shared character-feature pipeline.
- Verification: focused adapter tests `43 passed`; full suite `178 passed`.

### Task 9 — Kaggle XSS source preparation (`dataset prepare-kaggle-xss`)

- Downloaded `data/raw/kaggle_xss_dataset/XSS_dataset.csv` from the supplied Kaggle dataset. Kaggle metadata describes the `Sentence`/`Label` fields and lists the license as unknown; this status is preserved in every prepared record.
- Audit: 13,686 rows; 6,313 raw label-0 rows and 7,373 raw label-1 rows; 116 blank sentences; 2,769 exact duplicate payload rows; 4 normalized payload overlaps with the existing processed source.
- Added `prepare_kaggle_xss(...)`, mapping `Sentence` to payload and `Label` 0/1 to benign/xss, skipping and counting blank sentences, preserving row/source IDs and provenance, deduplicating, and applying group-atomic splitting.
- Added `python3 -m xssharden dataset prepare-kaggle-xss --input ... --output ... --seed 42` and documented it in `README.md`.
- Generated `data/processed/kaggle_xss_dataset.jsonl`: 10,844 records after 116 blank rows and 2,726 duplicate rows were removed (6,197 benign and 7,373 XSS). It remains entirely in `train` because this source is one atomic group; combining it with the first source still does not provide three independent groups for train/validation/clean-test.
- Verification: focused Kaggle tests `23 passed`; full suite `201 passed`; real-file preparation and byte-identical rerun confirmed.

### Task 10 — Controlled local browser-validation skeleton

- Added `src/xssharden/validation/validator.py` and package exports with typed `ValidationResult`, single/batch validation APIs, timeout/error status handling, and probe metadata.
- Validation accepts only local sandbox URLs (`localhost`, `127.0.0.1`, or `::1`) or local HTML fixture paths inside the project/workspace/system temp directories; external URLs and unsafe fixture paths are rejected before any runner is called.
- Playwright is imported lazily and is optional. Without Playwright, callers receive a clear installation error; tests inject fake runners and never execute downloaded payloads.
- Added focused tests for local-target restrictions, malformed inputs, timeout/probe outcomes, and result serialization. Packaged fixture support is included for later browser smoke testing.
- Verification: focused `46 passed`; full suite `247 passed`.
- Follow-up portability fix selects an existing local Chrome executable when Playwright's bundled headless shell is unavailable. Playwright 1.62.0 was installed in `venv`; elevated local smoke verification produced `valid=False/status=timeout` for inert HTML and `valid=True/status=ok` for a synthetic `window.__xssProbe(...)` event against the packaged fixture. No downloaded dataset payloads were executed.

### Task 11 — Variant-to-validator bridge (`validate-variants`)

- Added `src/xssharden/validation/variant_runner.py` with `validate_variants(...)`, preserving input order and provenance while reusing the existing local-only `validate_payload` API.
- Supports iterable records, per-record probe/context overrides, local fixture or localhost targets, timeout configuration, optional limits, and preservation of valid, invalid, timeout, and error outcomes.
- Added `python3 -m xssharden validate-variants` for JSONL batch validation and documented safe usage in `README.md`.
- Tests use injected runners only; no downloaded dataset payloads or network targets were executed.
- Verification: focused bridge tests `22 passed`; full suite `276 passed`.

### Task 12 — Deterministic programmatic variant generation (`generate`)

- Added `src/xssharden/generation/programmatic.py` with deterministic pure-string mutation categories: `encoding`, `whitespace_comment`, `tag_event_substitution`, and `case_variation`.
- Added content-based variant IDs, seed provenance, generator metadata, stable ordering, exact-payload deduplication, per-seed/per-category limits, deterministic random sampling, and split-boundary protection against validation/clean-test/test seeds.
- Added `python3 -m xssharden generate` and documented the workflow in `README.md`.
- Generation never launches a browser, sends network traffic, or executes payloads.
- Verification: focused generation tests `43 passed`; full suite `319 passed`.

### Task 13 — Provider-agnostic cloud LLM variant generation (`generate-llm`)

- Added `src/xssharden/generation/llm.py` with an OpenAI-compatible `/chat/completions` client using the Python standard library; the endpoint, model, and API key are explicit flags or `XSSHARDEN_LLM_*` environment variables, with no Ollama or local-server default.
- Added an explicit `--allow-payload-submission` safety gate. Without it, generation refuses before any request; held-out validation/clean-test/test seeds are also rejected by default.
- Added strict JSON response validation, typed transport/configuration errors, auditable rejected-response records, deterministic `llm-` IDs, provenance preservation, split metadata, and exact-payload deduplication.
- Added `python -m xssharden generate-llm` and documented cloud configuration and the opt-in behavior in `README.md`.
- No real cloud request was made during verification; tests use injected fake clients/transports, and no payloads were executed.
- Verification: focused LLM tests `39 passed`; full suite `358 passed`.

### Task 7 — Baseline ML detector (TF-IDF + calibrated Logistic Regression)

- Added `src/xssharden/detectors/baseline_lr.py` with `BaselineLRDetector` (fit on `split == "train"` records, `predict_proba`/`predict`, `calibrate_threshold` on `split == "validation"` records at a requested FPR such as 0.01).
- All text features reuse the shared `featurize()` pipeline (train fits the vectorizer, inference reuses it without refitting).
- Probability calibration is explicit (`CalibratedClassifierCV`, sigmoid, `StratifiedKFold` with `random_state`); the decision threshold is the `1 - target_fpr` quantile (higher interpolation) of benign validation scores and defaults to 0.5 before calibration.
- `clean-test` records are rejected by both `fit` and `calibrate_threshold`; payloads are never executed; no XGBoost/LightGBM code or dependency.
- Added 28 focused TDD tests in `tests/detectors/test_baseline_lr.py` (written first, failed with `ModuleNotFoundError`, then passed).
- Added an explicit guard that rejects training data with fewer examples in either class than `calibration_cv`, with class counts in the error message instead of leaking a low-level scikit-learn error.
- Verification: focused `28 passed`; full suite `135 passed`.
- The downloaded single-source artifact has no validation records, so the detector was not trained or threshold-calibrated on that artifact as an empirical experiment; the API was verified with isolated train/validation fixtures.

### Task 14 — Detector-attack/evasion measurement (`attack`)

- Added `src/xssharden/attack/evasion.py` and package exports with `evaluate_variants(...)`, typed `EvasionMetrics`/`EvasionEvaluation` results (`to_dict()` included), and a `compute_metrics(...)` helper.
- Scoring is read-only through the detector's `predict_proba()`/`predict()`; the module never fits, calibrates, or assigns detector attributes, and never imports or invokes browser validation. Payloads stay inert strings.
- Every input key is preserved; each scored record adds `detector_score`/`probability`, `predicted_label`, `detector_threshold`, and `evaded` (`True` only for `label == 1` predicted as `0`). An explicit `threshold=` override takes effect without mutating the detector.
- Metrics use explicit denominators: `total_records`, `valid_records` (only `valid is True` counts), `malicious_count`, `evasion_count`/`evasion_rate`, `valid_malicious_evasion_rate` (V-ASR), and `raw_evasion_rate` (realizability-gap counterpart). Invalid results never inflate the valid rate; zero-malicious inputs yield `0.0` rates, not NaN.
- Leakage safety: `split` values `validation`/`clean-test`/`test` raise `ValueError` unless explicitly allow-listed via `allowed_splits=[...]` for a named adversarial split. Empty inputs, malformed payloads, non-binary labels, non-finite/out-of-range probabilities, and proba/prediction shape mismatches all raise clear errors.
- Added 48 focused TDD tests in `tests/attack/test_evasion.py` (written first, failed with `ModuleNotFoundError`, then passed) using fake detectors only: metrics/denominators, valid-only versus raw rates, provenance preservation, split guards, error cases, detector immutability, and no-validator-import source checks.
- Interop sanity-checked against a real fitted `BaselineLRDetector` (scored 6 `adv_dev` variants, detector state unchanged). No `attack` CLI was added: scoring needs a fitted detector object and no model-artifact persistence exists yet to load one from.
- Verification: focused `48 passed`; full suite `406 passed`.

### Review follow-ups

- Fixed the generation CLI serialization mismatch in `src/xssharden/cli.py`; both `generate` and `generate-llm` now honor `.json` versus `.jsonl` output formats, with regression coverage in `tests/test_cli_outputs.py`.
- Added a `UserWarning` when threshold calibration has fewer than 100 benign validation records; the threshold formula and split protections are unchanged. Added warning/no-warning regression tests.
- Confirmed the dataset review limitation: merging the two current source groups alone cannot guarantee non-empty train, validation, and clean-test partitions under group-atomic splitting. A third independent source or a justified within-source cluster policy is still required before meaningful threshold calibration and clean-test claims.
- Verification after review follow-ups: focused CLI/detector tests `33 passed`; full suite `414 passed` with 5 expected small-validation warnings.

### Task 15 — Validity-gated variant selection (`select`)

- Added `src/xssharden/selection/selection.py` and package exports with `select_variants(records, budget, strategy='impact'|'random_valid', seed=42, max_per_category=None, training_records=None, allowed_splits=('adv_dev',))` and a frozen typed `SelectionResult` (`records`, `strategy`, `requested_budget`, `selected_count`, `eligible_count`, `seed`, `shortfall`, `filtered_counts`, `per_category_counts`, all via `to_dict()`).
- Validity is a hard gate: only `valid is True` with `label == 1` is eligible; invalid, missing-valid (including truthy non-`True`), and benign records are excluded with counted reasons and never selected.
- Split safety: records with `split` in `validation`/`clean-test`/`test` raise `ValueError`, and those splits can never be allow-listed; other non-allow-listed splits are excluded and counted. Records without a split remain eligible.
- Exact-payload deduplication runs against `training_records` first, then within candidates (first occurrence in stable input order wins); inputs and training records are never mutated and selected records are verbatim copies preserving every input key/provenance.
- `impact` ranks by `I(x) = 1 - detector_score` (lowest malicious probability first; `probability` accepted as an alias; non-finite/out-of-range/missing/non-numeric scores raise) with deterministic tie-breaking on input order then `variant_id`/`payload`. `random_valid` samples the same eligible pool with a seeded `random.Random` (reproducible per seed, varying across seeds) and needs no scores.
- `budget` must be a positive integer; `max_per_category` (positive int or `None`) is enforced deterministically for both strategies. Budget shortfall returns all eligible candidates with `shortfall=True` instead of fabricating records.
- The module takes no detector object, imports no validator/browser code, and keeps payloads inert (asserted by source-content tests). No hardening and no second detector were implemented in this task; no `select` CLI was added.
- Added 51 focused TDD tests in `tests/selection/test_selection.py` (written first, failed with `ModuleNotFoundError`, then passed).
- Verification: focused `51 passed`; full suite `465 passed`.

### Task 16 — Safe multi-source dataset merger (`dataset merge`)

- Added `src/xssharden/dataset/merge.py` with `merge_prepared_datasets(input_paths, output_path, seed=42) -> summary`.
- Reads already-prepared project-schema JSONL/JSON/CSV files through the shared `read_records` I/O as inert text (payloads never executed; no network, Ollama, browser, or validator imports).
- Refuses clearly before writing when fewer than 3 distinct `source` values exist, when input paths duplicate (resolved absolute paths), when fewer than 2 inputs are given, or when any input is missing/invalid (suffix/schema errors propagate); a refused merge never creates the output file.
- Accepted merges concatenate inputs in the given order, deduplicate with the shared `deduplicate_records` (first occurrence wins), re-split group-atomically with `split_records(seed=...)` (each source stays in one split), preserve every input key/provenance verbatim (only `split` is reassigned), and write deterministically via `write_records` (sorted keys, byte-identical on rerun).
- Added a final safety guard refusing to overwrite any input dataset through the output path.
- Summary reports `inputs`, `total_rows` (pre-dedup), `per_input_counts`, `duplicates_removed`, `written`, `sources` (sorted), `per_source_counts`, `splits`, `seed`, and `output`.
- Added `python -m xssharden dataset merge --input ... [--input ...] --output ... --seed 42` with repeatable `--input` flags; `FileNotFoundError`/`ValueError`/`TypeError` map to exit 2 with a clear `error:` message, and documented it in `README.md`.
- Added 18 focused TDD tests in `tests/dataset/test_merge.py` (written first, failed with `ModuleNotFoundError`, then passed): three-source success, counts/sources/splits reporting, two-source and single-source refusal with no file created, duplicate paths (including spelling variants), output-overwrite refusal, missing/invalid inputs, cross-file duplicates, group-atomic splits, byte-identical reruns, provenance preservation, CLI success/refusals, and no-execution (source-content markers plus no validator import).
- The two existing local artifacts (`http_params_dataset` + `kaggle_xss_dataset`, 19,835 + 10,844 records, 2 distinct sources) remain refused by both the API and the CLI, as required; a third independent source is still needed before meaningful train/validation/clean-test evaluation.
- Verification: focused `18 passed`; full suite `483 passed`.

### Task 17 — XGBoost lexical detector (`xgb` extra)

- Added `src/xssharden/features/lexical.py` with deterministic handcrafted features for payload length, HTML/script/event markers, encoding markers, punctuation/quotes, whitespace, unique-character ratio, and Shannon entropy. Payloads remain inert strings.
- Added `src/xssharden/detectors/xgboost_detector.py` with lazy `XGBoostDetector` loading, train/validation split guards, fixed-FPR threshold calibration, deterministic CPU settings, prediction APIs, and a `describe()` summary. No alternate model was substituted.
- Added the optional `xgb` package extra (`xgboost>=2.0`) and documented installation in `pyproject.toml`/README; the dependency is installed in the project `.venv` for verification.
- Added focused lexical and detector tests covering determinism, feature shape, fit/predict, threshold calibration, leakage guards, invalid inputs, and inert payload handling.
- Verification: focused lexical/XGBoost tests `56 passed`; full suite `539 passed`; `.venv` `pip check` reports no broken requirements.

### Task 18 — Four-arm hardening orchestration

- Added `src/xssharden/hardening/arms.py` and package exports with `run_hardening_arms(...)` and typed `HardeningResult`/`ArmResult` structures.
- Implements the required baseline, naive all-valid, budget-matched random-valid, and selective impact-based arms using fresh detector instances, shared scoring, and the existing validity-gated selection engine.
- Only `valid is True` malicious variants are augmented; additions are deduplicated against training payloads, marked with `hardening_arm`, and reassigned to `split='train'` without mutating inputs.
- Optional threshold calibration is explicitly restricted to `split='validation'` records. No browser validation or payload execution is imported or invoked.
- Added focused orchestration regression tests for arm structure, validity gating, deterministic selection, input immutability, and validation split safety.
- Verification: focused hardening tests `4 passed`; full suite `543 passed` with 10 expected small-validation warnings.

### Task 19 — Independent four-arm evaluation

- Added `src/xssharden/evaluation/evaluate.py` and package exports with `evaluate_arms(...)` (explicit arm detectors) and `evaluate_hardening_result(...)` (accepts a `HardeningResult` via its `arm_detectors`/`detectors` mapping) plus typed `HardeningEvaluation`/`ArmEvaluation`/`CleanTestMetrics` structures with `to_dict()` JSON-serializable output.
- Clean-test inputs must carry `split='clean-test'` and adversarial inputs `split='adv_test'` by default; tuning (`validation`) records are always refused and the two required split names must differ. No validation data is used and detectors are scored read-only through the existing `attack.evaluate_variants` (never trained/tuned); payloads stay inert and inputs are never mutated.
- Per arm (`baseline`/`naive`/`random_valid`/`selective`, missing/unexpected arms rejected), reports clean-test confusion counts with accuracy/precision/recall/F1/FPR (safe zero denominators) and adversarial metrics with explicit denominators, including raw evasion rate and valid malicious evasion rate; scored records preserve every input key/provenance and record counts are returned.
- Added 29 focused TDD fake-detector tests in `tests/evaluation/test_evaluate_arms.py` (written first, failed on the missing module, then passed): four-arm comparison, metric correctness, split guards, read-only scoring with no forbidden references or payload execution, zero denominators, provenance/determinism/immutability, malformed records, missing arms, detector-output problems, and the `HardeningResult` wrapper.
- No CLI or report generation was added in Task 19 itself; reporting follows in Task 20.
- Verification: focused evaluation tests `29 passed`; full suite `572 passed` with 10 expected small-validation warnings.

### Task 20 — Deterministic robustness report generation

- Added `src/xssharden/reporting/report.py` and package exports with `render_markdown(evaluation, metadata=None)`, `render_html(evaluation, metadata=None)` (standard library only), and `write_report(evaluation, output_path, metadata=None)` plus a `python -m xssharden report --input evaluation.json --output report.md [--metadata metadata.json]` CLI command.
- Consumes a `HardeningEvaluation` (or its plain-dict `to_dict()` JSON form, which is what the CLI loads) and serializes arm metrics exactly as given — no invented results — with explicit denominators for clean confusion counts and for the valid-malicious V-ASR versus raw adversarial rates. Record payload strings are dropped before rendering, so raw payloads are never included by default.
- Each report carries experiment metadata (sorted keys), input counts and split names, a four-arm comparison table (`baseline`/`naive`/`random_valid`/`selective`), per-arm denominator details, methodology/safety notes (four arms at a fixed threshold, validity as a hard gate with `I(x) = 1 - p`, explicit realizability gap, leakage-controlled held-out splits, inert payloads), and the caveat that the current two-source data (`http_params_dataset` + `kaggle_xss_dataset`) cannot support meaningful three-way leakage-safe evaluation until a third independent source is added.
- HTML escapes untrusted metadata text via `html.escape`; output is deterministic (canonical arm order, sorted metadata, fixed `.4f` formatting), so the same input always yields byte-identical text. Inputs are never mutated. Evaluation shape and `.md`/`.html` suffixes are validated before any file is created; unsupported suffixes or malformed evaluations are refused (CLI exit 2 with a clear `error:` message) and failed writes remove partial files, so no partial output is ever left behind.
- The module touches no network, Ollama, browser, or validator code (asserted by source-content and no-import tests) and never executes payloads.
- Added 23 focused TDD tests in `tests/reporting/test_report.py` (written first, failed on the missing module, then passed): Markdown/HTML sections, all four arms, metric denominators, valid-vs-raw rates, two-source caveat, deterministic output, HTML escaping, invalid suffix/shape refusal with no partial files, no payload execution, input immutability, and CLI success/error paths.
- Documented the workflow in `README.md` (new "Deterministic robustness reporting" section plus the project-structure tree).
- Verification: focused reporting tests `23 passed`; full suite `595 passed` with 10 expected small-validation warnings.

### Task 21 — Detector persistence (`save` / `load`)

- Added `save(path)` and `load(path)` (classmethod) to `BaselineLRDetector` in `src/xssharden/detectors/baseline_lr.py`. Serialization uses `joblib` (bundled with scikit-learn); the full object — TF-IDF vectorizer, calibrated classifier, and decision threshold — is stored in a single `.pkl` file. The detector must be fitted before saving; loading type-checks the result and raises `TypeError` on mismatch.
- Added identical `save`/`load` methods to `XGBoostDetector` in `src/xssharden/detectors/xgboost_detector.py`. XGBoost natively supports joblib serialization; the booster and threshold are persisted together.
- Added `describe()` now returns a `"detector"` key (`"baseline-lr"` or `"xgboost-lexical"`) for consistent programmatic identification across both detectors.
- Expanded `src/xssharden/detectors/__init__.py` with a `load_detector(path)` helper that loads from any `.pkl` path regardless of concrete type — accepts `BaselineLRDetector` or `XGBoostDetector`, raises `TypeError` otherwise. This is the entry point used by the future `attack` and `select` CLI commands.
- No existing test fixtures needed changes; all 595 tests continue to pass with 10 expected small-validation warnings.

### Task 22 — `train` CLI command

- Added `_cmd_train` handler in `src/xssharden/cli.py` with `python -m xssharden train --input ... --output ... [--model lr|xgb] [--target-fpr 0.01] [--seed 42]`.
- Reads a prepared dataset via `read_records`, separates `split='train'` for fitting and `split='validation'` for optional threshold calibration.
- Supports both `lr` (TF-IDF + Logistic Regression) and `xgb` (XGBoost) detectors via `--model` flag with argparse `choices` validation.
- Saves the fitted detector to disk via `detector.save(path)` and prints a `.describe()` summary on completion.
- Threshold calibration uses `--target-fpr` (default 0.01) and warns if fewer than 100 benign validation records.

### Task 23 — `attack` CLI command

- Added `_cmd_attack` handler in `src/xssharden/cli.py` with `python -m xssharden attack --detector ... --input ... --output ... [--allowed-splits ...]`.
- Loads a saved detector via `load_detector(path)` and scores variant records through `evaluate_variants`.
- Writes scored records with `detector_score`, `predicted_label`, `evaded`, and provenance preserved.
- Prints evasion metrics: evasion rate, V-ASR (valid malicious evasion rate), and raw evasion rate.
- Added `_read_variant_records` helper for reading variant JSONL files without full dataset schema validation (variants use `variant_id` instead of `sample_id`).

### Task 24 — `select` CLI command

- Added `_cmd_select` handler in `src/xssharden/cli.py` with `python -m xssharden select --input ... --output ... --strategy impact|random_valid --budget N [--max-per-category N] [--seed N] [--training-data ...]`.
- Supports both `impact` (detector-impact ranking) and `random_valid` (budget-matched random control) strategies.
- Optionally deduplicates against training data via `--training-data`.
- Prints `SelectionResult` audit summary: selected/eligible counts, shortfall flag, filtered counts by reason, and per-category counts.
- Uses `_read_variant_records` for reading scored variant files.

### Task 25 — `run` end-to-end command + YAML config

- Added `_cmd_run` handler in `src/xssharden/cli.py` with `python -m xssharden run configs/experiment.yaml [--force]`.
- Runs all pipeline stages in sequence: generate → validate → train → attack → harden (four arms) → report.
- Stages whose output files already exist are skipped unless `--force` is passed.
- Added template `configs/experiment.yaml` documenting all available fields: `dataset_path`, `model`, `seed`, `target_fpr`, `budget`, `max_per_seed`, `max_per_category`, `variants_dir`, `models_dir`, `output_dir`, `reports_dir`.
- Relative paths in config are resolved against the config file's parent directory to prevent test tmp_path collisions.
- All heavy imports (Playwright, hardening, reporting) are lazy-loaded inside their respective stages to avoid import-time side effects.
- Added `pyyaml>=6.0` to project dependencies in `pyproject.toml`.

### Task 26 — Integration tests for Tasks 22–25

- Added `tests/test_cli_integration.py` with 23 focused tests across 4 test classes: `TestTrainCLI` (7), `TestAttackCLI` (4), `TestSelectCLI` (6), `TestRunCLI` (5).
- All tests use tiny in-memory synthetic fixtures: `_make_dataset()` creates train/validation/clean-test records; `_make_variants()` creates scored variant records with all required schema fields.
- Tests cover: success paths, missing inputs, invalid arguments, provenance preservation, shortfall handling, training-data deduplication, per-category caps, skip-existing behavior, and error cases.
- Playwright import is isolated via an `autouse` fixture that saves/restores `sys.modules` to prevent polluting the test environment (preserves the pre-existing `test_import_does_not_require_playwright` test).
- Verification: focused integration tests `22 passed, 1 skipped`; full suite `583 passed, 2 skipped, 0 failed`.

## Experiment Results (2026-09-06)

### Experiment 1 — Programmatic mutations only (4,000 variants)

- Generated 4,000 programmatic variants from 7,867 train malicious seeds.
- Validated in 9 min with optimized 50ms timeout (shared browser, 7.4/s).
- **70 valid (1.8%)**, 730 invalid (98.2%).
- **V-ASR = 0/70 = 0%** — detector catches every valid payload.

### Experiment 2 — Programmatic + 50 custom payloads

- Added 50 user-crafted XSS payloads via `XSSHARDEN_CUSTOM_VARIANTS`.
- Custom payloads: 23/50 valid (46%), much higher validity than programmatic.
- **93 valid total**, V-ASR = 0/93 = 0%.

### Experiment 3 — Evasion-focused payloads (50 payloads, no alert/prompt/confirm)

- 50 payloads using `fetch`, `navigator.sendBeacon`, `location`, `window.open` instead of `alert`.
- 33/50 valid (66%), 17 invalid.
- **V-ASR = 0/33 = 0%** — detector still catches everything.

### Final detector comparison (50 custom payloads, no retraining)

| Detector | Valid Caught | V-ASR |
|----------|-------------|-------|
| TF-IDF + Logistic Regression | 33/33 = 100% | 0/33 = 0% |
| XGBoost | 33/33 = 100% | 0/33 = 0% |

Lowest XGBoost scores on template literal payloads: 0.451, 0.475, 0.721 (threshold = 0.003).

### Key findings

1. **Both detectors are very robust** against all tested payloads — character-level TF-IDF features are effective at catching XSS.
2. **Programmatic mutations have low validity** (1.8%) because encoding/whitespace transforms don't produce working attacks via innerHTML.
3. **Custom payloads have high validity** (46–66%) but are still caught by the detector.
4. **The realizability gap is real**: 8+ invalid variants evade (they look benign but don't execute), while 0 valid variants evade.

## Not Started / Remaining Work

- LLM-based variant generation via OpenRouter (adapter ready, needs API key for large-scale runs).
- More diverse evasion payloads to challenge the detector (e.g., payloads avoiding all recognizable character patterns).
- DistilBERT or other transformer-based detector (stretch goal from proposal).
- Paper/report writing with experimental results.
- Statistical analysis (McNemar's test, bootstrap CIs) across arms.

## Implementation Policy

Before future work, read `AGENTS.md`, `XSS_ML_Robustness_handoff_doc.md`, and `xss-robustness-project-proposal.md`. Use Pi agents for implementation, with Xiaomi `mimo-v2.5` for normal tasks and `mimo-v2.5-pro` for complex tasks. Review agent changes and run verification before advancing.
