# XSSHarden

A runnable Python research pipeline for XSS variant generation, browser verification, detector evasion measurement, and selective ML detector hardening.

## Setup

Requirements: Python 3.10+.

```bash
# Create an isolated environment (if it does not already exist)
python3 -m venv .venv

# Install in editable mode with development dependencies
./.venv/bin/python -m pip install -e ".[dev]"

# Run checks inside the same environment
./.venv/bin/python -m pytest -q
```

## Quick Start

```bash
# CLI help
python -m xssharden --help

# Dataset build help
python -m xssharden dataset build --help

# Build a dataset: deduplicate + assign leakage-safe splits
python -m xssharden dataset build --input data/raw.jsonl --output data/built.jsonl

# Same, with an explicit seed and CSV output
python -m xssharden dataset build --input data/raw.csv --output data/built.csv --seed 42
```

Input and output files may be JSONL (`.jsonl`/`.json`, one record per line)
or CSV (`.csv`). Every record must contain `sample_id`, `payload`, `label`,
`source`, `attack_category`, and `split`. Records are validated on read and
write, kept in file order, and never executed.

## Preparing an independent labeled-payload source

```bash
# Generic adapter: CSV or JSONL with payload,label fields
python -m xssharden dataset prepare-labeled --help

python -m xssharden dataset prepare-labeled \
  --input data/second_source.csv \
  --output data/second_source.prepared.jsonl \
  --source-name second_source --seed 42
```

Labels may be `0`/`1` or `benign`/`xss` (case-insensitive). Each output
record carries `source` (= `--source-name`), `raw_label`,
`context_target="unknown"`, and `raw_row_number` provenance, plus a
deterministic content-based `sample_id` (`<source>-<16 hex>` over
`source|label|payload`). Rows are deduplicated with the shared cleaning
and split with the group-atomic `split_records`. Missing/blank payloads,
unsupported labels, and single-class inputs are rejected; `source_name`
equal to `http_params_dataset` is refused to prevent accidental
self-merging. Payloads are read as inert text and never executed.

## Preparing the Kaggle XSS dataset

```bash
# Kaggle adapter: XSS_dataset.csv with Sentence,Label columns
python -m xssharden dataset prepare-kaggle-xss --help

python -m xssharden dataset prepare-kaggle-xss \
  --input data/raw/kaggle_xss_dataset/XSS_dataset.csv \
  --output data/processed/kaggle_xss.prepared.jsonl \
  --seed 42
```

`Sentence` maps to `payload` (verbatim) and `Label` `0`/`1` maps to
`benign` (`0`) / `xss` (`1`); any other label is rejected, and a missing
`Sentence`/`Label` column is rejected. Blank (empty/whitespace-only)
`Sentence` rows are skipped and counted as `skipped_blank` in the
summary. Each output record carries `source="kaggle_xss_dataset"`,
`raw_label`, `raw_row_number` (1-indexed data-row position),
`raw_source_id` (the original first-column index value),
`context_target="unknown"`, and `license_status="unknown"`, plus a
deterministic content-based `sample_id`
(`kaggle_xss_dataset-<16 hex>` over `source|label|payload`, stable under
raw-row reordering). Rows are deduplicated with the shared cleaning and
split with the group-atomic `split_records`. Payloads are read as inert
text and never executed.

## Browser validation (controlled local sandbox)

```bash
# Validate one payload against the packaged local fixture
python -m xssharden validate --payload "hello" --help
python -m xssharden validate --payload "hello"

# Explicit local fixture or local sandbox URL only
python -m xssharden validate --payload "hello" --fixture src/xssharden/validation/fixtures/sandbox.html
python -m xssharden validate --payload "hello" --sandbox-url http://localhost:8000/sandbox.html
```

Only `http(s)://localhost`, `http(s)://127.0.0.1`, `http(s)://[::1]`, or a
local HTML fixture file is accepted; any other URL is rejected before the
payload is sent, so payloads never leave the host. Each result records
`payload`, `valid`, `context_target`, `probe`, `duration_ms`/`timeout_ms`/
`timed_out`, and `error`/`status`. Playwright/Chromium is optional and
imported lazily: without it the API raises a `RuntimeError` telling you to
run `pip install playwright` followed by `playwright install chromium`.
Programmatic use (tests inject a fake `_runner` so no browser is needed):

```python
from xssharden.validation import validate_payload

result = validate_payload(
    "hello",
    sandbox_url="http://localhost:8000/sandbox.html",
    probe="alert",
    timeout_ms=3000,
)
print(result.valid, result.status)
```

## Validating generated variant records (batch)

```bash
# Batch-validate variant records (one JSON object per line) to validated results
python -m xssharden validate-variants --help

python -m xssharden validate-variants \
  --input variants/generated.jsonl \
  --output variants/validated.jsonl \
  --sandbox-url http://localhost:8000/sandbox.html

# Same, with the packaged fixture, an explicit probe default, and a limit
python -m xssharden validate-variants \
  --input variants/generated.jsonl \
  --output variants/validated.jsonl \
  --probe alert --context-target reflected_html --timeout-ms 3000 --limit 100
```

Each input record must contain a non-empty `payload` string plus provenance
such as `sample_id`, `seed_id`, `mutation_category`, and `source`; all input
keys are preserved and each output adds `valid`, `status`, `timed_out`,
`error`, `duration_ms`/`timeout_ms`, `probe`, and `context_target` in input
order (`invalid`/`timeout`/`error` outcomes are kept, never filtered).
Exactly one of `--fixture`/`--sandbox-url` is required and only local
targets are accepted. Programmatic use (tests inject a fake `_runner` so no
browser or network is needed):

```python
from xssharden.validation import validate_variants

results = validate_variants(
    [{"sample_id": "S001", "seed_id": "SEED-1",
      "mutation_category": "encoding", "source": "adv_dev",
      "context_target": "reflected_html", "payload": "hello"}],
    sandbox_url="http://localhost:8000/sandbox.html",
    probe="alert",
    timeout_ms=3000,
)
print(results[0]["valid"], results[0]["status"])
```

## Generating deterministic programmatic variants

```bash
# Generate variants from project-schema seed records (train/adv_dev only)
python -m xssharden generate --help

python -m xssharden generate \
  --input data/seeds.jsonl \
  --output variants/generated.jsonl

# Same, restricted to encoding mutations with caps
python -m xssharden generate \
  --input data/seeds.jsonl \
  --output variants/generated.jsonl \
  --categories encoding --seed 42 --max-per-seed 4 --max-per-category 500
```

Seeds are read as inert text and never executed. Each seed must contain a
non-empty `payload` and `sample_id`; `label`/`attack_category` are preserved
when present. Supported `--categories` values are `encoding`,
`whitespace_comment`, `tag_event_substitution`, and `case_variation` (default:
all four). Every variant record carries `variant_id` (`var-` + 16 hex chars of
SHA-256 over `seed_id|mutation_category|payload`), `seed_id`, `payload`,
`mutation_category`, `context_target`, `source`/`seed_source`, `generator`
(`programmatic`), `generator_version`, `seed_split`, and `split="adv_dev"`.
Outputs are deterministically ordered, deduplicated by exact payload (variants
never repeat a seed payload), and truncated deterministically under
`--max-per-seed`/`--max-per-category` using `--seed`. Seeds whose `split` is
outside `train`/`adv_dev` (for example `validation` or `clean-test`) are
rejected unless `--allow-non-train-seeds` is passed. Programmatic use:

```python
from xssharden.generation import generate_variants

variants = generate_variants(seeds, seed=42, max_per_seed=4)
print(variants[0]["variant_id"], variants[0]["mutation_category"])
```

Programmatic generation does not launch a browser, and the separate cloud-LLM
adapter below is opt-in; neither generation path executes payloads.

## Generating cloud-LLM variants (explicit opt-in)

```bash
# Without opt-in the command refuses before any network request
python -m xssharden generate-llm \
  --input data/seeds.jsonl \
  --output variants/llm.jsonl

# With explicit opt-in; credentials come from the environment
export XSSHARDEN_LLM_BASE_URL="https://api.example.com/v1"
export XSSHARDEN_LLM_API_KEY="<secret>"
export XSSHARDEN_LLM_MODEL="example-model"
python -m xssharden generate-llm \
  --input data/seeds.jsonl \
  --output variants/llm.jsonl \
  --allow-payload-submission
```

Seed payload text is submitted to the configured OpenAI-compatible
`/chat/completions` endpoint only when `--allow-payload-submission` is
passed; the default refuses before any network request. Configuration may be
passed as `--base-url`/`--model`/`--api-key`/`--organization`/`--project`/
`--timeout-s` flags or via `XSSHARDEN_LLM_BASE_URL`, `XSSHARDEN_LLM_API_KEY`,
`XSSHARDEN_LLM_MODEL`, and the optional `XSSHARDEN_LLM_TIMEOUT_S`,
`XSSHARDEN_LLM_ORGANIZATION`, `XSSHARDEN_LLM_PROJECT` variables — secrets are
never hardcoded and there is no local-server default. The model must reply
with strict JSON (`{"variants": [{"seed_id", "payload",
"mutation_category", "context_target"}]}`); malformed JSON, missing fields,
wrong types, blank payloads, unsupported/missing mutation categories, and
`seed_id` mismatches are rejected, counted, and reported per seed — never
silently accepted. Accepted variants carry deterministic `llm-` IDs (SHA-256
over `seed_id|mutation_category|payload`), full seed provenance, `generator`
(`llm`), and the model name, and are deduplicated against seed payloads and
within outputs. Seeds from held-out splits are rejected unless
`--allow-non-train-seeds` is passed. Library use with an injected client
(tests inject fakes and never need credentials or network):

```python
from xssharden.generation import (
    LLMConfig,
    OpenAICompatibleClient,
    generate_llm_variants,
)

config = LLMConfig.from_env()  # reads XSSHARDEN_LLM_* variables
result = generate_llm_variants(
    seeds,
    client=OpenAICompatibleClient(config),
    allow_payload_submission=True,
)
print(len(result.variants), len(result.errors))
```

No browser is launched and no payload is executed during generation.

## Measuring detector evasion on validated variants

```python
from xssharden.attack import evaluate_variants

evaluation = evaluate_variants(validated_records, detector)
print(evaluation.records[0]["evaded"])
print(evaluation.metrics.to_dict())
```

`evaluate_variants` scores already-validated variant records with an
 already-fitted detector through `predict_proba()`/`predict()` only — it
 never fits, calibrates, or otherwise mutates the detector. Every input key
 is preserved and each scored record adds `detector_score`/`probability`
 (the malicious-class probability), `predicted_label`,
 `detector_threshold`, and `evaded` (`True` only when `label == 1` and the
 detector predicts `0`). Aggregate `EvasionMetrics` report `total_records`,
 `valid_records` (only `valid is True` counts — invalid results never
 inflate the valid rate), `malicious_count`, `evasion_count`/`evasion_rate`,
 `valid_malicious_evasion_rate` (V-ASR), and `raw_evasion_rate` (the
 realizability-gap counterpart). Records with `split` in
 `validation`/`clean-test`/`test` are rejected unless explicitly allowed via
 `allowed_splits=[...]` for a named adversarial split; an explicit
 `threshold=` override takes effect without mutating the detector. Payloads
 stay inert strings and the validator is never imported.

## XGBoost lexical detector

The second required detector uses deterministic lexical/handcrafted features
with XGBoost. Install its optional dependency inside the project environment:

```bash
./.venv/bin/python -m pip install -e ".[xgb]"
```

Use `xssharden.detectors.xgboost_detector.XGBoostDetector` for train-only
fitting, validation-only threshold calibration, and inference. The detector
raises a clear installation error if XGBoost is unavailable; it does not
silently substitute another algorithm.

## Detector persistence

Both detectors support joblib-based `save`/`load` persistence so a trained
model can be reused across pipeline steps:

```python
from xssharden.detectors.baseline_lr import BaselineLRDetector
from xssharden.detectors.xgboost_detector import XGBoostDetector
from xssharden.detectors import load_detector  # type-agnostic helper

# Fit and save
det = BaselineLRDetector(random_state=42)
det.fit(train_records)
det.calibrate_threshold(validation_records, target_fpr=0.01)
det.save("models/baseline_lr.pkl")
print(det.describe())  # includes "detector": "baseline-lr"

# Load back (type-checked)
det2 = BaselineLRDetector.load("models/baseline_lr.pkl")
# Or via the type-agnostic helper (accepts either detector)
det3 = load_detector("models/baseline_lr.pkl")
```

The full object — vectorizer, calibrated classifier, and threshold for LR;
booster and threshold for XGBoost — is stored in a single `.pkl` file. The
detector must be fitted before calling `save()`; `load()` raises `TypeError`
if the file contains an unexpected type. `joblib` is bundled with
scikit-learn so no additional dependency is required.

## Four-arm hardening orchestration

```python
from xssharden.hardening import run_hardening_arms

result = run_hardening_arms(
    train_records,
    validated_variants,
    detector_factory=lambda: BaselineLRDetector(random_state=42),
    budget=1000,
    validation_records=validation_records,
)
print(result.added_counts)
```

The runner fits four fresh detectors: baseline, naive all-valid,
budget-matched random-valid, and selective impact-based augmentation. Only
behaviorally valid malicious variants are added, and optional calibration uses
validation records only. It does not execute payloads or launch a browser.

## Independent four-arm evaluation

```python
from xssharden.evaluation import evaluate_arms, evaluate_hardening_result

result = evaluate_arms(arm_detectors, clean_test_records, adversarial_records)
# or: result = evaluate_hardening_result(hardening_result, clean_test_records, adversarial_records)
print(result.arms["selective"].clean_metrics.to_dict())
print(result.arms["selective"].adversarial_metrics.to_dict())
```

`evaluate_arms` scores every arm detector read-only on held-out inputs:
clean-test records must carry `split="clean-test"` and adversarial records
`split="adv_test"` by default, while tuning (`validation`) records are always
refused. Per arm it reports clean-test confusion counts with
accuracy/precision/recall/F1/FPR (safe zero denominators) and adversarial
metrics via the shared read-only attack scorer (raw evasion rate and valid
malicious evasion rate with explicit denominators). Inputs are never mutated,
no detector is trained or tuned, and payloads stay inert. Reports are rendered
from the typed result with `xssharden.reporting` (see below).

## Deterministic robustness reporting

```bash
# From an evaluation JSON file (HardeningEvaluation to_dict() form)
python -m xssharden report --input evaluation.json --output report.md
python -m xssharden report --input evaluation.json --output report.html

# Same, with experiment metadata from a flat JSON object file
python -m xssharden report --input evaluation.json --output report.md \
  --metadata metadata.json
```

```python
from xssharden.reporting import render_markdown, render_html, write_report

text = render_markdown(evaluation, metadata={"experiment_id": "exp-001"})
html = render_html(evaluation)
summary = write_report(evaluation, "report.md")
print(summary["format"], summary["arms"])
```

`render_markdown`/`render_html`/`write_report` consume a
`HardeningEvaluation` (or its plain-dict `to_dict()` JSON form) plus optional
flat metadata and serialize arm metrics exactly as given — no invented results
— with explicit denominators for clean confusion counts and for the
valid-malicious (V-ASR) versus raw adversarial rates. Each report carries
experiment metadata, input counts, a four-arm comparison table, per-arm
denominator details, methodology/safety notes, and the caveat that the current
two-source data (`http_params_dataset` + `kaggle_xss_dataset`) cannot support
meaningful three-way evaluation until a third independent source is added.
HTML rendering uses only the standard library and escapes untrusted metadata
text; raw payloads are never included by default. Output is deterministic
(canonical arm order, sorted metadata keys, fixed number formatting), so the
same input always yields byte-identical text. Evaluation shape and `.md`/`.html`
suffixes are validated up front: unsupported suffixes or malformed evaluations
are refused with a clear `error:` message (CLI exit 2) and never leave partial
files. Inputs are never mutated, payloads stay inert strings, and no network,
browser, or validator code is touched.

## Selecting validity-gated hardening variants

```python
from xssharden.selection import select_variants

result = select_variants(scored_records, budget=1000, strategy="impact")
print(result.records[0]["variant_id"])
print(result.to_dict()["per_category_counts"])
```

`select_variants` picks a budgeted subset of already-scored variant records
 for hardening. Validity is a hard gate: only records with `valid is True`
 and `label == 1` are eligible, and invalid, missing-valid, benign,
 held-out, or duplicated records are never selected. Records with `split` in
 `validation`/`clean-test`/`test` raise `ValueError` (that split can never be
 allow-listed); other non-allow-listed splits are excluded and counted, with
 `allowed_splits=("adv_dev",)` by default. Exact payloads are deduplicated
 against `training_records` and then within candidates, keeping the first
 occurrence in stable input order. The `impact` strategy ranks by detector
 impact `I(x) = 1 - detector_score` (lowest malicious probability first,
 requiring a finite `detector_score`/`probability` in [0, 1]) with
 deterministic tie-breaking on input order then `variant_id`/`payload`; the
 `random_valid` strategy draws the mandatory budget-matched random-valid
 control from the same eligible pool with a seeded deterministic RNG. A
 requested `budget` must be a positive integer; an optional
 `max_per_category` cap is enforced deterministically for both strategies.
 When fewer eligible candidates exist than requested, all of them are
 returned and the shortfall is reported instead of fabricating records. The
 typed `SelectionResult` carries the selected record copies plus audit
 metadata (`strategy`, requested budget, selected/eligible counts, filtered
 counts by reason, `seed`, shortfall flag, and per-category counts). Inputs
 are never mutated, payloads stay inert strings, and the module never
 imports the validator nor trains/tunes any detector. No `select` CLI is
 provided yet.

## Merging prepared datasets (requires 3+ sources)

```bash
# Merge prepared project-schema files into one split corpus
python -m xssharden dataset merge --help

python -m xssharden dataset merge \
  --input data/processed/http_params_norm_xss.jsonl \
  --input data/processed/kaggle_xss_dataset.jsonl \
  --input data/processed/third_source.prepared.jsonl \
  --output data/processed/merged.jsonl --seed 42
```

```python
from xssharden.dataset.merge import merge_prepared_datasets

summary = merge_prepared_datasets(
    ["data/a.prepared.jsonl", "data/b.prepared.jsonl",
     "data/c.prepared.jsonl"],
    "data/merged.jsonl",
    seed=42,
)
print(summary["sources"], summary["splits"])
```

`merge_prepared_datasets` reads already-prepared project-schema files
(JSONL `.jsonl`/`.json` or CSV `.csv`) through the shared I/O layer as
inert text — payloads are never executed and no browser, network, Ollama,
or validator code is touched. At least 2 input files and at least 3
distinct `source` values are required; merges with fewer than 3 sources,
duplicate input paths (compared by resolved absolute path), missing files,
 unsupported suffixes, or schema violations are refused with a clear error
 before anything is written, so a refused merge never creates an output
 file. The two current local artifacts (`http_params_dataset` +
 `kaggle_xss_dataset`, 2 sources) are therefore still refused until a third
 independent source is prepared. Accepted inputs concatenate in the given
 order, deduplicate with the shared cleaner (first occurrence wins),
 re-split group-atomically with `split_records(seed=...)` so each source
 stays in one split, and write deterministically (sorted keys, file order
 preserved, byte-identical on rerun). Every input key — including
 provenance extras such as `raw_label`, `raw_row_number`,
 `context_target`, and `license_status` — is preserved verbatim; only
 `split` is reassigned. The summary reports `inputs`, `total_rows`
 (pre-dedup), `per_input_counts`, `duplicates_removed`, `written`,
 `sources` (sorted), `per_source_counts`, `splits`, `seed`, and `output`.

## Project Structure

```
src/xssharden/
├── __init__.py          # Package root, exposes __version__
├── cli.py               # Command-line interface
├── config.py            # Experiment configuration
├── dataset/
│   ├── __init__.py
│   ├── schema.py        # Record validation
│   ├── clean.py         # Normalization and deduplication
│   ├── split.py         # Leakage-safe splitting
│   ├── io.py            # JSONL/CSV I/O
│   ├── prepare_http_params.py      # First-source (http_params) adapter
│   ├── prepare_labeled_payloads.py # Generic labeled-payload adapter
│   ├── prepare_kaggle_xss.py       # Kaggle XSS_dataset.csv adapter
│   └── merge.py                    # Safe multi-source merger (3+ sources)
├── detectors/
│   ├── __init__.py      # load_detector() type-agnostic helper
│   ├── baseline_lr.py   # TF-IDF + calibrated LR + save/load persistence
│   └── xgboost_detector.py  # Lexical features + XGBoost + save/load persistence
├── attack/
│   ├── __init__.py
│   └── evasion.py       # Read-only evasion scoring + V-ASR/raw metrics
├── selection/
│   ├── __init__.py
│   └── selection.py     # Validity-gated impact/random-valid budgeted selection
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py      # Independent four-arm clean-test + adv_test scoring
├── reporting/
│   ├── __init__.py
│   └── report.py        # Deterministic Markdown/HTML robustness reports
├── generation/
│   ├── __init__.py
│   ├── programmatic.py  # Deterministic programmatic mutations
│   └── llm.py           # Provider-agnostic cloud-LLM adapter (opt-in)
├── validation/
│   ├── __init__.py
│   ├── validator.py     # Local-only Playwright-optional probe validator
│   ├── variant_runner.py # Batch variant-record -> validator connector
│   └── fixtures/
│       └── sandbox.html # Packaged local reflected-HTML sink
└── features/
    ├── __init__.py
    ├── text.py          # Shared character-level TF-IDF featurization (for LR)
    └── lexical.py       # Handcrafted lexical features (for XGBoost)
```

## Testing

```bash
# Full suite
pytest -q

# Focused: dataset I/O and dataset build command
python3 -m pytest tests/dataset/test_io.py -q

# Smoke tests only
pytest tests/test_smoke.py -q
```

## Training a detector

```bash
# Train a TF-IDF + Logistic Regression detector and save it
python -m xssharden train \
  --input data/processed/merged.jsonl \
  --output models/baseline_lr.pkl \
  --model lr --target-fpr 0.01 --seed 42

# Train an XGBoost detector (requires xgboost extra)
python -m xssharden train \
  --input data/processed/merged.jsonl \
  --output models/xgb.pkl \
  --model xgb --seed 42
```

Reads a prepared project-schema dataset, fits the requested detector on
`split='train'` records, optionally calibrates the decision threshold on
`split='validation'` records at the requested false-positive rate, and saves
the fitted detector to a `.pkl` file. Prints a `.describe()` summary on
completion.

## Attacking a detector with validated variants

```bash
# Score variant records against a saved detector
python -m xssharden attack \
  --detector models/baseline_lr.pkl \
  --input variants/validated.jsonl \
  --output variants/scored.jsonl

# Explicitly allowed splits
python -m xssharden attack \
  --detector models/baseline_lr.pkl \
  --input variants/scored.jsonl \
  --output variants/scored2.jsonl \
  --allowed-splits adv_dev
```

Loads a saved detector, scores variant records through `predict_proba`, writes
scored records with `detector_score`/`predicted_label`/`evaded`, and prints
evasion metrics (evasion rate, V-ASR, raw evasion rate).

## Selecting hardening variants

```bash
# Select top-B by detector-impact ranking
python -m xssharden select \
  --input variants/scored.jsonl \
  --output variants/selected.jsonl \
  --strategy impact --budget 500

# Budget-matched random-valid control
python -m xssharden select \
  --input variants/scored.jsonl \
  --output variants/random_control.jsonl \
  --strategy random_valid --budget 500

# With per-category cap and training-data deduplication
python -m xssharden select \
  --input variants/scored.jsonl \
  --output variants/selected.jsonl \
  --strategy impact --budget 500 --max-per-category 100 \
  --training-data data/processed/merged.jsonl
```

Reads scored variant records, applies validity-gated selection (impact ranking
or random-valid control), enforces per-category caps, deduplicates against
training data, and writes the selected subset. Prints a `SelectionResult`
audit summary.

## Running the full pipeline

```bash
# Run all stages from a YAML config
python -m xssharden run configs/experiment.yaml

# Re-run all stages even if output files exist
python -m xssharden run configs/experiment.yaml --force
```

Runs all pipeline stages in sequence: generate → validate → train → attack →
harden (four arms) → report. Stages whose output files already exist are
skipped unless `--force` is passed. See `configs/experiment.yaml` for all
available fields.

## Testing your own XSS payloads

Create a JSONL file with your payloads:

```jsonl
{"payload": "<img src=x onerror=alert(1)>"}
{"payload": "<svg onload=alert(1)>"}
```

Then run the experiment:

```bash
export XSSHARDEN_CUSTOM_VARIANTS="my_payloads.jsonl"
python3 run_experiment.py
```

This validates your payloads through the browser, tests them against both detectors, and reports evasion results. No retraining — just testing.

## LLM variant generation (via OpenRouter)

```bash
export XSSHARDEN_LLM_API_KEY="sk-or-your-key-here"
python3 test_llm.py  # quick test with 3 seeds
```

## Status

Tasks 1–26 complete. 583 tests passing, 2 skipped (xgboost optional).

**Implemented CLI commands:**
`dataset build` · `dataset prepare` · `dataset prepare-labeled` · `dataset prepare-kaggle-xss` · `dataset merge` · `validate` · `validate-variants` · `generate` · `generate-llm` · `train` · `attack` · `select` · `run` · `report`

**Experimental results (50 custom payloads, no retraining):**

| Detector | Valid Caught | V-ASR |
|----------|-------------|-------|
| TF-IDF + Logistic Regression | 33/33 = 100% | 0/33 = 0% |
| XGBoost | 33/33 = 100% | 0/33 = 0% |

Both detectors catch every valid XSS payload. See [`progress.md`](progress.md) for full experiment results and the task-by-task implementation log.
