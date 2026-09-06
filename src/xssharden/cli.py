"""Command-line interface for xssharden."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _write_variant_records(output: str, variants: list[dict]) -> None:
    """Write generated variants according to the requested JSON suffix."""
    suffix = output.lower().rsplit(".", 1)[-1] if "." in output else ""
    if suffix not in ("jsonl", "json"):
        raise ValueError(
            f"unsupported output format '.{suffix}': expected '.jsonl' or '.json'"
        )
    with open(output, "w", encoding="utf-8") as handle:
        if suffix == "json":
            json.dump(variants, handle, sort_keys=True, ensure_ascii=False, indent=2)
            handle.write("\n")
        else:
            for variant in variants:
                handle.write(json.dumps(variant, sort_keys=True, ensure_ascii=False) + "\n")


def _read_variant_records(path: str) -> list[dict]:
    """Read variant records from a JSONL file without schema validation.

    Variant records (generated, validated, scored) use ``variant_id``
    instead of the dataset-schema ``sample_id`` field, so the strict
    :func:`xssharden.dataset.io.read_records` validator rejects them.
    This helper reads JSONL (or JSON array) files and performs basic
    structural checks without enforcing the full dataset schema.
    """
    from pathlib import Path

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Variant file not found: {resolved}")
    suffix = resolved.suffix.lower()
    records: list[dict] = []
    if suffix in (".jsonl",):
        with resolved.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: invalid JSON on line {lineno}: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise TypeError(
                        f"{path}: line {lineno} must be a JSON object, "
                        f"got {type(obj).__name__}"
                    )
                records.append(obj)
    elif suffix in (".json",):
        with resolved.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    raise TypeError(
                        f"{path}: record at index {idx} must be a JSON object, "
                        f"got {type(item).__name__}"
                    )
                records.append(item)
        else:
            raise ValueError(f"{path}: JSON root must be an array, got {type(data).__name__}")
    else:
        raise ValueError(
            f"Unsupported variant format '{suffix}': expected '.jsonl' or '.json'"
        )
    for idx, rec in enumerate(records):
        if "payload" not in rec:
            raise ValueError(f"{path}: record at index {idx}: missing 'payload'")
        if not isinstance(rec["payload"], str) or not rec["payload"].strip():
            raise ValueError(f"{path}: record at index {idx}: 'payload' must be a non-empty string")
    return records


def _cmd_dataset_build(args: argparse.Namespace) -> int:
    """Read, deduplicate, leakage-safe split, and write a dataset."""
    from xssharden.dataset.clean import deduplicate_records
    from xssharden.dataset.io import read_records, write_records
    from xssharden.dataset.split import split_records

    records = read_records(args.input)
    deduped = deduplicate_records(records)
    built = split_records(deduped, seed=args.seed)
    write_records(built, args.output)
    removed = len(records) - len(deduped)
    print(
        f"Wrote {len(built)} records "
        f"({len(records)} read, {removed} duplicates removed) "
        f"to {args.output}"
    )
    return 0


def _cmd_dataset_prepare_labeled(args: argparse.Namespace) -> int:
    """Prepare a generic labeled payload file into project-schema JSONL/CSV."""
    from xssharden.dataset.prepare_labeled_payloads import (
        prepare_labeled_payloads,
    )

    summary = prepare_labeled_payloads(
        args.input, args.output, args.source_name, seed=args.seed
    )
    print(
        f"Prepared {summary['written']} records to {summary['output']}: "
        f"{summary['total_rows']} read, "
        f"kept benign={summary['kept_benign']} xss={summary['kept_xss']} "
        f"(total kept={summary['kept_total']}, "
        f"duplicates removed={summary['duplicates_removed']}), "
        f"splits={summary['splits']}, "
        f"source={summary['source']}, seed={summary['seed']}"
    )
    return 0


def _cmd_dataset_prepare_kaggle_xss(args: argparse.Namespace) -> int:
    """Prepare the Kaggle XSS_dataset.csv into project-schema JSONL/CSV."""
    from xssharden.dataset.prepare_kaggle_xss import prepare_kaggle_xss

    summary = prepare_kaggle_xss(args.input, args.output, seed=args.seed)
    print(
        f"Prepared {summary['written']} records to {summary['output']}: "
        f"{summary['total_rows']} read, "
        f"kept benign={summary['kept_benign']} xss={summary['kept_xss']} "
        f"(total kept={summary['kept_total']}, "
        f"blank skipped={summary['skipped_blank']}, "
        f"duplicates removed={summary['duplicates_removed']}), "
        f"splits={summary['splits']}, "
        f"source={summary['source']}, seed={summary['seed']}"
    )
    return 0


def _cmd_dataset_merge(args: argparse.Namespace) -> int:
    """Merge prepared project-schema files into one split corpus."""
    from xssharden.dataset.merge import merge_prepared_datasets

    try:
        summary = merge_prepared_datasets(
            args.inputs, args.output, seed=args.seed
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Merged {summary['written']} records to {summary['output']}: "
        f"{summary['total_rows']} read from {len(summary['inputs'])} inputs "
        f"(per-input={summary['per_input_counts']}, "
        f"duplicates removed={summary['duplicates_removed']}), "
        f"sources={summary['sources']}, "
        f"splits={summary['splits']}, seed={summary['seed']}"
    )
    return 0


def _cmd_dataset_prepare(args: argparse.Namespace) -> int:
    """Prepare the http_params raw CSV into project-schema JSONL."""
    from xssharden.dataset.prepare_http_params import prepare_http_params

    summary = prepare_http_params(args.input, args.output, seed=args.seed)
    print(
        f"Prepared {summary['written']} records to {summary['output']}: "
        f"{summary['total_rows']} read, "
        f"kept norm={summary['kept_norm']} xss={summary['kept_xss']} "
        f"(total kept={summary['kept_total']}, "
        f"duplicates removed={summary['duplicates_removed']}), "
        f"excluded={summary['excluded_total']} {summary['excluded_by_type']}, "
        f"length mismatches={summary['length_mismatches']}, "
        f"splits={summary['splits']}, seed={summary['seed']}"
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate one payload against the controlled local sandbox."""
    import json

    from xssharden.validation.validator import (
        PACKAGED_FIXTURE_PATH,
        validate_payload,
    )

    if args.sandbox_url is not None and args.fixture is not None:
        print(
            "error: pass exactly one of --fixture or --sandbox-url, not both",
            file=sys.stderr,
        )
        return 2
    fixture = args.fixture
    sandbox_url = args.sandbox_url
    if sandbox_url is None and fixture is None:
        fixture = str(PACKAGED_FIXTURE_PATH)
    try:
        result = validate_payload(
            args.payload,
            sandbox_url=sandbox_url,
            fixture_path=fixture,
            context_target=args.context_target,
            probe=args.probe,
            timeout_ms=args.timeout_ms,
        )
    except RuntimeError as exc:  # e.g. Playwright not installed
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (ValueError, TypeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_validate_variants(args: argparse.Namespace) -> int:
    """Validate variant records from JSONL against the local sandbox."""
    import json

    from xssharden.validation.validator import PACKAGED_FIXTURE_PATH
    from xssharden.validation.variant_runner import validate_variants

    if args.sandbox_url is not None and args.fixture is not None:
        print(
            "error: pass exactly one of --fixture or --sandbox-url, not both",
            file=sys.stderr,
        )
        return 2
    fixture = args.fixture
    sandbox_url = args.sandbox_url
    if sandbox_url is None and fixture is None:
        fixture = str(PACKAGED_FIXTURE_PATH)
    try:
        with open(args.input, "r", encoding="utf-8") as handle:
            records = []
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"error: {args.input}: invalid JSON on line {lineno}: {exc}",
                        file=sys.stderr,
                    )
                    return 2
                if not isinstance(obj, dict):
                    print(
                        f"error: {args.input}: line {lineno} must be a JSON object",
                        file=sys.stderr,
                    )
                    return 2
                records.append(obj)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        results = validate_variants(
            records,
            sandbox_url=sandbox_url,
            fixture_path=fixture,
            context_target=args.context_target,
            probe=args.probe,
            timeout_ms=args.timeout_ms,
            limit=args.limit,
        )
    except RuntimeError as exc:  # e.g. Playwright not installed
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (ValueError, TypeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    with open(args.output, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n")
    n_valid = sum(1 for result in results if result.get("valid") is True)
    print(f"Wrote {len(results)} results ({n_valid} valid) to {args.output}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate deterministic programmatic variants from seed records."""
    from xssharden.dataset.io import read_records
    from xssharden.generation.programmatic import SUPPORTED_CATEGORIES, generate_variants

    categories = None
    if args.categories is not None:
        categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    try:
        seeds = read_records(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        variants = generate_variants(
            seeds,
            categories=categories,
            seed=args.seed,
            max_per_seed=args.max_per_seed,
            max_per_category=args.max_per_category,
            enforce_split_boundary=not args.allow_non_train_seeds,
        )
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        _write_variant_records(args.output, variants)
    except ValueError as exc:
        print(
            f"error: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        f"Wrote {len(variants)} variants "
        f"({len(seeds)} seeds, categories={list(categories) if categories else list(SUPPORTED_CATEGORIES)}) "
        f"to {args.output}"
    )
    return 0


def _cmd_generate_llm(args: argparse.Namespace) -> int:
    """Generate cloud-LLM variants from seed records (explicit opt-in)."""
    import os

    from xssharden.dataset.io import read_records
    from xssharden.generation.llm import (
        ENV_API_KEY,
        ENV_BASE_URL,
        ENV_MODEL,
        ENV_ORGANIZATION,
        ENV_PROJECT,
        ENV_TIMEOUT_S,
        LLMConfig,
        LLMConfigurationError,
        OpenAICompatibleClient,
        generate_llm_variants,
    )

    if not args.allow_payload_submission:
        print(
            "error: refusing to submit payload text to a cloud API without "
            "explicit opt-in: re-run with --allow-payload-submission",
            file=sys.stderr,
        )
        return 2
    categories = None
    if args.categories is not None:
        categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    raw_timeout = args.timeout_s
    if raw_timeout is None:
        raw_timeout = os.environ.get(ENV_TIMEOUT_S, "").strip() or None
    timeout_s = 30.0
    if raw_timeout is not None:
        try:
            timeout_s = float(raw_timeout)
        except (TypeError, ValueError):
            print(
                f"error: --timeout-s must be a positive number, got {raw_timeout!r}",
                file=sys.stderr,
            )
            return 2
    try:
        config = LLMConfig(
            base_url=args.base_url or os.environ.get(ENV_BASE_URL, ""),
            model=args.model or os.environ.get(ENV_MODEL, ""),
            api_key=args.api_key or os.environ.get(ENV_API_KEY, ""),
            timeout_s=timeout_s,
            organization=args.organization or os.environ.get(ENV_ORGANIZATION) or None,
            project=args.project or os.environ.get(ENV_PROJECT) or None,
        )
    except LLMConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        seeds = read_records(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        result = generate_llm_variants(
            seeds,
            client=OpenAICompatibleClient(config),
            allow_payload_submission=True,
            categories=categories,
            variants_per_seed=args.variants_per_seed,
            max_per_seed=args.max_per_seed,
            enforce_split_boundary=not args.allow_non_train_seeds,
        )
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        _write_variant_records(args.output, result.variants)
    except ValueError as exc:
        print(
            f"error: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        f"Wrote {len(result.variants)} variants "
        f"({len(seeds)} seeds, {len(result.errors)} rejected responses) "
        f"to {args.output}"
    )
    for error in result.errors:
        print(
            f"rejected seed {error.get('seed_id')}: {error.get('reason')}",
            file=sys.stderr,
        )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Fit a detector on a prepared dataset and save it to disk."""
    from xssharden.dataset.io import read_records

    try:
        records = read_records(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    train_records = [r for r in records if r.get("split") == "train"]
    validation_records = [r for r in records if r.get("split") == "validation"]

    if not train_records:
        print("error: no split='train' records found in the dataset", file=sys.stderr)
        return 2

    if args.model == "lr":
        from xssharden.detectors.baseline_lr import BaselineLRDetector

        detector = BaselineLRDetector(random_state=args.seed)
    elif args.model == "xgb":
        from xssharden.detectors.xgboost_detector import XGBoostDetector

        detector = XGBoostDetector(random_state=args.seed)
    else:
        print(
            f"error: unsupported model {args.model!r}; expected 'lr' or 'xgb'",
            file=sys.stderr,
        )
        return 2

    try:
        detector.fit(train_records)
    except (ValueError, TypeError) as exc:
        print(f"error: fit failed: {exc}", file=sys.stderr)
        return 2

    if validation_records:
        try:
            threshold = detector.calibrate_threshold(
                validation_records, target_fpr=args.target_fpr
            )
            print(f"Threshold calibrated to {threshold:.6f} (target FPR={args.target_fpr})")
        except (ValueError, TypeError) as exc:
            print(f"warning: threshold calibration skipped: {exc}", file=sys.stderr)

    try:
        detector.save(args.output)
    except (OSError, TypeError) as exc:
        print(f"error: save failed: {exc}", file=sys.stderr)
        return 2

    desc = detector.describe()
    print(f"Trained {desc['detector']} detector on {len(train_records)} records")
    print(f"  threshold: {desc['threshold']:.6f}")
    print(f"  saved to: {args.output}")
    return 0


def _cmd_attack(args: argparse.Namespace) -> int:
    """Score variant records against a saved detector and report evasion."""
    from xssharden.attack.evasion import evaluate_variants
    from xssharden.detectors import load_detector

    try:
        detector = load_detector(args.detector)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except TypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        variants = _read_variant_records(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        evaluation = evaluate_variants(
            variants,
            detector,
            allowed_splits=args.allowed_splits,
        )
    except (ValueError, TypeError) as exc:
        print(f"error: evaluation failed: {exc}", file=sys.stderr)
        return 2

    with open(args.output, "w", encoding="utf-8") as handle:
        for record in evaluation.records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    m = evaluation.metrics
    print(f"Scored {m.total_records} records ({m.valid_records} valid) "
          f"against {getattr(detector, 'describe', lambda: {})().get('detector', 'detector')}")
    print(f"  detector threshold:  {m.detector_threshold:.6f}")
    print(f"  evasion rate:        {m.evasion_count}/{m.malicious_count} = {m.evasion_rate:.4f}")
    print(f"  V-ASR:               {m.valid_evasion_count}/{m.valid_malicious_count} = {m.valid_malicious_evasion_rate:.4f}")
    print(f"  raw evasion rate:    {m.evasion_count}/{m.total_records} = {m.raw_evasion_rate:.4f}")
    print(f"Wrote scored records to {args.output}")
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    """Select a budgeted subset of scored variant records for hardening."""
    from xssharden.dataset.io import read_records
    from xssharden.selection.selection import select_variants

    try:
        variants = _read_variant_records(args.input)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    training_records = None
    if args.training_data is not None:
        try:
            training_records = read_records(args.training_data)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except (ValueError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        result = select_variants(
            variants,
            budget=args.budget,
            strategy=args.strategy,
            seed=args.seed,
            max_per_category=args.max_per_category,
            training_records=training_records,
        )
    except (ValueError, TypeError) as exc:
        print(f"error: selection failed: {exc}", file=sys.stderr)
        return 2

    with open(args.output, "w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"Selected {result.selected_count}/{result.eligible_count} eligible records "
          f"(strategy={result.strategy}, budget={result.requested_budget}, seed={result.seed})")
    if result.shortfall:
        print(f"  WARNING: shortfall — fewer eligible records than requested budget")
    print(f"  filtered: {result.filtered_counts}")
    print(f"  per-category: {result.per_category_counts}")
    print(f"Wrote selected records to {args.output}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline from a YAML config file."""
    import yaml

    from xssharden.dataset.io import read_records, write_records
    from xssharden.generation.programmatic import generate_variants

    try:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except yaml.YAMLError as exc:
        print(f"error: {args.config}: invalid YAML: {exc}", file=sys.stderr)
        return 2

    if not isinstance(config, dict):
        print("error: config must be a YAML mapping", file=sys.stderr)
        return 2

    seed = config.get("seed", 42)
    model_type = config.get("model", "lr")
    budget = config.get("budget", 500)
    target_fpr = config.get("target_fpr", 0.01)
    max_per_seed = config.get("max_per_seed", 10)
    max_per_category = config.get("max_per_category", None)

    dataset_path = config.get("dataset_path")
    if not dataset_path:
        print("error: config must specify 'dataset_path'", file=sys.stderr)
        return 2

    # Resolve dataset_path relative to the config file's parent directory.
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.normpath(os.path.join(config_dir, dataset_path))

    variants_dir = config.get("variants_dir", "variants")
    models_dir = config.get("models_dir", "models")
    output_dir = config.get("output_dir", "experiments")
    reports_dir = config.get("reports_dir", "reports")

    # Resolve relative paths against the config file's parent directory
    # so that different configs (e.g. in test tmp_paths) don't collide
    # with project-root relative paths.
    config_dir = os.path.dirname(os.path.abspath(args.config))

    def _resolve(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(config_dir, path))

    variants_dir = _resolve(variants_dir)
    models_dir = _resolve(models_dir)
    output_dir = _resolve(output_dir)
    reports_dir = _resolve(reports_dir)

    os.makedirs(variants_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    generated_path = os.path.join(variants_dir, "generated_programmatic.jsonl")
    validated_path = os.path.join(variants_dir, "validated.jsonl")
    scored_path = os.path.join(variants_dir, "scored.jsonl")
    selected_path = os.path.join(variants_dir, "selected.jsonl")
    detector_path = os.path.join(models_dir, f"baseline_{model_type}.pkl")
    evaluation_path = os.path.join(output_dir, "evaluation.json")
    report_path = os.path.join(reports_dir, "robustness_report.md")

    # -- Stage 1: Generate variants --
    if args.force or not os.path.exists(generated_path):
        print("=== Stage 1: Generate variants ===")
        try:
            seeds = read_records(dataset_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"error: dataset load failed: {exc}", file=sys.stderr)
            return 2

        train_seeds = [r for r in seeds if r.get("split") == "train" and int(r.get("label", 0)) == 1]
        if not train_seeds:
            print("error: no malicious train seeds found for generation", file=sys.stderr)
            return 2

        try:
            variants = generate_variants(
                train_seeds,
                seed=seed,
                max_per_seed=max_per_seed,
                max_per_category=max_per_category,
            )
        except (ValueError, TypeError) as exc:
            print(f"error: generation failed: {exc}", file=sys.stderr)
            return 2

        _write_variant_records(generated_path, variants)
        print(f"Generated {len(variants)} variants to {generated_path}")
    else:
        print(f"=== Stage 1: Generate variants === (skipped, exists)")

    # -- Stage 2: Validate variants --
    if args.force or not os.path.exists(validated_path):
        print("=== Stage 2: Validate variants ===")
        from xssharden.validation.validator import PACKAGED_FIXTURE_PATH
        from xssharden.validation.variant_runner import validate_variants

        try:
            generated = _read_variant_records(generated_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"error: could not read generated variants: {exc}", file=sys.stderr)
            return 2

        try:
            validated = validate_variants(
                generated,
                fixture_path=str(PACKAGED_FIXTURE_PATH),
                limit=config.get("validate_limit", None),
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"error: validation failed: {exc}", file=sys.stderr)
            return 2

        n_valid = sum(1 for r in validated if r.get("valid") is True)
        with open(validated_path, "w", encoding="utf-8") as handle:
            for record in validated:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        print(f"Validated {len(validated)} variants ({n_valid} valid) to {validated_path}")
    else:
        print(f"=== Stage 2: Validate variants === (skipped, exists)")

    # -- Stage 3: Train baseline detector --
    if args.force or not os.path.exists(detector_path):
        print("=== Stage 3: Train baseline detector ===")
        try:
            all_records = read_records(dataset_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"error: dataset load failed: {exc}", file=sys.stderr)
            return 2

        train_records = [r for r in all_records if r.get("split") == "train"]
        validation_records = [r for r in all_records if r.get("split") == "validation"]

        if not train_records:
            print("error: no train records found", file=sys.stderr)
            return 2

        if model_type == "lr":
            from xssharden.detectors.baseline_lr import BaselineLRDetector
            detector = BaselineLRDetector(random_state=seed)
        elif model_type == "xgb":
            from xssharden.detectors.xgboost_detector import XGBoostDetector
            detector = XGBoostDetector(random_state=seed)
        else:
            print(f"error: unsupported model {model_type!r}", file=sys.stderr)
            return 2

        try:
            detector.fit(train_records)
        except (ValueError, TypeError) as exc:
            print(f"error: fit failed: {exc}", file=sys.stderr)
            return 2

        if validation_records:
            try:
                detector.calibrate_threshold(validation_records, target_fpr=target_fpr)
            except (ValueError, TypeError) as exc:
                print(f"warning: threshold calibration skipped: {exc}", file=sys.stderr)

        detector.save(detector_path)
        print(f"Trained {model_type} detector, saved to {detector_path}")
    else:
        print(f"=== Stage 3: Train baseline detector === (skipped, exists)")

    # -- Stage 4: Attack / score variants --
    if args.force or not os.path.exists(scored_path):
        print("=== Stage 4: Score variants against detector ===")
        from xssharden.attack.evasion import evaluate_variants

        try:
            detector = __import__("xssharden.detectors", fromlist=["load_detector"]).load_detector(detector_path)
        except (FileNotFoundError, TypeError) as exc:
            print(f"error: could not load detector: {exc}", file=sys.stderr)
            return 2

        try:
            validated_for_attack = _read_variant_records(validated_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"error: could not read validated variants: {exc}", file=sys.stderr)
            return 2

        adv_dev_records = [r for r in validated_for_attack if r.get("split") == "adv_dev"]
        if not adv_dev_records:
            adv_dev_records = validated_for_attack

        try:
            evaluation = evaluate_variants(
                adv_dev_records,
                detector,
                allowed_splits=["adv_dev"],
            )
        except (ValueError, TypeError) as exc:
            print(f"error: evaluation failed: {exc}", file=sys.stderr)
            return 2

        with open(scored_path, "w", encoding="utf-8") as handle:
            for record in evaluation.records:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        m = evaluation.metrics
        print(f"Scored {m.total_records} variants: V-ASR={m.valid_malicious_evasion_rate:.4f}, "
              f"evasion_rate={m.evasion_rate:.4f}")
        print(f"Wrote scored records to {scored_path}")
    else:
        print(f"=== Stage 4: Score variants === (skipped, exists)")

    # -- Stage 5: Four-arm hardening --
    if args.force or not os.path.exists(evaluation_path):
        print("=== Stage 5: Four-arm hardening ===")
        from xssharden.hardening.arms import run_hardening_arms

        try:
            all_records = read_records(dataset_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"error: dataset load failed: {exc}", file=sys.stderr)
            return 2

        train_records = [r for r in all_records if r.get("split") == "train"]
        validation_records = [r for r in all_records if r.get("split") == "validation"]

        try:
            validated_variants = _read_variant_records(validated_path)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"error: could not read validated variants: {exc}", file=sys.stderr)
            return 2

        if model_type == "lr":
            from xssharden.detectors.baseline_lr import BaselineLRDetector
            def factory(): return BaselineLRDetector(random_state=seed)
        elif model_type == "xgb":
            from xssharden.detectors.xgboost_detector import XGBoostDetector
            def factory(): return XGBoostDetector(random_state=seed)
        else:
            print(f"error: unsupported model {model_type!r}", file=sys.stderr)
            return 2

        try:
            hardening_result = run_hardening_arms(
                train_records=train_records,
                validated_variants=validated_variants,
                detector_factory=factory,
                budget=budget,
                validation_records=validation_records if validation_records else None,
                seed=seed,
                max_per_category=max_per_category,
            )
        except (ValueError, TypeError) as exc:
            print(f"error: hardening failed: {exc}", file=sys.stderr)
            return 2

        for name, added in hardening_result.added_counts.items():
            print(f"  {name}: +{added} augmented records")
        print(f"Hardening complete")
    else:
        print(f"=== Stage 5: Four-arm hardening === (skipped, exists)")

    # -- Stage 6: Report --
    if args.force or not os.path.exists(report_path):
        print("=== Stage 6: Generate report ===")
        from xssharden.reporting import write_report

        try:
            hardening_result_dict = hardening_result.to_dict()
        except NameError:
            print("error: hardening result not available (run with --force)", file=sys.stderr)
            return 2

        metadata = {
            "model": model_type,
            "seed": seed,
            "budget": budget,
            "target_fpr": target_fpr,
        }
        try:
            report_summary = write_report(hardening_result_dict, report_path, metadata=metadata)
        except (ValueError, TypeError) as exc:
            print(f"error: report generation failed: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote {report_summary['format']} report to {report_path}")
    else:
        print(f"=== Stage 6: Generate report === (skipped, exists)")

    print("\nPipeline complete.")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Render a robustness report from a JSON evaluation file."""
    from xssharden.reporting import write_report

    try:
        with open(args.input, "r", encoding="utf-8") as handle:
            evaluation = json.load(handle)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: {args.input}: invalid JSON: {exc}", file=sys.stderr)
        return 2
    metadata = None
    if args.metadata is not None:
        try:
            with open(args.metadata, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except json.JSONDecodeError as exc:
            print(
                f"error: {args.metadata}: invalid JSON: {exc}",
                file=sys.stderr,
            )
            return 2
    try:
        summary = write_report(evaluation, args.output, metadata=metadata)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {summary['format']} report to {summary['output']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="xssharden",
        description="XSSHarden — XSS variant generation and detector hardening research pipeline.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command")

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Dataset processing commands.",
    )
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")

    build_parser_ = dataset_subparsers.add_parser(
        "build",
        help="Deduplicate and leakage-safe split a raw dataset file.",
        description=(
            "Read a JSONL/CSV dataset, remove exact/normalized duplicates, "
            "assign leakage-safe train/validation/clean-test splits, and "
            "write the result. Never executes payloads."
        ),
    )
    build_parser_.add_argument(
        "--input",
        required=True,
        help="Input JSONL (.jsonl/.json) or CSV (.csv) dataset file.",
    )
    build_parser_.add_argument(
        "--output",
        required=True,
        help="Output JSONL or CSV file for the built dataset.",
    )
    build_parser_.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for split assignment (default: 42).",
    )
    build_parser_.set_defaults(func=_cmd_dataset_build)

    prepare_parser = dataset_subparsers.add_parser(
        "prepare",
        help="Prepare the http_params raw CSV (norm+xss only) into project-schema JSONL.",
        description=(
            "Read the raw http_params payload_full.csv as text without "
            "executing payloads, keep only attack_type norm and xss, map "
            "them to the project schema, deduplicate, leakage-safe split, "
            "and write JSONL. Non-XSS anomaly types are excluded and counted."
        ),
    )
    prepare_parser.add_argument(
        "--input",
        required=True,
        help="Input raw http_params CSV file (payload,length,attack_type,label).",
    )
    prepare_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for the prepared project-schema dataset.",
    )
    prepare_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for split assignment (default: 42).",
    )
    prepare_parser.set_defaults(func=_cmd_dataset_prepare)

    labeled_parser = dataset_subparsers.add_parser(
        "prepare-labeled",
        help="Prepare a generic labeled payload file (payload,label) into project-schema records.",
        description=(
            "Read a generic CSV (.csv) or JSONL (.jsonl/.json) file with "
            "payload and label columns/fields as text without executing "
            "payloads, map labels 0/1 or benign/xss (case-insensitive) to "
            "the project schema, deduplicate with the shared cleaning, "
            "leakage-safe split, and write the result. Missing/blank "
            "payloads, unsupported labels, and single-class inputs are "
            "rejected; source_name 'http_params_dataset' is reserved."
        ),
    )
    labeled_parser.add_argument(
        "--input",
        required=True,
        help="Input raw CSV (.csv) or JSONL (.jsonl/.json) file with payload,label fields.",
    )
    labeled_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL or CSV file for the prepared project-schema dataset.",
    )
    labeled_parser.add_argument(
        "--source-name",
        "--source",
        dest="source_name",
        required=True,
        help="Provenance source name (stored in 'source', embedded in sample IDs; 'http_params_dataset' is reserved).",
    )
    labeled_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for split assignment (default: 42).",
    )
    labeled_parser.set_defaults(func=_cmd_dataset_prepare_labeled)

    kaggle_parser = dataset_subparsers.add_parser(
        "prepare-kaggle-xss",
        help="Prepare the Kaggle XSS_dataset.csv (Sentence,Label) into project-schema records.",
        description=(
            "Read the raw Kaggle XSS_dataset.csv as text without executing "
            "payloads, map Sentence to payload and Label 0/1 to benign/xss, "
            "skip (and count) blank payloads, reject unsupported labels and "
            "missing Sentence/Label columns, preserve source "
            "kaggle_xss_dataset plus raw_label/raw_row_number/raw_source_id "
            "provenance with context_target=unknown and "
            "license_status=unknown, deduplicate with the shared cleaning, "
            "leakage-safe split, and write the result."
        ),
    )
    kaggle_parser.add_argument(
        "--input",
        required=True,
        help="Input raw Kaggle CSV file (index,Sentence,Label).",
    )
    kaggle_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL or CSV file for the prepared project-schema dataset.",
    )
    kaggle_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for split assignment (default: 42).",
    )
    kaggle_parser.set_defaults(func=_cmd_dataset_prepare_kaggle_xss)

    merge_parser = dataset_subparsers.add_parser(
        "merge",
        help="Merge prepared project-schema files (requires 3+ sources).",
        description=(
            "Read prepared project-schema files (JSONL .jsonl/.json or "
            "CSV .csv) as inert text without executing payloads, require "
            "at least 3 distinct source values, deduplicate with the "
            "shared cleaning, assign leakage-safe group-atomic splits, "
            "and write the merged result. Merges with fewer than 3 "
            "sources, duplicate input paths, or invalid inputs are "
            "refused before anything is written."
        ),
    )
    merge_parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input prepared file (repeatable: pass once per file).",
    )
    merge_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL or CSV file for the merged dataset.",
    )
    merge_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for split assignment (default: 42).",
    )
    merge_parser.set_defaults(func=_cmd_dataset_merge)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate one payload in the controlled local sandbox.",
        description=(
            "Validate a single payload against a local sandbox URL "
            "(http(s)://localhost|127.0.0.1|::1) or a local HTML fixture "
            "file. Non-local URLs are rejected before any payload is "
            "sent. Requires Playwright + Chromium unless this entry "
            "point is exercised with an injected test runner."
        ),
    )
    validate_parser.add_argument(
        "--payload",
        required=True,
        help="Payload string to validate (treated as inert text until the local browser probe).",
    )
    validate_parser.add_argument(
        "--fixture",
        default=None,
        help="Local HTML fixture file (default: packaged sandbox.html).",
    )
    validate_parser.add_argument(
        "--sandbox-url",
        default=None,
        help="Local sandbox URL (http(s)://localhost|127.0.0.1|::1 only).",
    )
    validate_parser.add_argument(
        "--probe",
        default="alert",
        help="Probe name to observe (default: alert).",
    )
    validate_parser.add_argument(
        "--context-target",
        default="reflected_html",
        help="Injection context under test (default: reflected_html).",
    )
    validate_parser.add_argument(
        "--timeout-ms",
        type=float,
        default=3000,
        help="Probe timeout in milliseconds (default: 3000).",
    )
    validate_parser.set_defaults(func=_cmd_validate)

    batch_parser = subparsers.add_parser(
        "validate-variants",
        help="Validate variant records from JSONL in the controlled local sandbox.",
        description=(
            "Read JSONL variant records (one JSON object per line, each with "
            "a payload plus provenance such as sample_id, seed_id, "
            "mutation_category, and source), validate each payload against a "
            "local sandbox URL (http(s)://localhost|127.0.0.1|::1) or a local "
            "HTML fixture file, and write one JSON result per input in order. "
            "Non-local URLs are rejected before any payload is sent. Requires "
            "Playwright + Chromium for real browser runs."
        ),
    )
    batch_parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with one variant record per line.",
    )
    batch_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for one validation result per input.",
    )
    batch_parser.add_argument(
        "--fixture",
        default=None,
        help="Local HTML fixture file (default: packaged sandbox.html).",
    )
    batch_parser.add_argument(
        "--sandbox-url",
        default=None,
        help="Local sandbox URL (http(s)://localhost|127.0.0.1|::1 only).",
    )
    batch_parser.add_argument(
        "--probe",
        default="alert",
        help="Default probe name to observe (default: alert; per-record probe overrides it).",
    )
    batch_parser.add_argument(
        "--context-target",
        default="reflected_html",
        help="Default injection context (default: reflected_html; per-record context_target/context overrides it).",
    )
    batch_parser.add_argument(
        "--timeout-ms",
        type=float,
        default=3000,
        help="Probe timeout in milliseconds (default: 3000).",
    )
    batch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Validate only the first N records (default: all).",
    )
    batch_parser.set_defaults(func=_cmd_validate_variants)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate deterministic programmatic variants from seed records.",
        description=(
            "Read dataset-schema seed records (JSONL/CSV, payloads treated "
            "as inert text and never executed), apply deterministic "
            "programmatic mutations (encoding, whitespace_comment, "
            "tag_event_substitution, case_variation), and write one "
            "variant record per mutation as JSONL with provenance "
            "(variant_id, seed_id, mutation_category, source, generator). "
            "Seeds from held-out splits (validation/clean-test/test) are "
            "rejected unless --allow-non-train-seeds is passed."
        ),
    )
    generate_parser.add_argument(
        "--input",
        required=True,
        help="Input seed file (JSONL .jsonl/.json or CSV .csv, project schema).",
    )
    generate_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for the generated variant records.",
    )
    generate_parser.add_argument(
        "--categories",
        default=None,
        help=(
            "Comma-separated mutation categories "
            "(default: encoding,whitespace_comment,tag_event_substitution,case_variation)."
        ),
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for capped sampling (default: 42).",
    )
    generate_parser.add_argument(
        "--max-per-seed",
        type=int,
        default=None,
        help="Maximum variants kept per seed (default: all).",
    )
    generate_parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Maximum variants kept per mutation category (default: all).",
    )
    generate_parser.add_argument(
        "--allow-non-train-seeds",
        action="store_true",
        help="Allow seeds from non-train/adversarial-dev splits (default: reject them).",
    )
    generate_parser.set_defaults(func=_cmd_generate)

    llm_parser = subparsers.add_parser(
        "generate-llm",
        help="Generate cloud-LLM variants from seed records (explicit opt-in).",
        description=(
            "Read dataset-schema seed records (JSONL/CSV, payloads treated "
            "as inert text and never executed), request rewrites from an "
            "OpenAI-compatible cloud chat endpoint, strictly validate the "
            "JSON response contract, and write accepted variant records as "
            "JSONL with provenance (variant_id, seed_id, mutation_category, "
            "source, generator, model). Seed payload text is submitted to "
            "the cloud API only when --allow-payload-submission is passed; "
            "credentials come from explicit flags or the XSSHARDEN_LLM_* "
            "environment variables and are never hardcoded. Rejected model "
            "responses are counted and reported, never silently accepted. "
            "Seeds from held-out splits (validation/clean-test/test) are "
            "rejected unless --allow-non-train-seeds is passed. Never "
            "launches a browser or executes payloads."
        ),
    )
    llm_parser.add_argument(
        "--input",
        required=True,
        help="Input seed file (JSONL .jsonl/.json or CSV .csv, project schema).",
    )
    llm_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for the accepted variant records.",
    )
    llm_parser.add_argument(
        "--base-url",
        default=None,
        help="API root URL (default: $XSSHARDEN_LLM_BASE_URL).",
    )
    llm_parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (default: $XSSHARDEN_LLM_MODEL).",
    )
    llm_parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: $XSSHARDEN_LLM_API_KEY; never hardcoded).",
    )
    llm_parser.add_argument(
        "--organization",
        default=None,
        help="Optional organization header (default: $XSSHARDEN_LLM_ORGANIZATION).",
    )
    llm_parser.add_argument(
        "--project",
        default=None,
        help="Optional project header (default: $XSSHARDEN_LLM_PROJECT).",
    )
    llm_parser.add_argument(
        "--timeout-s",
        default=None,
        help="Per-request timeout in seconds (default: $XSSHARDEN_LLM_TIMEOUT_S or 30).",
    )
    llm_parser.add_argument(
        "--categories",
        default=None,
        help=(
            "Comma-separated mutation categories the model may use "
            "(default: encoding,whitespace_comment,tag_event_substitution,case_variation)."
        ),
    )
    llm_parser.add_argument(
        "--variants-per-seed",
        type=int,
        default=3,
        help="Variants requested from the model per seed (default: 3).",
    )
    llm_parser.add_argument(
        "--max-per-seed",
        type=int,
        default=None,
        help="Maximum accepted variants kept per seed (default: all).",
    )
    llm_parser.add_argument(
        "--allow-payload-submission",
        action="store_true",
        help="Explicit opt-in to submit seed payload text to the cloud API (required).",
    )
    llm_parser.add_argument(
        "--allow-non-train-seeds",
        action="store_true",
        help="Allow seeds from non-train/adversarial-dev splits (default: reject them).",
    )
    llm_parser.set_defaults(func=_cmd_generate_llm)

    report_parser = subparsers.add_parser(
        "report",
        help="Render a deterministic robustness report from an evaluation JSON file.",
        description=(
            "Read a HardeningEvaluation JSON file (the to_dict() form with "
            "per-arm clean-test and adversarial metrics), validate its shape, "
            "and write a deterministic Markdown (.md) or HTML (.html) "
            "robustness report with explicit denominators, methodology and "
            "safety notes, and the two-source caveat. Metrics are serialized "
            "as given without inventing results; raw payloads are never "
            "included. Payloads stay inert strings and no browser, network, "
            "or validator code is touched."
        ),
    )
    report_parser.add_argument(
        "--input",
        required=True,
        help="Input evaluation JSON file (HardeningEvaluation to_dict() form).",
    )
    report_parser.add_argument(
        "--output",
        required=True,
        help="Output report file ending in .md or .html.",
    )
    report_parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON file with a flat object of experiment metadata.",
    )
    report_parser.set_defaults(func=_cmd_report)

    # -- train command --
    train_parser = subparsers.add_parser(
        "train",
        help="Fit a detector on a prepared dataset and save it to disk.",
        description=(
            "Read a prepared project-schema dataset file, fit a requested "
            "detector (lr or xgb) on split='train' records, optionally "
            "calibrate the decision threshold on split='validation' records "
            "at a configurable target FPR, and save the fitted detector to "
            "disk. Payloads are treated as inert text and never executed."
        ),
    )
    train_parser.add_argument(
        "--input",
        required=True,
        help="Input dataset file (JSONL .jsonl/.json or CSV .csv, project schema).",
    )
    train_parser.add_argument(
        "--output",
        required=True,
        help="Output .pkl file for the fitted detector.",
    )
    train_parser.add_argument(
        "--model",
        choices=["lr", "xgb"],
        default="lr",
        help="Detector architecture: 'lr' for TF-IDF+LogisticRegression (default), 'xgb' for XGBoost.",
    )
    train_parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Target false-positive rate for threshold calibration on validation data (default: 0.01).",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random state for the detector (default: 42).",
    )
    train_parser.set_defaults(func=_cmd_train)

    # -- attack command --
    attack_parser = subparsers.add_parser(
        "attack",
        help="Score variant records against a saved detector and report evasion.",
        description=(
            "Load a previously saved detector from disk, score variant "
            "records from a JSONL file through the detector's predict_proba, "
            "write scored records with detector_score/predicted_label/evaded, "
            "and print evasion metrics (evasion rate, V-ASR, raw evasion "
            "rate). Payloads are treated as inert text and never executed."
        ),
    )
    attack_parser.add_argument(
        "--detector",
        required=True,
        help="Path to a saved detector .pkl file.",
    )
    attack_parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with variant records to score.",
    )
    attack_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for the scored records.",
    )
    attack_parser.add_argument(
        "--allowed-splits",
        nargs="*",
        default=None,
        help="Explicitly allowed split values for scoring (default: auto-detect from records).",
    )
    attack_parser.set_defaults(func=_cmd_attack)

    # -- select command --
    select_parser = subparsers.add_parser(
        "select",
        help="Select a budgeted subset of scored variant records for hardening.",
        description=(
            "Read scored variant records from a JSONL file, apply a "
            "validity-gated selection strategy (impact ranking or random "
            "valid control), enforce per-category caps, and write the "
            "selected subset. Payloads are treated as inert text and never "
            "executed."
        ),
    )
    select_parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with scored variant records.",
    )
    select_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for the selected records.",
    )
    select_parser.add_argument(
        "--strategy",
        choices=["impact", "random_valid"],
        default="impact",
        help="Selection strategy: 'impact' for detector-impact ranking (default), 'random_valid' for budget-matched random control.",
    )
    select_parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Maximum number of records to select (positive integer).",
    )
    select_parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Optional per-category cap (positive integer).",
    )
    select_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for random_valid strategy and tie-breaking (default: 42).",
    )
    select_parser.add_argument(
        "--training-data",
        default=None,
        help="Optional JSONL/CSV file with training records for deduplication.",
    )
    select_parser.set_defaults(func=_cmd_select)

    # -- run command --
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full pipeline from a YAML config file.",
        description=(
            "Accept a YAML config file and run all pipeline stages in "
            "sequence: generate → validate → train → attack → harden → "
            "report. Stages whose output files already exist are skipped "
            "unless --force is passed. All payloads are treated as inert "
            "text; browser validation requires Playwright + Chromium."
        ),
    )
    run_parser.add_argument(
        "config",
        help="Path to a YAML configuration file.",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages even if output files exist.",
    )
    run_parser.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI. Returns 0 on success."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is not None:
        return func(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
