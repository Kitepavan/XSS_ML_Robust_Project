#!/usr/bin/env python3
"""Run the XSSHarden experiment efficiently with a shared browser instance."""

import json
import os
import sys
import time
from pathlib import Path

# -- Configuration --
DATASET = "data/processed/merged_corpus.jsonl"
OUTPUT_DIR = Path("run_output")
VARIANTS_DIR = OUTPUT_DIR / "variants"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

SEED = 42
TARGET_FPR = 0.01
BUDGET = 500
MAX_PER_SEED = 3
MAX_PER_CATEGORY = 1000
LLM_VARIANTS_PER_SEED = 2
LLM_MAX_SEEDS = 25  # 25 seeds × 2 variants = ~50 LLM variants
CUSTOM_VARIANTS_FILE = os.environ.get("XSSHARDEN_CUSTOM_VARIANTS", "")

VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GENERATED_PATH = VARIANTS_DIR / "generated_programmatic.jsonl"
VALIDATED_PATH = VARIANTS_DIR / "validated.jsonl"
SCORED_PATH = VARIANTS_DIR / "scored.jsonl"
SELECTED_PATH = VARIANTS_DIR / "selected.jsonl"
DETECTOR_PATH = MODELS_DIR / "baseline_lr.pkl"
EVALUATION_PATH = OUTPUT_DIR / "experiments" / "evaluation.json"
REPORT_PATH = REPORTS_DIR / "robustness_report.md"


def read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


# ============================================================
# Stage 1: Generate
# ============================================================
print("=" * 60)
print("STAGE 1: Generate programmatic variants")
print("=" * 60)
t0 = time.time()

from xssharden.dataset.io import read_records
from xssharden.generation.programmatic import generate_variants

records = read_records(DATASET)
train_malicious = [r for r in records if r.get("split") == "train" and r.get("label") == 1]
print(f"  Train malicious seeds: {len(train_malicious)}")

variants = generate_variants(
    train_malicious,
    seed=SEED,
    max_per_seed=MAX_PER_SEED,
    max_per_category=MAX_PER_CATEGORY,
)
write_jsonl(variants, GENERATED_PATH)
print(f"  Generated {len(variants)} programmatic variants in {time.time()-t0:.1f}s")
print(f"  Saved to {GENERATED_PATH}")

# ============================================================
# Stage 1b: Generate LLM variants (if API key available)
# ============================================================
LLM_PATH = VARIANTS_DIR / "generated_llm.jsonl"
llm_variants = []

api_key = os.environ.get("XSSHARDEN_LLM_API_KEY", "")
if api_key:
    print()
    print("=" * 60)
    print("STAGE 1b: Generate LLM variants via OpenRouter")
    print("=" * 60)
    t0 = time.time()

    try:
        from xssharden.generation.llm import (
            LLMConfig,
            OpenAICompatibleClient,
            generate_llm_variants,
        )

        config = LLMConfig.from_env()
        client = OpenAICompatibleClient(config)
        print(f"  Model: {config.model}")
        print(f"  Endpoint: {config.base_url}")

        # Use a subset of seeds for LLM generation
        llm_seeds = train_malicious[:LLM_MAX_SEEDS]
        print(f"  Seeds: {len(llm_seeds)} (generating {LLM_VARIANTS_PER_SEED} variants each)")

        result = generate_llm_variants(
            llm_seeds,
            client=client,
            allow_payload_submission=True,
            variants_per_seed=LLM_VARIANTS_PER_SEED,
        )
        llm_variants = result.variants
        write_jsonl(llm_variants, LLM_PATH)
        print(f"  Generated {len(llm_variants)} LLM variants ({len(result.errors)} errors) in {time.time()-t0:.1f}s")
        print(f"  Saved to {LLM_PATH}")
    except Exception as exc:
        print(f"  LLM generation failed: {exc}")
        print(f"  Continuing with programmatic variants only.")
else:
    print()
    print("  Skipping LLM generation (XSSHARDEN_LLM_API_KEY not set)")

# ============================================================
# Stage 1c: Load custom variants (if file provided)
# ============================================================
custom_variants = []

if CUSTOM_VARIANTS_FILE and os.path.exists(CUSTOM_VARIANTS_FILE):
    print()
    print("=" * 60)
    print("STAGE 1c: Load custom variants from file")
    print("=" * 60)
    t0 = time.time()

    with open(CUSTOM_VARIANTS_FILE) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Ensure required fields
                if "payload" not in record:
                    continue
                if "label" not in record:
                    record["label"] = 1
                if "source" not in record:
                    record["source"] = "custom"
                if "attack_category" not in record:
                    record["attack_category"] = "xss"
                if "split" not in record:
                    record["split"] = "adv_dev"
                if "sample_id" not in record:
                    import hashlib
                    h = hashlib.sha256(record["payload"].encode()).hexdigest()[:16]
                    record["sample_id"] = f"custom-{h}"
                if "mutation_category" not in record:
                    record["mutation_category"] = "llm"
                if "generator" not in record:
                    record["generator"] = "custom"
                custom_variants.append(record)

    print(f"  Loaded {len(custom_variants)} custom variants from {CUSTOM_VARIANTS_FILE}")
    print(f"  Time: {time.time()-t0:.1f}s")
elif CUSTOM_VARIANTS_FILE:
    print(f"  WARNING: Custom variants file not found: {CUSTOM_VARIANTS_FILE}")

# Combine all variants
all_variants = variants + llm_variants + custom_variants
print(f"\n  Total variants: {len(all_variants)} "
      f"({len(variants)} programmatic + {len(llm_variants)} LLM + {len(custom_variants)} custom)")

# ============================================================
# Stage 2: Validate (shared browser)
# ============================================================
print()
print("=" * 60)
print("STAGE 2: Validate all variants with Playwright (shared browser)")
print("=" * 60)
t0 = time.time()

from xssharden.validation.validator import PACKAGED_FIXTURE_PATH
from playwright.sync_api import sync_playwright

fixture_uri = PACKAGED_FIXTURE_PATH.resolve().as_uri()
validated = []

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page()

    # Auto-dismiss dialogs (alert/confirm/prompt) so they don't block
    page.on("dialog", lambda d: d.dismiss())

    # Arm the probe hooks via add_init_script — these persist across navigations
    page.add_init_script("""
        window.__probeFired = false;
        window.__probeName = null;
        for (const name of ['alert', 'confirm', 'prompt']) {
          try {
            const original = window[name];
            window[name] = function (...args) {
              window.__probeFired = true;
              window.__probeName = name;
              return name === 'prompt' ? null : undefined;
            };
          } catch (e) {}
        }
        window.__xssProbe = function (name) {
          window.__probeFired = true;
          window.__probeName = name || 'custom';
        };
    """)

    for i, variant in enumerate(all_variants):
        payload = variant["payload"]
        result = dict(variant)

        try:
            # Build reflected XSS URL — the sandbox reads ?q= and injects
            # via innerHTML during DOMContentLoaded. The probe hooks are
            # already armed via add_init_script, so they fire before we
            # check. One navigation per variant, no separate injection step.
            from urllib.parse import quote
            test_url = fixture_uri + "?q=" + quote(payload)
            page.goto(test_url, wait_until="domcontentloaded", timeout=5000)

            # Probe should have fired during page load if payload is valid.
            # Short wait_for_function catches late-firing probes (e.g. onload).
            fired = page.evaluate("() => window.__probeFired === true")
            if not fired:
                try:
                    page.wait_for_function("() => window.__probeFired === true", timeout=50)
                    fired = True
                except Exception:
                    fired = False

            status = "ok" if fired else "timeout"

            result["valid"] = fired
            result["status"] = status
            result["timed_out"] = (status == "timeout")
            result["error"] = None
            result["duration_ms"] = 0.0
            result["timeout_ms"] = 50.0
        except Exception as exc:
            result["valid"] = False
            result["status"] = "error"
            result["timed_out"] = False
            result["error"] = str(exc)[:200]
            result["duration_ms"] = 0.0
            result["timeout_ms"] = 50.0

        validated.append(result)

        if (i + 1) % 200 == 0:
            valid_count = sum(1 for v in validated if v["valid"])
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(all_variants) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(all_variants)}] valid={valid_count} "
                  f"({valid_count/(i+1)*100:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta:.0f}s")

    browser.close()

write_jsonl(validated, VALIDATED_PATH)
n_valid = sum(1 for v in validated if v["valid"])
print(f"  Validated {len(validated)} variants: {n_valid} valid ({n_valid/len(validated)*100:.1f}%)")
print(f"  Time: {time.time()-t0:.1f}s")
print(f"  Saved to {VALIDATED_PATH}")

# ============================================================
# Stage 3: Train baseline detector
# ============================================================
print()
print("=" * 60)
print("STAGE 3: Train baseline LR detector")
print("=" * 60)
t0 = time.time()

from xssharden.detectors.baseline_lr import BaselineLRDetector

train_records = [r for r in records if r.get("split") == "train"]
validation_records = [r for r in records if r.get("split") == "validation"]
print(f"  Train: {len(train_records)} records, Validation: {len(validation_records)} records")

detector = BaselineLRDetector(random_state=SEED)
detector.fit(train_records)
threshold = detector.calibrate_threshold(validation_records, target_fpr=TARGET_FPR)
detector.save(str(DETECTOR_PATH))
print(f"  Threshold: {threshold:.6f} (target FPR={TARGET_FPR})")
print(f"  Saved to {DETECTOR_PATH}")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# Stage 4: Attack / score variants
# ============================================================
print()
print("=" * 60)
print("STAGE 4: Score variants against detector")
print("=" * 60)
t0 = time.time()

from xssharden.attack.evasion import evaluate_variants

adv_dev = [v for v in validated if v.get("split") == "adv_dev"]
if not adv_dev:
    adv_dev = validated

evaluation = evaluate_variants(adv_dev, detector, allowed_splits=["adv_dev"])
write_jsonl(evaluation.records, SCORED_PATH)
m = evaluation.metrics
print(f"  Scored {m.total_records} records ({m.valid_records} valid)")
print(f"  Evasion rate: {m.evasion_count}/{m.malicious_count} = {m.evasion_rate:.4f}")
print(f"  V-ASR: {m.valid_evasion_count}/{m.valid_malicious_count} = {m.valid_malicious_evasion_rate:.4f}")
print(f"  Raw evasion: {m.evasion_count}/{m.total_records} = {m.raw_evasion_rate:.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# Stage 5: Four-arm hardening
# ============================================================
print()
print("=" * 60)
print("STAGE 5: Four-arm hardening")
print("=" * 60)
t0 = time.time()

from xssharden.hardening.arms import run_hardening_arms

hardening = run_hardening_arms(
    train_records=train_records,
    validated_variants=validated,
    detector_factory=lambda: BaselineLRDetector(random_state=SEED),
    budget=BUDGET,
    validation_records=validation_records,
    seed=SEED,
    max_per_category=MAX_PER_CATEGORY,
)

for name in ["baseline", "naive", "random_valid", "selective"]:
    added = hardening.added_counts.get(name, 0)
    size = hardening.arm_sizes.get(name, 0)
    print(f"  {name}: +{added} augmented, total={size}")

print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# Stage 6: Independent evaluation
# ============================================================
print()
print("=" * 60)
print("STAGE 6: Independent evaluation (clean test + adversarial)")
print("=" * 60)
t0 = time.time()

from xssharden.evaluation.evaluate import evaluate_arms

clean_test = [r for r in records if r.get("split") == "clean-test"]
# Use validated variants as the adversarial test (in production, use held-out seeds)
adv_test = [v for v in validated if v.get("split") == "adv_dev"]
if not adv_test:
    adv_test = validated

print(f"  Clean test: {len(clean_test)} records")
print(f"  Adversarial test: {len(adv_test)} records")

final_eval = evaluate_arms(
    hardening.arm_detectors,
    clean_test,
    adv_test,
    adv_split="adv_dev",
)

for name in ["baseline", "naive", "random_valid", "selective"]:
    arm = final_eval.arms[name]
    ct = arm.clean_metrics
    am = arm.adversarial_metrics
    print(f"  {name}:")
    print(f"    Clean:   acc={ct.accuracy:.4f} F1={ct.f1:.4f} FPR={ct.fpr:.4f}")
    print(f"    Adv:     V-ASR={am.valid_malicious_evasion_rate:.4f} "
          f"evasion={am.evasion_rate:.4f}")

print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# Stage 7: Generate report
# ============================================================
print()
print("=" * 60)
print("STAGE 7: Generate report")
print("=" * 60)
t0 = time.time()

from xssharden.reporting import write_report

metadata = {
    "model": "lr",
    "seed": SEED,
    "budget": BUDGET,
    "target_fpr": TARGET_FPR,
    "max_per_seed": MAX_PER_SEED,
    "max_per_category": MAX_PER_CATEGORY,
    "total_variants": len(all_variants),
    "valid_variants": n_valid,
    "validity_rate": f"{n_valid/len(validated)*100:.1f}%",
}

report_summary = write_report(
    final_eval.to_dict(),
    str(REPORT_PATH),
    metadata=metadata,
)
print(f"  Wrote {report_summary['format']} report to {REPORT_PATH}")
print(f"  Time: {time.time()-t0:.1f}s")

# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
print(f"  Variants generated: {len(all_variants)} ({len(variants)} programmatic + {len(llm_variants)} LLM)")
print(f"  Variants valid:     {n_valid} ({n_valid/len(validated)*100:.1f}%)")
print(f"  Realizability gap:  evasion(all)={m.evasion_rate:.4f} vs evasion(valid)={m.valid_malicious_evasion_rate:.4f}")
print()
print("  Four-arm comparison (V-ASR on adversarial test):")
for name in ["baseline", "naive", "random_valid", "selective"]:
    am = final_eval.arms[name].adversarial_metrics
    print(f"    {name:15s}  V-ASR={am.valid_malicious_evasion_rate:.4f}")
print()
print(f"  Report: {REPORT_PATH}")
