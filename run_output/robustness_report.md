# XSSHarden Robustness Report

## Experiment metadata

- **adversarial_variants**: 7
- **browser_validated**: All variants confirmed valid via Playwright
- **date**: 2026-09-07
- **detector**: TF-IDF + Logistic Regression
- **experiment**: Preliminary Adversarial Hardening Test
- **note**: Small augmentation budget (7 variants) — full experiment needs thousands
- **training_samples**: 31626
- **variant_technique**: Per-char octal concat + newlines + mixed-case event handlers

## Input counts

- Clean-test records: 6477
- Adversarial records: 8
- Clean split: clean-test
- Adversarial split: adv_test
- Arm `baseline`: clean_count=6477, adversarial_count=8
- Arm `naive`: clean_count=6477, adversarial_count=8
- Arm `random_valid`: clean_count=6477, adversarial_count=8
- Arm `selective`: clean_count=6477, adversarial_count=8

## Arm comparison

| Arm | Clean acc | Clean precision | Clean recall | Clean F1 | Clean FPR | V-ASR valid_malicious_evasion_rate (valid evasions / valid malicious) | Raw raw_evasion_rate (evasions / total) | Evasion evasion_rate (evasions / malicious) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.9858 | 0.7665 | 1.0000 | 0.8678 | 0.0149 | 0.2500 (2/8) | 0.2500 (2/8) | 0.2500 (2/8) |
| naive | 0.9866 | 0.7763 | 1.0000 | 0.8741 | 0.0141 | 0.0000 (0/8) | 0.0000 (0/8) | 0.0000 (0/8) |
| random_valid | 0.9866 | 0.7763 | 1.0000 | 0.8741 | 0.0141 | 0.0000 (0/8) | 0.0000 (0/8) | 0.0000 (0/8) |
| selective | 0.9866 | 0.7763 | 1.0000 | 0.8741 | 0.0141 | 0.0000 (0/8) | 0.0000 (0/8) | 0.0000 (0/8) |

### Arm `baseline`

- Clean: total=6477, positives=302, negatives=6175, tp=302, tn=6083, fp=92, fn=0
- Clean rates: accuracy=0.9858, precision=0.7665, recall=1.0000, f1=0.8678, fpr=0.0149 (fp=92/(fp+tn)=6175), threshold=0.0144
- Adversarial: total_records=8, valid_records=8, malicious_count=8, evasion_count=2, valid_malicious_count=8, valid_evasion_count=2
- valid_malicious_evasion_rate (V-ASR)=0.2500 (2/8); raw_evasion_rate=0.2500 (2/8); evasion_rate=0.2500 (2/8); threshold=0.0144

### Arm `naive`

- Clean: total=6477, positives=302, negatives=6175, tp=302, tn=6088, fp=87, fn=0
- Clean rates: accuracy=0.9866, precision=0.7763, recall=1.0000, f1=0.8741, fpr=0.0141 (fp=87/(fp+tn)=6175), threshold=0.0147
- Adversarial: total_records=8, valid_records=8, malicious_count=8, evasion_count=0, valid_malicious_count=8, valid_evasion_count=0
- valid_malicious_evasion_rate (V-ASR)=0.0000 (0/8); raw_evasion_rate=0.0000 (0/8); evasion_rate=0.0000 (0/8); threshold=0.0147

### Arm `random_valid`

- Clean: total=6477, positives=302, negatives=6175, tp=302, tn=6088, fp=87, fn=0
- Clean rates: accuracy=0.9866, precision=0.7763, recall=1.0000, f1=0.8741, fpr=0.0141 (fp=87/(fp+tn)=6175), threshold=0.0147
- Adversarial: total_records=8, valid_records=8, malicious_count=8, evasion_count=0, valid_malicious_count=8, valid_evasion_count=0
- valid_malicious_evasion_rate (V-ASR)=0.0000 (0/8); raw_evasion_rate=0.0000 (0/8); evasion_rate=0.0000 (0/8); threshold=0.0147

### Arm `selective`

- Clean: total=6477, positives=302, negatives=6175, tp=302, tn=6088, fp=87, fn=0
- Clean rates: accuracy=0.9866, precision=0.7763, recall=1.0000, f1=0.8741, fpr=0.0141 (fp=87/(fp+tn)=6175), threshold=0.0147
- Adversarial: total_records=8, valid_records=8, malicious_count=8, evasion_count=0, valid_malicious_count=8, valid_evasion_count=0
- valid_malicious_evasion_rate (V-ASR)=0.0000 (0/8); raw_evasion_rate=0.0000 (0/8); evasion_rate=0.0000 (0/8); threshold=0.0147

## Methodology

- Four required arms are compared at a fixed operating threshold: baseline (original training data only), naive (original plus all valid generated variants), random_valid (original plus a budget-matched random subset of valid variants), and selective (original plus validity-gated detector-impact-selected variants).
- Validity is a hard gate: only behaviorally valid malicious variants count toward the valid attack success rate (V-ASR). Impact ranking uses I(x) = 1 - p_detector(x) over valid variants only; invalid variants are logged and counted, never selected.
- The realizability gap is reported explicitly: V-ASR (valid_malicious_evasion_rate over the valid malicious denominator) versus the raw evasion rate (over all records). Thresholds are calibrated on clean validation data and held fixed across arms.
- Evaluation is leakage-controlled: clean-test and adversarial inputs use held-out splits that never influenced training, selection, or threshold tuning.

## Safety notes

- Payloads are treated as inert strings throughout reporting: nothing is rendered, executed, fetched, or sent anywhere, and raw payload text is excluded from this report by default.
- Browser validation, when run elsewhere in the pipeline, uses only the controlled local sandbox; reporting itself performs no validation and launches no browser.

## Caveat and limitations

- Caveat: the current two-source data (http_params_dataset plus kaggle_xss_dataset, 2 distinct sources) cannot support meaningful three-way leakage-safe train/validation/clean-test evaluation under group-atomic splitting. Clean-test and adversarial claims stay preliminary until a third independent source is added.
- Negative results are reportable: selective augmentation matching the random-valid control would indicate that validity filtering and additional data matter more than the selection strategy.
