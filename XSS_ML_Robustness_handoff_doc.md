# PROJECT HANDOFF DOCUMENT
## AI-Driven Adversarial Testing and Selective Hardening of Machine Learning-Based XSS Detectors

**Document purpose:**  
This document transfers the complete project context, decisions, architecture, research direction, constraints, and next steps to another AI/agent. Treat this as the current source of truth for the project.

---

# 1. PROJECT STATUS

We have finished the idea-selection and research-direction phase.

**Do NOT restart the project from scratch.**  
Do NOT repeatedly question whether the general idea is good unless new implementation or literature evidence creates a concrete problem.

The project direction is currently **locked**.

The next phase is implementation and experimentation.

The project is being developed by a **3-person student team** with approximately **8 weeks** available.

The final product is a **runnable research prototype/pipeline**, not a large commercial web application.

---

# 2. PROJECT TITLE

## AI-Driven Adversarial Testing and Selective Hardening of Machine Learning-Based XSS Detectors

Short working name:

**XSSHarden**

---

# 3. PROJECT IN SIMPLE TERMS

Web applications can be attacked using Cross-Site Scripting (XSS), where malicious input causes JavaScript to execute in a victim's browser.

Machine-learning models can be trained to detect malicious XSS payloads.

The problem is that a detector may recognize the *appearance* of known attacks rather than their underlying malicious behavior.

For example, an attacker can modify an XSS payload using:

- encoding
- different syntax
- whitespace
- capitalization
- alternate HTML structures
- different JavaScript construction
- other semantic rewrites

while still producing a functioning XSS attack.

LLMs make it easier to automatically generate many such variants.

The central question of this project is:

> If we generate XSS variants using an LLM, verify that they actually execute as real XSS attacks, identify which valid attacks can fool our detector, and retrain the detector using those difficult examples, can we make the detector more robust against previously unseen attack variants?

---

# 4. CORE RESEARCH IDEA

The project has two sides.

### Attacker side

```text
Known XSS
   ↓
Generate variants
   ↓
Validate whether they actually work
   ↓
Test against ML detector
   ↓
Find successful evasions
```

### Defender side

```text
Valid difficult attacks
   ↓
Select useful examples
   ↓
Add them to training data
   ↓
Retrain detector
   ↓
Test against fresh unseen attacks
```

The research is primarily about whether **behavioral validation + detector-aware selective augmentation** improves robustness compared with simpler augmentation approaches.

---

# 5. PROBLEM STATEMENT

Machine-learning-based detectors are increasingly used to identify web application attacks such as Cross-Site Scripting (XSS). Although these detectors can achieve high performance on standard benchmark datasets, their effectiveness can decrease when attackers modify or obfuscate malicious payloads while preserving attack intent. Large language models make automated generation of such variants increasingly accessible, but simply adding generated examples to detector training does not necessarily produce robust detectors.

This project therefore develops a system that generates XSS variants, validates whether they remain behaviorally functional in a controlled browser environment, measures their ability to evade ML-based detection, selectively identifies valid detector-challenging examples, and uses them to harden the detector. The hardened detectors are then evaluated against fresh attack seeds and previously unseen mutation categories under a leakage-controlled experimental protocol.

---

# 6. RESEARCH GAP

Existing research already covers several individual components:

- ML-based XSS detection
- adversarial/mutation-based web attack evasion
- LLM-generated XSS variants
- browser/runtime validation of generated XSS
- adversarial training
- hard-example mining
- closed-loop adversarial systems

Therefore, we are NOT claiming these individual components as completely new.

The closest identified work is:

**Gabbireddy & Saha (2026), "Evaluating LLM-Generated Obfuscated XSS Payloads for Machine Learning-Based Detection."**

Our project treats this work as the closest baseline/extension point.

The intended research gap is:

1. Their behavioral validation is primarily a validity filter.
2. We additionally examine **detector impact** when selecting valid adversarial examples.
3. We use a larger augmentation budget.
4. We compare selective augmentation against a **budget-matched random-valid control**.
5. We use a leakage-controlled independent adversarial test set.
6. We reserve mutation categories/styles for the final test so that the detector cannot simply memorize the generator's patterns.
7. We explicitly measure the difference between raw generated variants and behaviorally valid variants.

The project therefore investigates whether **validity-gated detector-impact selection** produces better robustness than simply adding valid generated data.

---

# 7. NOVELTY CLAIM

Do NOT claim:

> "We invented LLM-generated XSS attacks."

Do NOT claim:

> "Behavioral validation of adversarial examples is completely new."

Do NOT claim:

> "Adversarial training is new."

Do NOT claim:

> "Closed-loop adversarial testing is new."

Instead, the novelty is the **domain-specific combination and empirical evaluation**:

> A behavior-validated, detector-impact-based selective augmentation pipeline for ML-based XSS detection, evaluated at meaningful augmentation scale using leakage-controlled, category-holdout adversarial testing and compared against naive and budget-matched random augmentation.

The key research question is whether the **selection stage actually provides additional robustness** beyond simply adding more valid data.

---

# 8. FINAL RESEARCH QUESTIONS

### RQ1

Can an LLM generate valid, behaviorally-real XSS variants at meaningful scale?

### RQ2

How much do behaviorally-valid AI-generated variants degrade ML-based XSS detector performance compared with standard test data?

Also measure the difference between:

- all generated variants
- behaviorally validated variants

This is the **realizability gap**.

### RQ3

Does naively augmenting training data with valid generated XSS variants improve detector robustness on an independent adversarial test set?

### RQ4

Does detector-impact-based selection of valid examples outperform:

1. naive all-valid augmentation, and
2. a budget-matched random-valid control?

---

# 9. FIXED PROJECT SCOPE

## Included

- XSS only
- ML-based XSS detection
- LLM-generated XSS variants
- programmatic XSS mutations
- browser-based behavioral validation
- adversarial evasion testing
- selective adversarial augmentation
- detector hardening
- independent evaluation
- robustness reporting

## Explicitly excluded

### SQL Injection

Do NOT add SQL injection during the main project.

SQLi requires a separate validation environment and would substantially increase complexity.

It can be mentioned as future work.

### Large-scale web application deployment

Not required.

### Commercial WAF

Not required.

### Full production dashboard

Not required.

### Large transformer model

Not required.

---

# 10. INJECTION CONTEXT

Primary context:

**Reflected XSS in an HTML body context.**

A second context may be added only if the core system is already stable.

Possible second context:

- HTML attribute
- another controlled XSS sink

Do not sacrifice the core experiment to add more contexts.

---

# 11. ML DETECTORS

Two detectors are required.

## Detector 1

```text
XSS payload
    ↓
Character-level TF-IDF
    ↓
Logistic Regression
```

This is the simple baseline.

## Detector 2

```text
XSS payload
    ↓
Lexical + handcrafted structural features
    ↓
XGBoost / LightGBM
```

A transformer such as DistilBERT is an optional stretch goal.

If time becomes limited, do NOT add the transformer.

The research quality comes from the experimental design, not from having many models.

---

# 12. DATA PIPELINE

Initial dataset processing:

```text
Raw XSS dataset
       ↓
Cleaning
       ↓
Exact deduplication
       ↓
Normalized deduplication
       ↓
Label audit
       ↓
Source-aware / cluster-aware splitting
```

The dataset should contain:

```text
sample_id
payload
label
source
attack_category
split
```

Avoid naive random splitting where possible because similar XSS payload families can appear in multiple sets.

---

# 13. DATA SPLITS

The experiment must prevent leakage.

Conceptually:

```text
Original Dataset
       │
       ├── Train
       ├── Validation
       └── Clean Test
```

Separately:

```text
Train-side attack seeds
       ↓
Adversarial Development Pool
       ↓
Generation
       ↓
Validation
       ↓
Selection
       ↓
Hardening
```

Final adversarial evaluation:

```text
Fresh attack seeds
       +
Mutation categories/styles
never used during hardening
       ↓
FINAL ADVERSARIAL TEST
```

The final adversarial test must never influence:

- generation decisions
- selection
- model training
- threshold tuning

This is a critical requirement.

---

# 14. ATTACK GENERATION

Use a hybrid generation system.

## A. Programmatic mutation

Examples of mutation categories:

- encoding transformations
- whitespace changes
- comment insertion
- capitalization changes
- syntactic variations
- HTML structure variations
- event-handler substitutions

These are useful because they are:

- deterministic
- cheap
- reproducible
- easy to categorize

## B. LLM generation

The LLM generates alternative forms of known XSS payloads.

Possible local model:

- Ollama-based model

An API model can be used as fallback.

The LLM is a generator, NOT the validator.

The LLM must never be trusted to decide whether its own payload works.

---

# 15. VARIANT METADATA

Every generated payload should have provenance information.

Example:

```json
{
  "variant_id": "V001245",
  "seed_id": "S023",
  "payload": "...",
  "generator": "llm",
  "mutation_category": "encoding",
  "context": "reflected_html",
  "valid": null,
  "detector": null,
  "prediction": null,
  "confidence": null,
  "evasion": null,
  "selected": false,
  "split": "adv_dev"
}
```

The exact schema can evolve slightly, but provenance must not be removed.

---

# 16. PROBE / MARKER DESIGN

A major implementation concern is **marker leakage**.

Do NOT make every payload execute exactly the same obvious marker such as:

```text
alert(MARKER123)
```

because the detector might learn the marker rather than XSS behavior.

Use varied probes where appropriate.

Possible approaches include:

- different probe function names
- randomized identifiers
- different observable execution mechanisms
- controlled browser instrumentation

The exact implementation should prioritize reliable validation while preventing a single fixed marker from becoming a trivial ML feature.

---

# 17. BEHAVIORAL VALIDATION

This is one of the most important components.

Use:

- Playwright
- Chromium
- controlled local vulnerable application/sandbox

Pipeline:

```text
Generated XSS
      ↓
Controlled vulnerable page
      ↓
Browser execution
      ↓
Observe probe
      ↓
┌───────────────┐
│               │
VALID          INVALID
```

A payload is considered valid if the intended JavaScript execution is observed within a defined timeout.

Record:

```text
variant_id
valid
context
execution_result
timeout
error
```

Do NOT simply delete invalid examples.

Keep them in the experiment records because the invalid-generation rate itself is useful.

---

# 18. REALIZABILITY GAP

One important result is:

```text
All generated variants
        ↓
How many evade detector?
```

versus:

```text
Only behaviorally valid variants
        ↓
How many evade detector?
```

This tells us whether counting syntactically generated payloads overestimates the practical attack threat.

The direction of the result is not predetermined.

If the difference is small, that is still a result.

If the difference is large, that is also a result.

---

# 19. ADVERSARIAL TESTING

After behavioral validation:

```text
Valid XSS
    ↓
Shared feature pipeline
    ↓
ML detector
    ↓
Prediction
```

If:

```text
Actual = malicious
Prediction = benign
```

then the variant successfully evaded the detector.

The primary robustness metric can be defined as:

\[
V\text{-ASR}
=
\frac{\text{valid adversarial examples that evade}}
{\text{total valid adversarial examples}}
\]

Clearly define this metric in the paper rather than implying that the terminology itself is universally standardized.

---

# 20. THRESHOLD CALIBRATION

Do not arbitrarily choose a detector threshold.

Calibrate the decision threshold using clean validation data.

A target operating point such as:

```text
FPR = 1%
```

may be used if appropriate.

Then keep the threshold fixed when comparing adversarial performance.

This prevents apparent evasion from simply being caused by poor threshold calibration.

---

# 21. SELECTION ENGINE

Validity is a **hard requirement**.

Do NOT use:

\[
S(x)=\alpha V(x)+\beta I(x)
\]

because an invalid attack should never become selectable merely because it scores highly on detector impact.

Instead:

```text
Generated variants
       ↓
Behavioral validation
       ↓
Keep valid only
       ↓
Deduplicate
       ↓
Measure detector impact
       ↓
Rank
       ↓
Category balancing
       ↓
Select B examples
```

Possible detector-impact measure:

\[
I(x)=1-P(\text{malicious}|x)
\]

Higher impact means the detector is less confident that the valid attack is malicious.

The exact ranking definition may be refined during implementation.

---

# 22. FOUR REQUIRED EXPERIMENTAL ARMS

This is one of the most important parts of the project.

## Arm A — Baseline

```text
Original training data only
```

## Arm B — Naive

```text
Original training data
+
all valid generated variants
```

## Arm C — Random-valid

```text
Original training data
+
random subset of valid variants
```

The number of added examples must match Arm D.

## Arm D — Selective

```text
Original training data
+
detector-impact-selected valid variants
```

The selective set must contain the same number of added examples as the random-valid arm.

---

# 23. WHY RANDOM-VALID IS REQUIRED

Without Arm C, if selective augmentation performs better than the baseline, we cannot determine whether:

```text
selection strategy
```

caused the improvement

or simply:

```text
more valid training data
```

caused the improvement.

Therefore:

```text
Baseline
vs
Naive
vs
Random-valid
vs
Selective
```

is the core comparison.

---

# 24. FINAL EVALUATION

All four models must eventually be tested on:

### Clean test data

Measures:

- F1
- precision
- recall
- false-positive rate

### Independent adversarial test

Measures:

- valid attack success rate / evasion rate
- per-category performance
- detector confidence
- robustness against unseen attack styles

The final adversarial test should use:

- fresh seeds
- disjoint payload families where possible
- mutation categories/styles not used during hardening

---

# 25. PRODUCT

The product is NOT simply an ML model.

The product is the complete runnable pipeline:

```text
Dataset
   ↓
Train detector
   ↓
Generate attacks
   ↓
Validate attacks
   ↓
Attack detector
   ↓
Select useful attacks
   ↓
Harden detector
   ↓
Evaluate robustness
   ↓
Generate report
```

The product should be usable from the command line.

Example interface:

```bash
xssharden dataset build

xssharden train --model lr

xssharden generate --n 5000

xssharden validate

xssharden attack --model lr

xssharden select --strategy impact --budget 1000

xssharden retrain --model lr --augment selected

xssharden evaluate --model lr --test independent

xssharden report
```

A single end-to-end command can eventually be added:

```bash
xssharden run experiment.yaml
```

---

# 26. PRODUCT OUTPUT

The final product should produce a robustness report containing:

```text
Dataset statistics
↓
Generation statistics
↓
Behavioral validity rate
↓
Realizability gap
↓
Baseline detector performance
↓
Adversarial evasion results
↓
Training strategy comparison
↓
Clean FPR/F1
↓
Independent adversarial V-ASR
↓
Per-category results
```

Possible output:

```text
reports/
└── experiment_001.html
```

The HTML report is part of the product.

---

# 27. PROPOSED PROJECT STRUCTURE

```text
xssharden/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── seeds/
│   └── splits/
│
├── models/
│   ├── baseline/
│   ├── naive/
│   ├── random/
│   └── selective/
│
├── variants/
│   ├── generated.jsonl
│   ├── validated.jsonl
│   └── selected.jsonl
│
├── sandbox/
│   ├── vulnerable_app/
│   └── validator/
│
├── experiments/
│
├── reports/
│
├── src/
│   ├── dataset/
│   ├── detectors/
│   ├── generators/
│   ├── validator/
│   ├── attacker/
│   ├── selector/
│   ├── hardening/
│   ├── evaluation/
│   └── reporting/
│
├── tests/
│
├── configs/
│
└── README.md
```

The exact structure may change during implementation.

---

# 28. TECHNOLOGY STACK

Required:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost or LightGBM
- Playwright
- Chromium
- pytest
- SQLite or JSONL
- Jinja2 for reports

Generation:

- Ollama/local LLM where practical
- API fallback if required

Optional:

- PyTorch
- Transformers
- DistilBERT
- Streamlit
- Docker

Do not add unnecessary technologies just to make the architecture look impressive.

---

# 29. TEAM STRUCTURE

There are three people.

## Person A — Models & Evaluation

Responsible for:

- dataset processing
- ML models
- feature extraction
- threshold calibration
- experiments
- metrics
- statistical evaluation
- result tables

Main responsibility:

> Are our experimental results trustworthy?

---

## Person B — Behavioral Validation & Corpus

Responsible for:

- Playwright
- Chromium sandbox
- vulnerable test application
- execution probes
- validation pipeline
- dataset/seed auditing
- leakage checking

Main responsibility:

> Is every attack we call "valid" actually a working attack?

---

## Person C — Generation & Integration

Responsible for:

- programmatic mutations
- LLM integration
- output parsing
- deduplication
- selection engine
- pipeline orchestration
- report generation

Main responsibility:

> Does the complete system work as one reproducible pipeline?

---

# 30. 8-WEEK PLAN

## Week 1

Person A:

- dataset audit
- establish ML evaluation framework

Person B:

- Playwright validation prototype
- test with known XSS examples

Person C:

- repository
- schemas
- configuration system
- shared feature pipeline

Critical milestone:

**Dataset + validator architecture established.**

---

## Week 2

Person A:

- baseline Logistic Regression
- XGBoost
- clean metrics

Person B:

- working browser validator
- test with known valid/invalid cases

Person C:

- programmatic mutation engine

Critical milestone:

**A known XSS can be generated and behaviorally validated.**

---

## Week 3

Person A:

- baseline evaluation
- threshold calibration

Person B:

- validator reliability
- probe variation

Person C:

- LLM generation
- provenance
- deduplication

---

## Week 4

First complete adversarial experiment.

```text
Generate
 ↓
Validate
 ↓
Attack detector
 ↓
Measure evasion
```

Calculate the initial realizability gap.

---

## Week 5

Implement:

- Arm A
- Arm B
- Arm C
- Arm D

Run initial hardening experiments.

---

## Week 6

Freeze the experiment design.

Build:

- independent adversarial test
- unseen mutation categories
- leakage checks
- complete pipeline run

---

## Week 7

Run:

- ablations
- per-category analysis
- statistical analysis
- final experiments

Build HTML report.

---

## Week 8

Focus on:

- final experiments
- paper/report
- figures
- methodology
- limitations
- reproducibility
- product packaging

---

# 31. DROP ORDER IF WE FALL BEHIND

If time becomes limited:

### First remove

Streamlit/dashboard.

### Second

Transformer baseline.

### Third

Second XSS injection context.

### Fourth

Extra ablation experiments.

### Never remove

1. Behavioral validation
2. Leakage-controlled evaluation
3. Independent adversarial test
4. Random-valid control
5. Selective augmentation
6. Baseline comparison

These are central to the research question.

---

# 32. IMPORTANT RESEARCH PRINCIPLES

### Principle 1 — Don't assume generated = valid

An LLM output is only a candidate until the browser confirms execution.

### Principle 2 — Don't confuse evasion with broken attacks

A payload that doesn't work should not be counted as a successful real-world attack.

### Principle 3 — Don't train on the final test

The final test set must remain untouched.

### Principle 4 — Don't claim selection is novel by itself

Hard-example selection is an established ML idea.

### Principle 5 — Compare against random

Otherwise we cannot tell whether selection actually matters.

### Principle 6 — Negative results are acceptable

If:

```text
Selective ≈ Random
```

that does NOT mean the project failed.

It may indicate:

> Validity filtering and additional data matter more than detector-impact-based selection.

That is a useful research finding.

### Principle 7 — Avoid scope creep

The project is XSS-focused.

Do not suddenly add:

- SQL injection
- network intrusion detection
- malware
- multiple WAF products
- huge model comparisons

unless the project scope is explicitly changed.

---

# 33. EXPECTED ARCHITECTURE

```text
                    ┌─────────────────────┐
                    │     XSS Dataset     │
                    │ Malicious + Benign  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Dataset Processor   │
                    │ Clean / Dedup /     │
                    │ Split / Audit       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ↓                                 ↓
     ┌───────────────────┐            ┌───────────────────┐
     │ Baseline ML       │            │ Seed XSS Payloads │
     │ Detectors         │            └─────────┬─────────┘
     │                   │                      │
     │ TF-IDF + LR       │             ┌────────┴────────┐
     │ XGBoost           │             ↓                 ↓
     └─────────┬─────────┘      ┌─────────────┐   ┌─────────────┐
               │                │ Programmatic│   │ LLM         │
               │                │ Mutator     │   │ Generator   │
               │                └──────┬──────┘   └──────┬──────┘
               │                       └────────┬─────────┘
               │                                ↓
               │                     ┌──────────────────┐
               │                     │ Candidate XSS    │
               │                     └────────┬─────────┘
               │                              ↓
               │                     ┌──────────────────┐
               │                     │ Behavioral       │
               │                     │ Validator        │
               │                     │ Playwright       │
               │                     │ Chromium         │
               │                     └────────┬─────────┘
               │                              │
               │                    ┌─────────┴─────────┐
               │                    ↓                   ↓
               │                INVALID              VALID
               │                    │                   │
               │                    │                   ↓
               │                    │          ┌─────────────────┐
               └────────────────────┼─────────→│ Adversarial     │
                                    │          │ Tester          │
                                    │          └────────┬────────┘
                                    │                   ↓
                                    │          ┌─────────────────┐
                                    │          │ Selection       │
                                    │          │ Engine          │
                                    │          └────────┬────────┘
                                    │                   │
                                    │          ┌────────┴────────┐
                                    │          ↓        ↓        ↓
                                    │       Naive    Random  Selective
                                    │          │        │        │
                                    │          └────────┼────────┘
                                    │                   ↓
                                    │          ┌─────────────────┐
                                    │          │ Hardened Models │
                                    │          └────────┬────────┘
                                    │                   ↓
                                    │          ┌─────────────────┐
                                    │          │ Independent     │
                                    │          │ Adversarial Test│
                                    │          │ Fresh Seeds +   │
                                    │          │ Held-out Styles │
                                    │          └────────┬────────┘
                                    │                   ↓
                                    │          ┌─────────────────┐
                                    └─────────→│ Robustness      │
                                               │ Report          │
                                               └─────────────────┘
```

---

# 34. CURRENT DECISION

The following decisions are considered **frozen**:

- XSS only
- two baseline ML detectors
- hybrid programmatic + LLM generation
- browser-based behavioral validation
- validity as a hard gate
- detector-impact-based selection
- four experimental arms
- random-valid control
- independent adversarial evaluation
- mutation-category holdout
- CLI-based product
- HTML/Markdown report
- approximately 8-week implementation window

Do not redesign these without a concrete technical reason.

---

# 35. WHAT THE NEXT AI/AGENT SHOULD DO

The project is now ready for implementation.

The next AI should NOT respond with another generic explanation of the idea.

Instead, it should help execute the project in this order:

### Phase 1 — Foundation

1. Create the repository structure.
2. Define the data schema.
3. Define configuration files.
4. Establish reproducible experiment IDs/seeds.
5. Implement dataset loading and cleaning.
6. Implement the shared feature pipeline.

### Phase 2 — Baseline

1. Train TF-IDF + Logistic Regression.
2. Train XGBoost.
3. Establish clean validation/test metrics.
4. Implement threshold calibration.

### Phase 3 — Behavioral Validation

1. Create controlled local XSS sandbox.
2. Implement Playwright validator.
3. Test it against known valid and invalid cases.
4. Add regression tests.
5. Ensure validation is reliable before generating thousands of examples.

### Phase 4 — Generation

1. Implement programmatic mutations.
2. Implement LLM generation.
3. Implement JSON parsing.
4. Implement provenance.
5. Implement deduplication.

### Phase 5 — Adversarial Testing

1. Generate candidate variants.
2. Validate them.
3. Test valid variants against baseline detectors.
4. Calculate evasion/V-ASR.
5. Calculate realizability statistics.

### Phase 6 — Selection & Hardening

Implement:

```text
Baseline
Naive
Random-valid
Selective
```

with equal-budget comparison.

### Phase 7 — Independent Evaluation

Build the final leakage-controlled adversarial test set.

Do not allow it to influence training or selection.

### Phase 8 — Reporting

Generate:

- tables
- metrics
- graphs
- robustness comparison
- final HTML report

---

# 36. IMPORTANT: HOW TO WORK WITH THIS PROJECT

When making implementation decisions:

1. Prefer the simplest implementation that preserves the research validity.
2. Do not add features merely because they sound impressive.
3. Keep every experiment reproducible.
4. Keep provenance for every generated sample.
5. Keep training/test boundaries explicit in code.
6. Treat the behavioral validator as a critical component.
7. Treat the random-valid control as mandatory.
8. Report negative results honestly.
9. Avoid changing multiple experimental variables at once.
10. When uncertain about a research-methodology decision, explain the trade-off before changing the design.

---

# 37. FINAL ONE-SENTENCE DESCRIPTION

> **XSSHarden is a research prototype that generates XSS variants using programmatic mutation and LLMs, verifies that they remain functional attacks through browser execution, measures their ability to evade ML detectors, selectively uses difficult valid examples to harden those detectors, and evaluates the resulting robustness against fresh unseen attack styles.**

---

# END OF HANDOFF DOCUMENT
