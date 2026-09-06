# AI-Driven Adversarial Testing and Selective Hardening of Machine Learning-Based XSS Detectors

**Consolidated final proposal — merges all decisions from the ChatGPT, Zai, Qwen, and Claude review rounds. This is the single reference document; treat prior chat threads as superseded.**

---

## 0. What This Project Is (Plain Explanation)

Websites get attacked by malicious inputs like XSS (scripts injected into a page to steal data or hijack sessions). Security teams train ML models to catch these automatically, and those models do well on textbook attacks — but attackers can rewrite a payload (different encoding, casing, structure) so it still works identically while looking different to the model. LLMs make generating hundreds of these rewrites trivial.

The closest existing research (Gabbireddy & Saha, 2026) already checks whether generated rewrites still work as real attacks before using them, but only filters on that (valid vs. invalid) — and their augmented set was too small (under 1% of their data) to show much effect either way. Nobody has combined **real-attack validation** with **detector-impact-based selection**, tested at a **meaningful scale**, under a **leakage-free evaluation**.

**So the project builds and tests, step by step:**
1. Train baseline ML detectors on a cleaned XSS dataset.
2. Generate rewritten attack variants (rule-based mutations + LLM rewrites) from known attacks.
3. Confirm each variant is a **real, working** attack by actually running it in a browser sandbox — not just checking if it looks like one.
4. Test how many of these confirmed-real variants slip past the current detector.
5. Retrain the detector four different ways — untouched, naive (add everything), random (add a same-sized random sample), and selective (add the examples that are both real and specifically fool the detector) — and compare which training approach actually produces the most robust detector.
6. Test all four fairly, on fresh attacks and attack styles none of them saw during training, so nothing is just memorizing the test.
7. Package it as a runnable tool that outputs a before/after robustness report — that tool is the product deliverable.

**The research question in one sentence:** if you only count AI-generated attacks that are confirmed to genuinely work, and you specifically pick the ones that fool your detector (rather than adding attacks at random), does that produce a measurably more robust detector than the simpler approaches — evaluated in a way that can't cheat by testing on data too similar to training.

---

## 1. Problem Statement

Machine-learning-based detectors are increasingly used to identify web application attacks such as Cross-Site Scripting (XSS). Although these detectors can achieve high performance on standard benchmark datasets, their effectiveness can decrease when attackers modify or obfuscate malicious payloads while preserving attack intent. Large language models make automated generation of such variants increasingly accessible, but recent evidence shows that simply adding generated examples to detector training does not reliably improve robustness — suggesting that **which** generated examples are used, and whether they are confirmed to still function as real attacks, matters more than generation volume alone.

This project builds a system that generates LLM-based XSS variants, validates their behavioral reality in a controlled sandbox, measures their impact on an ML detector, selectively identifies useful adversarial examples at a meaningful scale, and uses those examples to harden the detector — evaluated under a leakage-free protocol.

---

## 2. Research Gap

**What already exists (verified, not assumed):**
- ML-based XSS detection is mature (lexical/TF-IDF, transformer-based).
- Mutation-based adversarial evasion of ML/WAF web detectors is established (WAF-A-MoLE).
- Adversarial training for web-attack detectors is established (ModSec-AdvLearn, for SQLi).
- LLM-generated obfuscated XSS with browser-based runtime validation is established — **confirmed directly** via full-text reading of Gabbireddy & Saha, 2026 (arXiv:2604.19526), a Penn State undergraduate honors thesis. Their pipeline: deterministic obfuscation chains → LLM fine-tuning on behavior-filtered pairs → browser-based runtime match-rate evaluation → downstream Random Forest comparison (original / naive-augmented / valid-only-augmented).
- Closed-loop generate→attack→adapt loops are established in adjacent domains (autonomous driving, Android malware, LLM-text detection) — not novel in general, only their application here.
- Hard-example mining / active learning (selecting difficult examples for training) is a mature, general ML technique — must be cited as prior art, not claimed as new.

**What Gabbireddy & Saha (2026) does NOT do — this is the specific, verified gap:**
1. No detector-impact-based selection — their only filter is binary behavioral validity; they never select on "does this valid payload actually fool the detector."
2. Augmentation scale is small — 1,000 chains generated, only 200 validated, ~100 evaluated per model, augmented set was under 1% of their 37,605-row dataset. Their own paper attributes the null result to this: *"generated samples represent a very small fraction of the full dataset."*
3. No category-holdout / leakage-free independent test — they used a random 80/20 split of the same generated pool, not fresh seeds and unseen mutation categories.
4. Single classifier (Random Forest) only.
5. No standalone realizability-gap measurement (raw vs. validity-filtered evasion rate as its own reported finding).
6. Their own conclusion explicitly invites this: *"future work should evaluate whether behavior-filtered augmentation has different effects... in a larger augmentation setting."*

---

## 3. Novelty Statement

We are **not** claiming: LLM generation of attacks, behavioral/runtime validation itself, adversarial training, or closed-loop hardening — all exist and are cited as prior art.

We **are** claiming: at a properly scaled augmentation budget, with detector-impact-based selection layered on top of validity filtering (not validity filtering alone), evaluated under a leakage-free, category-holdout protocol, and compared against a budget-matched random-valid control — does selective augmentation produce a measurable robustness gain that a validity-only, small-scale study (the closest prior work) could not detect? This is a direct, named extension of Gabbireddy & Saha's own stated limitation, and it fills a gap they left explicitly open.

---

## 4. Research Questions

- **RQ1:** Can an LLM generate valid, behaviorally-real (execution-confirmed) adversarial XSS variants at meaningful scale?
- **RQ2:** How much do behaviorally-valid AI-generated variants degrade ML-based XSS detector performance relative to standard test data — and does the evasion rate differ materially between validated and unvalidated (raw) generated variants (the "realizability gap")?
- **RQ3:** Does naively augmenting training data with all valid generated examples improve robustness on an independent, leakage-free adversarial test set?
- **RQ4:** Does detector-impact-based selection of valid examples outperform (a) naive all-valid augmentation and (b) a budget-matched random-valid control, at equal augmentation size?

---

## 5. Scope

- **XSS only.** SQL injection is an explicit non-goal for this project — not even a stretch extension — because it requires a second, semantically fuzzier validator (DB sandbox/parser) that would double validation-engineering risk for no proportional benefit to the core research question.
- **One primary injection context first:** reflected HTML body. A second context (attribute value, or `javascript:` URI) only if ahead of schedule by week 5.
- **Two baseline detectors required:** TF-IDF char n-gram + calibrated Logistic Regression, and XGBoost/LightGBM on lexical + handcrafted features. A small transformer (DistilBERT) is a stretch goal only, added if the core pipeline is stable by week 6 — never a requirement.

---

## 6. Methodology

### Stage 1 — Dataset
- Primary source: one audited public XSS payload corpus (payload/benign structure), deduplicated (exact-hash and normalized-text), with a manual audit of a random sample of "benign" rows before trusting the label.
- Dedicated benign corpus: real JavaScript/HTML snippets from public sources — necessary for a meaningful false-positive-rate measurement, not optional.
- Seed pools (mutation sources, not test data): PortSwigger XSS cheat sheet, PayloadsAllTheThings — verify license before any redistribution.
- **Splitting rule:** source-aware and cluster-aware, never a naive random split. Some payload families/categories are reserved exclusively for the final independent test set and must never be used as generation seeds.

### Stage 2 — Baseline Detectors
Train the two required models (Stage 5 above) on one shared `featurize()` module used identically everywhere in the pipeline — an inconsistent feature pipeline invalidates every downstream comparison. Calibrate probabilities (needed for both thresholding and the impact-ranking step in Stage 6).

### Stage 3 — Generation (hybrid)
- **Programmatic mutators** (deterministic, cheap, auditable): encoding chains, whitespace/comment insertion, tag/event-handler substitution, case swapping.
- **LLM mutator** (local via Ollama, e.g. Llama 3.1 8B, with a cheap API as fallback): semantic variation, novel bypass phrasing, context re-wrapping.
- Output contract: JSON with `{payload, mutation_category, seed_id, context_target}`. Every payload embeds a **varied** probe (not one fixed marker — alternate between `alert()`, `prompt()`, `confirm()`, a randomized custom function name) to avoid the detector learning to recognize a single marker string instead of XSS structure in general (marker-leakage risk). Reject duplicates by hash.
- Target volume: enough that the augmented set is a **meaningful fraction of training data**, not the <1% that likely explains Gabbireddy & Saha's null result — aim for thousands of validated variants, not hundreds.

### Stage 4 — Behavioral Validation (the project's spine)
- Playwright + headless Chromium + a local sandboxed vulnerable page, one sink per context in scope.
- A payload is **valid** only if it causes the embedded probe to fire, observed via hooked `alert`/`confirm`/`prompt` or a custom probe function, within a timeout (2-5s).
- Record validity as a hard binary flag plus context metadata for every variant — invalid variants are kept in the dataset (not discarded from records) because the invalid rate is itself a reportable finding (compare to Gabbireddy & Saha's 44% chain-level validation rate as a benchmark).

### Stage 5 — Attack the Baseline Detector
- Feed only validated variants through the shared feature pipeline.
- Calibrate the decision threshold on clean validation data at a fixed FPR (e.g. 1%) before measuring evasion — prevents evasion numbers from being an artifact of a miscalibrated threshold.
- Report the **realizability gap** explicitly as its own result: evasion rate on all generated variants vs. evasion rate on validated-only variants.

### Stage 6 — Selection (validity-gated impact ranking, not a weighted formula)
Do **not** use a weighted score like `S(x) = αV(x) + βI(x)`. Validity is a hard gate, not one input among several — an invalid payload should never be selectable regardless of how much it confuses the detector, because it isn't a real attack.

```
1. Filter: keep only V(x) = 1 (behaviorally valid).
2. Deduplicate against training set and already-selected examples.
3. Rank survivors by detector impact: I(x) = 1 − p_detector(x)  (lower detector confidence = higher impact).
4. Select top-B under a fixed budget B.
5. Enforce a small cap per mutation category, so augmentation isn't dominated by one obfuscation trick.
```

### Stage 7 — Hardening: Required Comparison Arms
This is the core experiment. All four arms are required — the random-valid arm is non-negotiable, because without it you cannot separate "selection strategy works" from "more data helps":

| Arm | Training data |
|---|---|
| A — Baseline | Original data only |
| B — Naive augmentation | Original + all valid generated variants |
| C — Random-valid (budget-matched control) | Original + a random subset of valid variants, same size as D |
| D — Selective augmentation (our method) | Original + top-B selected by validity-gated impact ranking |

Retrain each detector architecture from scratch per arm, 3 random seeds, report mean ± std. Recalibrate thresholds on clean validation before every adversarial evaluation.

### Stage 8 — Independent Evaluation (leakage-free — this is where the project succeeds or fails)

| Set | Contents | Used for |
|---|---|---|
| Train | 70% of original corpus | Training |
| Validation | 15% | Tuning, threshold calibration |
| Clean test | 15% held out — never touched by generation/selection | Clean performance |
| Adversarial-dev pool | Attack seeds drawn only from train-side | Variant generation + selection |
| Adversarial test (final) | Fresh seeds disjoint from train and adv-dev, mutated using mutation categories held out from hardening | Final evaluation only |

The category-holdout (final test uses mutation styles never used during hardening) is the direct answer to the circularity objection — hardening must generalize to unseen attack styles, not memorize the generator's habits.

### Stage 9 — Statistical Evaluation
- Primary metric: **V-ASR (Valid Attack Success Rate)** — the proportion of valid adversarial payloads that evade detection at the fixed operating threshold — measured on the final independent test set, per arm.
- Secondary: clean F1, clean false-positive rate, per-mutation-category evasion, confidence shift (paired, seed vs. variant).
- Tests: McNemar's test for paired before/after predictions; bootstrap confidence intervals for V-ASR; Holm correction across the arm comparisons; report effect sizes, not just p-values.

---

## 7. Product Architecture

A CLI-driven Python research prototype (not a dashboard-first product):

```
xssharden dataset build
xssharden train --model lr|xgb
xssharden generate --n <count>
xssharden validate
xssharden attack --model <model>
xssharden select --strategy impact --budget <B>
xssharden retrain --model <model> --augment naive|random|selected
xssharden evaluate --model <model> --test independent
xssharden report
```

Every artifact (payload, hash, split membership, mutation category, validity, provenance) is logged so leakage is auditable after the fact. An HTML/Markdown robustness report is generated at the end. A Streamlit dashboard is an optional week-8 nicety only if the core pipeline is done early — never a required deliverable.

---

## 7a. System Architecture Diagram

Corrects two errors from an earlier draft: (1) generation must start from the original seed payloads, not from the trained detector — the detector only enters at the attack-testing stage; (2) hardening must branch into all four required comparison arms, not just the selective one.

```
                     ┌──────────────────────┐
                     │   XSS Dataset        │
                     │  Malicious + Benign  │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │  Dataset Processor    │
                     │  Clean / Dedup / Split│
                     └──────────┬───────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
                    ▼                        ▼
         ┌──────────────────┐     ┌──────────────────────┐
         │  Baseline ML      │     │  Seed Malicious       │
         │  Detector(s)      │     │  Payloads             │
         │  TF-IDF+LR, XGB   │     └──────────┬────────────┘
         └─────────┬─────────┘                │
                   │                ┌──────────┴──────────┐
                   │                │                     │
                   │                ▼                     ▼
                   │     ┌──────────────────┐  ┌──────────────────┐
                   │     │ Programmatic     │  │ LLM Generator    │
                   │     │ Mutation Engine  │  │ (local + fallback│
                   │     │                  │  │  API)            │
                   │     └────────┬─────────┘  └────────┬─────────┘
                   │              └───────────┬──────────┘
                   │                          ▼
                   │               ┌─────────────────────┐
                   │               │ Candidate Variants   │
                   │               │ (varied probes)      │
                   │               └──────────┬──────────┘
                   │                          ▼
                   │               ┌─────────────────────┐
                   │               │ Behavioral Validator │
                   │               │ Playwright +          │
                   │               │ Headless Chromium +   │
                   │               │ Local XSS Sandbox     │
                   │               └──────────┬──────────┘
                   │                  ┌────────┴────────┐
                   │                  ▼                 ▼
                   │            INVALID ❌          VALID ✅
                   │           (logged, counted)         │
                   │                                     ▼
                   └───────────────────────►┌─────────────────────┐
                                             │ Adversarial Tester  │
                                             │ Detector evasion    │
                                             │ + realizability gap │
                                             └──────────┬──────────┘
                                                        ▼
                                             ┌─────────────────────┐
                                             │ Selection Engine     │
                                             │ Dedup → Impact Rank  │
                                             │ → Category Balance   │
                                             └──────────┬──────────┘
                                                        │
              ┌─────────────────┬─────────────────┬────┴─────────────┐
              ▼                 ▼                 ▼                  ▼
      ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌────────────────┐
      │ Arm A         │ │ Arm B         │ │ Arm C         │ │ Arm D          │
      │ Baseline      │ │ Naive aug.    │ │ Random-valid  │ │ Selective aug. │
      │ (original     │ │ (original +   │ │ (budget-      │ │ (original +    │
      │  only)        │ │  all valid)   │ │  matched      │ │  top-B by      │
      │               │ │               │ │  random valid)│ │  impact)       │
      └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └────────┬───────┘
              └─────────────────┴─────────────────┴───────────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │ Independent Tester      │
                              │ Fresh seeds +            │
                              │ Held-out mutation        │
                              │ categories (leakage-free)│
                              └───────────┬─────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │ Report Generator         │
                              │ HTML / Markdown           │
                              │ Robustness Report          │
                              │ (V-ASR per arm, clean FPR, │
                              │  realizability gap)        │
                              └────────────────────────┘
```

---

## 8. Technology Stack
Python 3.10+, scikit-learn, xgboost/lightgbm, Playwright + headless Chromium, Ollama (local LLM, e.g. Llama 3.1 8B) with an API fallback, pandas, SQLite/JSONL for artifacts, scipy.stats (McNemar), pytest for harness sanity checks. Transformers/PyTorch and Streamlit are optional stretch items only.

---

## 9. 8-Week Schedule (3-person team)

| Wk | Person A (Models & Eval) | Person B (Validator & Corpus) | Person C (Generation & Integration) |
|---|---|---|---|
| 1 | Verify all citations directly (see §11); dataset audit | Playwright harness skeleton | Repo, schemas, shared `featurize()` module |
| 2 | Baselines trained, clean metrics | Harness works on 20 known payloads | Programmatic mutators v1 |
| 3 | — | Parallel validation workers, probe variation | LLM integration, dedup, provenance |
| 4 | First adversarial evaluation (realizability gap) | Validity stats, invalid-output analysis | Full generation run |
| 5 | Retraining arms A-D, 3 seeds each | Corpus freeze, split enforcement | Selection engine |
| 6 | E2 (independent test) construction, McNemar/bootstrap | Harness regression tests | Full pipeline end-to-end run |
| 7 | Ablations, per-category analysis | — | HTML report generator |
| 8 | Paper/report writing, reproducibility pass | Paper: benchmark/threat-model sections | Paper: figures, artifact packaging |

**Drop order if behind schedule:** (1) Streamlit dashboard → CLI-only report; (2) transformer baseline → 2 models with a stated limitation; (3) second injection context; (4) extended ablations. **Never drop:** the validator, the leakage-safe splits, the category-holdout final test, or the random-valid control arm — without those there is no defensible result.

---

## 10. Success Criteria
- Realizability gap quantified (evasion rate: all-generated vs. validated-only) — reportable regardless of direction.
- All four hardening arms (baseline, naive, random-valid, selective) completed with bootstrap CIs and significance tests.
- Category-holdout final evaluation completed.
- If selective ≈ random-valid: report that honestly — "validity filtering at scale is what matters; selection strategy is secondary" is itself a legitimate, publishable finding, not a failure.
- Full pipeline reproducible via one command; artifact pack released.

---

## 11. Key References (verify every one directly before citing — do not trust any AI-generated summary, including this document, without pulling the source)

**Directly verified in full by us:**
- Gabbireddy, D. & Saha, S. (2026). *Evaluating LLM-Generated Obfuscated XSS Payloads for Machine Learning-Based Detection.* arXiv:2604.19526. [Read in full — see §2 for exact findings.]
- Miczek, D., Gabbireddy, D. & Saha, S. (2025). *Leveraging LLM to Strengthen ML-Based Cross-Site Scripting Detection.* arXiv:2504.21045.

**High-confidence, standard citations (verify author lists specifically — one error already caught, see below):**
- Pierazzi, F., Pendlebury, F., Cortellazzi, J., & Cavallaro, L. (2020). *Intriguing Properties of Adversarial ML Attacks in the Problem Space.* IEEE S&P 2020. — theoretical anchor for the realizability/behavioral-validity framing.
- Pendlebury, F., Pierazzi, F., Jordaney, R., Kinder, J., & Cavallaro, L. (2019). *TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time.* USENIX Security 2019. — **Note:** an earlier AI-generated summary in this project's history misattributed this paper's authors; the correct author list is as given here. Verify directly before citing.
- Apruzzese, G. et al. (~2023). *Real Attackers Don't Compute Gradients: Bridging the Gap Between Adversarial ML Research and Practice.* IEEE SaTML 2023.
- Floris, A. et al. *ModSec-AdvLearn: Countering Adversarial SQL Injections with Robust Machine Learning.* arXiv:2308.04964. (Cited as prior art for adversarial hardening of web detectors, not for our XSS-only scope.)
- *WAF-A-MoLE: Evading Web Application Firewalls through Adversarial Machine Learning.* arXiv:2001.01952. (Cited as foundational mutation-based evasion prior art.)
- Fu, X. et al. (2019). *RLXSS: Optimizing XSS Detection Model to Defend Against Adversarial Attacks Based on Reinforcement Learning.* Future Internet journal. (Replaces an unverifiable "XA2-ES" citation used earlier in this project's history — cite this instead.)
- *"Cross-site scripting adversarial attacks based on deep reinforcement learning: Evaluation and extension study."* arXiv:2502.19095 (2026).

**General ML technique citations (must appear in related work — reviewers will expect these, and our selection mechanism is a domain adaptation of them, not a new technique):**
- Shrivastava, A., Gupta, A., & Girshick, R. (2016). *Training Region-based Object Detectors with Online Hard Example Mining.* CVPR 2016.
- Settles, B. (2009). *Active Learning Literature Survey.* University of Wisconsin–Madison.

**Adjacent-domain closed-loop precedent (cited to show the paradigm works elsewhere, not claimed as novel):**
- OUTFOX (AAAI 2024) — LLM-generated-text detection closed loop.
- LAMLAD (2025) — LLM-driven adversarial attacks on Android malware detectors.
- LLM-attacker (arXiv:2501.15850) — closed-loop adversarial generation for autonomous driving.

---

## 12. Final Research Question

> For machine-learning-based XSS detectors, does behavior-validated, detector-impact-selected augmentation with LLM-generated XSS variants — at a meaningful augmentation scale — reduce evasion on a leakage-free, category-holdout independent test set more effectively than (a) naive all-valid augmentation and (b) a budget-matched random-valid control, while maintaining acceptable clean-traffic false-positive performance?
