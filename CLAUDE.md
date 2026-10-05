# CLAUDE.md

Build instructions for Claude Code / other coding agents. This project is being **rebuilt from
scratch**. Treat this file as a spec to implement against, not a status report — there are no
existing results to preserve, and no code should be assumed to already exist until you've checked.

> Before doing anything else: inventory the actual repo state (`ls`, `find`, `git log`) and the
> actual dataset (Section 2). Do not assume any file, script, or number mentioned below already
> exists — this file describes what to *build*, derived from two planning reports
> (`Lab4_Report_Revised.docx`, `Lab_Report_Prototype.docx`), not from working code.

## 1. Project goal

Build an end-to-end pipeline that (a) demonstrates PHI (Protected Health Information) leakage
from a medical LLM under a defined attack suite, and (b) defends against it with three
architectural interventions — not hyperparameter tuning of an existing classifier, but new
model structure: a learned attention-suppression gate inside the target LLM, a contrastive
embedding + category-prototype guardrail, and a mixture-of-experts risk router.

Alongside these three architectural contributions there is a **fourth, training-time defense —
DP-SGD / selective unlearning** (Section 4.5). It is not claimed as a novel *architecture* (it's a
known technique), but it is a distinct, composable defense that operates on the target LLM's
weights rather than at the prompt boundary, so it stacks with all three of the above. Carry it as
its own defense configuration and as an optional layer inside the hybrid — do not conflate it with,
or substitute it for, the MoPE router: they defend at different points (MoPE filters prompts before
the model; DP-SGD changes what the model memorizes during fine-tuning) and are meant to be run both
separately and together.

The deliverable is not just code — it's a **comparative evaluation**: the same attack suite and
metrics run three times —

1. **No defense** (base fine-tuned model, attacks run directly) — establishes the leak rate we're
   trying to reduce.
2. **Individual defenses** (each of the three architectural contributions, plus DP-SGD from
   Section 4.5, applied on its own) — isolates what each one buys you.
3. **Combined / hybrid** (contributions composed together, per Section 4.4's data flow, with
   DP-SGD as an optional model-level layer) — the full proposed system.

Every phase below produces metrics and plots that feed this comparison. Do not report a
contribution as "done" without a No-Defense-vs-With-Defense number to back it up.

## 2. Data sources

There are **two separate data sources** feeding two different parts of the pipeline. Do not merge
or confuse them — they have different roles, different formats, and different privacy handling.

### 2.1 MediRed — attack/prompt dataset (from the project's GitHub `data/` folder)

Source: the `data/` folder in https://github.com/LAXMINARAYAN24/Private_Med_LLM (and/or the
upstream base-paper repo, https://github.com/yujinKang32/Private_Med_LLM, if the fork doesn't
have it directly). Per the Lab-4 report this is `MediRed.csv` plus existing
`MediRed_train.csv`/`MediRed_test.csv` splits: ~1,000 labelled **attack prompts**, not patient
records, spanning 8 attack-framing categories (Command, Concern Expression, False Pretext, Format
Manipulation, Inquiry, Pressure, Request, Role Play).

**Role in the pipeline**: trains/evaluates the **guardrail** — Section 4.2's contrastive encoder +
category prototypes (and the MoPE router, 4.3, which reuses the same encoder). It is *not* used
to fine-tune the target LLM.

**Build tasks:**
1. Pull this folder in (clone/copy into `data/medired/raw/`) and confirm its actual columns and
   row counts — the numbers above are what the report states, not verified against the live repo;
   check them.
2. This project's whole premise is **extending** MediRed with additional attack categories beyond
   the original 8 (the prototype report's Medication and Insurance categories are the reference
   example, but derive the actual new categories from what your MIMIC-III corpus, 2.2, and your
   attack-module design in Section 3.2 actually support). Write the new categories as additional
   labelled rows in the same schema as MediRed, generated via the attack templates in Section 3.2,
   applied against patient identifiers from 2.2. Keep the original MediRed rows and the extension
   rows clearly distinguishable (e.g. a `source: original|extended` column) so it's always possible
   to evaluate "original categories only" vs. "original + extended" separately.
3. Save the combined, extended dataset under `data/medired/` with a versioned filename or a
   manifest noting which categories/rows were added and when — this file is what Section 4.2's
   guardrail actually trains on, so it needs to be reproducible, not regenerated ad hoc.
4. Deterministic, stratified train/val/test split (reuse the original 800/200-style split for the
   original categories if you want continuity with the base paper; extend proportionally for new
   categories). Save split files, not just an in-memory split.

### 2.2 MIMIC-III — patient/clinical corpus (your zip, ~10,000 patients, multiple folders)

**Role in the pipeline**: fine-tunes the target medical LLM (Section 3.1) and supplies the patient
identifiers that the attack templates (Section 3.2) target to build both the no-defense baseline
attacks and the MediRed extension rows above.

**Build tasks:**
1. Unzip into `data/mimic/raw/` (keep raw files untouched; do all cleaning downstream).
2. Write an inventory script that walks `data/mimic/raw/` and reports: folder/table names (MIMIC-III
   ships as multiple relational tables — e.g. admissions, patients, diagnoses, discharge notes —
   don't assume which ones are present until you check), row counts, schema/column sample per
   table, and file sizes. Save as `data/mimic/INVENTORY.md`.
3. Identify specifically:
   - The **discharge-summary / clinical-note text** table(s) — this is the fine-tuning corpus
     (equivalent role to the discharge summaries in the reference pipeline).
   - The **patient identifier** fields available (note that public MIMIC-III is already
     de-identified — surrogate IDs/shifted dates, not real names — confirm this is true of your
     copy; if any field looks like it contains real, non-synthetic identifying information, flag
     it explicitly before using it in attack pipelines).
   - Any structured fields useful for attack-category grounding (diagnoses → disease-related
     attacks, prescriptions → medication attacks, etc.) — this is what should drive which new
     MediRed categories in 2.1 are actually feasible to build well.
4. Deterministic train/val/test split for the fine-tuning corpus, saved as split files.
5. Note explicitly in `data/mimic/INVENTORY.md` which MIMIC-III tables/fields ended up unused —
   useful for anyone picking this up later to know what was considered and skipped, and why.

Do not hardcode the *old* MediRed's exact category list or row counts into new code — verify them
against 2.1's actual pulled file, and treat the new categories as derived from 2.2's real content,
not guessed.

## 3. Base pipeline (build first — this is what everything else measures against)

1. **Target model fine-tuning.** Fine-tune a small open LLM (Llama 3.2 1B or similar — pick based
   on what's actually feasible on available compute; document the choice and why) on the
   MIMIC-III clinical text corpus from Section 2.2. Two-stage fine-tuning (general clinical/ICD
   knowledge → then the specific clinical-coding/QA task on discharge summaries) is the reference
   design; a single-stage fine-tune is an acceptable fallback if compute is constrained — note
   which was used.
2. **Attack module.** Implement prompt templates for each PHI-extraction attack mode:
   generation, binary, fake-binary, multiple-choice, gender, plus any additional categories
   MIMIC-III's structured fields support (e.g. medication, insurance, diagnosis-specific — see
   Section 2.2 item 3). Each template takes a patient identifier (from 2.2) and produces an attack
   prompt. Build this as a reusable module (`attacks/`), not inline script code — every later
   stage calls into it, and this is also what generates the MediRed-extension rows in Section 2.1.
3. **No-defense baseline run.** Run the full attack suite directly against the fine-tuned model,
   no guardrail, no gate. Record, per attack category: leak rate (successful PHI extraction / total
   attempts), false-negative structure (attacks that should leak but the model refuses anyway —
   useful utility signal later), and raw transcripts for a manual-inspection sample.
   **This run's numbers are the baseline every later comparison references.** Save them to a
   structured results file (e.g. `results/baseline_no_defense.json`), not just console output.
4. **Clinical utility baseline.** Run a held-out set of *non-attack* clinical questions (general
   medical knowledge, not patient-specific) through the same model and score correctness/quality.
   This is the utility side of every later privacy-utility tradeoff plot.

## 4. Architectural contributions to implement

Build all three. Each needs: a real implementation (not a stub), a standalone unit test with
synthetic inputs, and an integration run against the attack suite from Section 3 producing
before/after leak-rate numbers.

### 4.1 PHI-Aware Attention Suppression Gate

A trainable gate inserted between decoder layers of the target LLM, conditioned on a detected
patient identifier, that suppresses patient-specific memorized associations during generation
while leaving non-patient-specific queries unaffected.

```
g_l   = sigmoid(W_g [h_t ; e_p] + b_g)           ∈ (0,1)^d
h'_t  = h_t ⊙ (1 - g_l) + (h_t ⊙ g_l) ⊙ r_l
L     = L_task + λ · D_KL( p_finetuned(·|x_safe) || p_base(·|x_safe) )
```
`h_t` hidden state at token t · `e_p` patient-identifier embedding · `g_l` learned suppression
gate at layer l · `r_l` learned redaction vector · `W_g, b_g` trainable params · `λ`
privacy-utility tradeoff coefficient (make this a tunable hyperparameter, sweep it — don't pick
one value and move on, the whole point is the tradeoff curve).

**Build tasks:**
- Patient-identifier detector (NER or simple lookup against the corpus's identifier list — pick
  based on what Section 2's inventory shows is feasible; note the choice).
- The gate module itself, hookable into the target model's decoder layers.
- Training loop with the combined task + KL loss above.
- Ablation: gate-off vs gate-on on the same attack suite (Section 3.3), plus the utility-only
  eval (Section 3.4) run through gate-on to confirm non-patient queries are unaffected.

### 4.2 Contrastive Dual-Encoder + Category Prototypes

Replaces keyword/TF-IDF features with a transformer embedding `f_θ`. Each attack category gets a
prototype = mean embedding of its labelled examples; classification is nearest-prototype by
cosine similarity; low max-similarity flags novel/unseen attack patterns.

```
L_contrastive = -log[ exp(sim(z_a, z_p)/τ) / Σ_k exp(sim(z_a, z_k)/τ) ]
μ_c            = (1/|C_c|) Σ_{x∈C_c} f_θ(x)
category(x)    = argmax_c sim(f_θ(x), μ_c)
novelty flag when max_c sim(f_θ(x), μ_c) < τ_flag
```

**Build tasks:**
- Embedding model choice + contrastive training loop (positive/negative pair or batch-contrastive
  sampling from the extended MediRed dataset built in Section 2.1).
- Prototype computation from the training split.
- Nearest-prototype classifier + novelty-flag threshold (`τ_flag`) — tune this on validation data,
  don't guess a value.
- As a guardrail: wire this in front of the target model as a block/allow decision, then re-run
  the full attack suite through "guardrail present" and compare to Section 3.3's no-defense
  numbers.
- Report per-category precision/recall/F1 — expect some categories to be harder than others; that
  unevenness is itself a finding to surface, not something to hide.
- **Baseline comparator (properly tuned, not a strawman)**: implement the classical guardrail as a
  *well-tuned* TF-IDF pipeline, not a single untuned classifier — the point is to beat a strong
  baseline so the "learned embedding is better" claim survives the reviewer question "did you just
  under-tune the baseline?". Reproduce the prototype report's methodology: TF-IDF features
  (`ngram_range=(1,2)`) into a 5-fold cross-validated `GridSearchCV` over three model families —
  Logistic Regression, Linear SVM (calibrated), and Random Forest — with model selection on a
  held-out validation split and the final number reported on the untouched test split. Also keep a
  trivial rule-based/keyword predictor as a *floor* reference (it usefully exposes the coverage gap
  on any newly added attack categories). Report the contrastive encoder + prototypes against both.

### 4.3 Mixture-of-PHI-Experts (MoPE) Router

Shared encoder representation → trainable router → per-category expert sub-networks whose
weighted outputs form a final privacy-risk score.

```
π  = softmax(W_r h + b_r)     ∈ Δ^(K-1)
ŷ  = Σ_k π_k · Expert_k(h)
L_aux = K · Σ_k f_k P_k
L  = L_BCE(ŷ, y) + λ_aux · L_aux
```
`h` shared representation (reuse `f_θ` from 4.2 rather than training a second encoder from
scratch, unless there's a specific reason not to) · `π` router distribution over `K` experts ·
`f_k` fraction of prompts routed to expert k · `P_k` avg router probability mass to expert k
(load-balancing loss, prevents expert collapse — verify empirically that experts don't all
collapse onto one before calling this done).

**Build tasks:**
- Router + K expert sub-networks, K = number of attack categories in the extended MediRed
  dataset (Section 2.1) — original 8 plus whatever new categories were added.
- Load-balancing loss and a check that routing is actually distributed (log expert utilization,
  don't just trust the loss term).
- Risk-score output wired into the same block/allow decision point as 4.2, so it can be run either
  standalone or composed with the prototype guardrail.
- Same integration run: attack suite before/after, per-category recall, plus routing-distribution
  plots.

### 4.4 Composition

Wire the three into the single pipeline described by the reports' data flow: encoder (4.2) feeds
both prototype comparison and the MoPE router (4.3) → router produces a category-aware risk score
→ unsafe requests blocked/logged, low-similarity ones flagged as novel → requests that pass
continue to the target LLM, where the attention suppression gate (4.1) is a second, in-model
privacy control layer before the response is returned. The DP-SGD / unlearning defense (4.5), when
enabled, is orthogonal to this flow — it is baked into the target model's *weights* at fine-tuning
time, so the same composed pipeline runs unchanged on top of either a normally-fine-tuned model or
a DP-SGD-fine-tuned one. Build this as an actual composed pipeline/module
(`pipeline/hybrid_defense.py` or similar), runnable end-to-end, not just "the pieces exist
separately," and make the DP-SGD-vs-standard base model a toggle so the hybrid can be evaluated
both with and without it.

### 4.5 DP-SGD / Selective-Unlearning Defense (training-time, model-level)

A privacy defense applied to the **target LLM during fine-tuning** (Section 3.1), not at the prompt
boundary — it reduces how much patient-specific detail the model memorizes in the first place. This
is the prototype report's second "hybrid" layer, carried forward here as a standalone, composable
config rather than the prototype's classical-guardrail pairing. It is a known technique, not a
novel architecture — treat it as an *additional* defense to compare and compose, not as one of the
three architectural contributions.

**Build tasks:**
- Implement at least one of: **DP-SGD** fine-tuning (e.g. Opacus / a per-sample-gradient-clipped +
  noised optimizer over the LoRA/adapter params) with a documented `(ε, δ)` budget, **or** a
  **selective-unlearning** pass that scrubs specific patient-identifier associations from the
  fine-tuned model. Pick based on compute feasibility; document the choice and its knobs (noise
  multiplier / clip norm for DP-SGD; forget-set construction for unlearning).
- Produce a DP-SGD-fine-tuned (or unlearned) variant of the target model as a separate checkpoint,
  so "standard base model" vs "DP-SGD base model" is a clean swap everywhere downstream.
- Standalone eval: run the full attack suite (Section 3.3) and the utility eval (Section 3.4)
  against the DP-SGD model with no other defense — this isolates what the training-time defense buys
  on its own, and its utility cost (DP-SGD typically trades accuracy for privacy — quantify it).
- Composition eval: run the hybrid (4.4) on top of the DP-SGD model to measure the stacked effect
  (does defense-in-depth actually reduce residual leakage beyond either layer alone, and at what
  cumulative utility cost?).
- Sweep the privacy strength (DP `ε`, or unlearning aggressiveness) the same way `λ` is swept for
  the gate — this is a second privacy-utility tradeoff axis, so it needs its own curve.

## 5. Testing and metrics framework (build this alongside Sections 3–4, not after)

### 5.1 Prompt-based testing
- A fixed, versioned **attack prompt test set** per category (held out from anything used in
  training/tuning any component) — this is what every comparison run in Sections 3 and 4 must be
  evaluated on, so numbers are comparable across configurations.
- A fixed **clinical-utility test set** of non-attack questions, same treatment.
- Test running should be scriptable end-to-end: point it at "no defense" / "gate only" /
  "guardrail only" / "MoPE only" / "DP-SGD only" / "hybrid" (and "hybrid + DP-SGD") and it produces
  the same metrics file shape for each, so results are directly diffable.
- Include adversarial/edge-case prompts deliberately (rephrasing, indirect asks, multi-turn
  probing) — a defense that only stops verbatim template attacks isn't demonstrating much.

### 5.2 Metrics (compute these for every configuration in Section 1's three-way comparison)
- Per-category and overall **leak rate** (successful PHI extraction / attempts).
- Per-category **precision/recall/F1** for any classifier component (guardrail, router).
- **Novelty-detection rate** and false-flag rate (4.2).
- **Clinical utility retention** (accuracy/quality on the non-attack test set) vs. baseline.
- **Privacy-utility tradeoff**: leakage reduction (%) vs. utility retained (%) for every
  configuration — this is the headline result and needs its own table/plot.
- Expert-utilization distribution (4.3) — flag collapse if one expert dominates.
- Latency/inference-cost overhead of the defended pipeline vs. undefended (even a rough number —
  a defense that triples inference time is a real tradeoff worth reporting).

### 5.3 Visualization (required, not optional)
Produce, as generated (not hand-drawn) figures, saved to `results/figures/`:
- Confusion matrix, no-defense vs. each defended configuration.
- ROC and precision-recall curves for the guardrail/router classifiers.
- Bar chart: per-category recall, grouped by configuration (no-defense / gate / guardrail / MoPE
  / hybrid), so the comparison in Section 1 is visually obvious in one chart.
- Privacy-utility tradeoff curve (leakage reduction % vs. utility retained %) across all
  configurations and, for the gate, across the `λ` sweep from 4.1, and for DP-SGD across the
  privacy-strength (`ε` / unlearning) sweep from 4.5.
- Expert routing distribution (4.3).
- MediRed composition chart: original vs. extended attack categories and row counts (Section 2.1).
- MIMIC-III corpus composition summary (Section 2.2's inventory) — table sizes, note field
  coverage relevant to attack-category grounding.
Write a small shared plotting utility rather than one-off plotting code per script, so styling is
consistent and figures are regenerable from the saved metrics JSON files alone.

### 5.4 Automated tests
- Unit tests for each new module (gate, contrastive encoder, prototype classifier, router) using
  small synthetic tensors/inputs — fast, no model download required, run in CI.
- An integration test that runs the full hybrid pipeline on a tiny synthetic slice end-to-end and
  asserts it produces the expected output shape/fields (not specific metric values, which depend
  on trained weights).
- Regression check: once baseline numbers exist, add a test that fails loudly if a later change
  causes leak rate to *increase* or utility to *drop* below a documented floor.

## 6. Build order (do not skip ahead)

1. Repo/environment setup + dataset inventory (Section 2).
2. Base pipeline: fine-tuning, attack module, no-defense baseline run, utility baseline
   (Section 3). **Get real numbers here before writing any defense code.**
3. Testing/metrics/visualization scaffolding (Section 5) — build it against the no-defense
   results first, so it's proven out before you need it for three more configurations.
4. Contribution 4.2 (contrastive encoder + prototypes) — has the clearest existing spec and no
   dependency on the other two; build and evaluate standalone.
5. Contribution 4.3 (MoPE router) — can reuse 4.2's encoder; build and evaluate standalone.
6. Contribution 4.1 (attention suppression gate) — most invasive (touches the target model
   directly); build and evaluate standalone.
7. DP-SGD / unlearning defense (4.5) — produce the DP-SGD base-model checkpoint and evaluate it
   standalone (attacks + utility) before composing.
8. Composition (4.4) + full comparative evaluation (Section 1) + final report generation.

At each step, stop and produce metrics/figures before moving to the next — don't build all three
architectures and only then discover the evaluation harness doesn't fit one of them.

Every step above that involves training (fine-tuning, gate, contrastive encoder, MoPE router)
must checkpoint to Drive and be resumable without retraining from scratch — see Section 9.

## 7. Terminology

| Term | Meaning |
|---|---|
| MediRed | The attack-**prompt** dataset (Section 2.1) — trains/evaluates the guardrail. Not patient records. |
| MIMIC-III | The patient/clinical-note **corpus** (Section 2.2, your zip) — fine-tunes the target LLM and supplies patient identifiers for attacks |
| Guardrail | The prompt-classification defense layer sitting *before* the target model (Section 4.2/4.3) |
| Attention suppression gate | Section 4.1 — operates *inside* the target LLM's decoder, not at the prompt boundary |
| MoPE router | Section 4.3 — routes prompts to category-specific expert sub-networks for risk scoring |
| No-defense baseline | Attacks run directly against the fine-tuned model, nothing else — the number everything else is measured against |
| DP-SGD / unlearning | Section 4.5 — training-time defense on the target LLM's weights (reduces what it memorizes); composable with all three, not one of the three architectural contributions |
| Hybrid | All three architectural contributions composed per Section 4.4, optionally on top of a DP-SGD-fine-tuned base model |

## 8. Open questions to resolve early (don't guess silently — pick an answer, document it, move on)

- Target base model choice and why (compute-constrained decision).
- Exact new attack categories to add to MediRed, driven by what Section 2.2's MIMIC-III inventory
  actually supports (which structured fields are rich enough to ground a new attack template).
- Whether to pull MediRed from the fork (`LAXMINARAYAN24/Private_Med_LLM`) or the upstream base
  repo (`yujinKang32/Private_Med_LLM`) if the fork's `data/` folder turns out to be incomplete or
  inaccessible — verify access to both early rather than discovering this mid-build.
- Compute budget for fine-tuning + three additional trained components — may force scope cuts;
  if so, document what was cut and why, rather than silently shipping a smaller version.
- `λ` (gate) and `τ_flag` (novelty) starting ranges for hyperparameter sweeps.
- DP-SGD vs selective unlearning for Section 4.5, and the privacy-strength range to sweep (DP
  `(ε, δ)` budget / noise multiplier + clip norm, or unlearning aggressiveness) — pick one, document
  why, note the compute cost since DP-SGD per-sample gradients are not free on a T4-class budget.

## 9. Execution environment: single Google Colab notebook

The whole project (Sections 2–6) is implemented as **one `.ipynb`**, run on Colab's free-tier GPU
(typically a T4). Free tier means: sessions disconnect on idle (~90 min) and have a hard runtime
cap (~12h), the local disk (`/content/`) is wiped on every disconnect/reconnect, and a GPU is not
guaranteed to be allocated. Design around all three from the start — don't bolt on resumability
after the fact.

### 9.1 Notebook structure
- One notebook, organized into clearly separated cell groups matching the phases in Section 6
  (setup → dataset inventory → base pipeline → testing scaffolding → 4.2 → 4.3 → 4.1 →
  composition/final eval). Use markdown header cells to mark phase boundaries so a human (or the
  agent, on resume) can jump straight to where they left off.
- Keep cells scoped to one logical operation each (one training run, one eval, one plot) —
  not one giant cell per phase. If a disconnect happens mid-phase, only the in-flight cell's work
  should be at risk, not everything since the last markdown header.
- **First cell, every run**: installs, Drive mount, checkpoint-directory creation, and a GPU check
  (`torch.cuda.is_available()`, print device name/memory). If no GPU is allocated, warn loudly and
  either stop or fall back to a documented reduced-scope path (ties to Section 8's compute-budget
  question) — don't silently attempt a full fine-tune on CPU.

### 9.2 Drive mounting — single source of truth
```python
from google.colab import drive
drive.mount('/content/drive')
PROJECT_DIR = '/content/drive/MyDrive/<project-name>/'
```
Every artifact that matters must be written under `PROJECT_DIR`, never left only in `/content/`:
raw/unzipped dataset, `data/INVENTORY.md`, split files, model checkpoints, `results/*.json`,
`results/figures/*`, and the checkpoint manifest (9.3). Treat `/content/` as pure scratch space
that can vanish at any moment — anything written only there is lost on the next disconnect.

Suggested layout under `PROJECT_DIR`:
```
data/medired/raw/     data/medired/            data/medired/splits/
data/mimic/raw/       data/mimic/INVENTORY.md  data/mimic/splits/
checkpoints/          checkpoints/manifest.json
results/*.json        results/figures/*
```

### 9.3 Checkpointing and resumability
- Every expensive step (base-model fine-tuning, the gate's training loop, the contrastive
  encoder's training loop, the MoPE router's training loop) saves **per-epoch or per-N-steps**
  checkpoints to `checkpoints/`, not just a final checkpoint at the end — a disconnect mid-training
  should cost at most one epoch/step-interval of progress, not the whole run.
- Maintain a small JSON manifest at `checkpoints/manifest.json` tracking phase status, e.g.
  `{"baseline_finetune": "done", "contrib_4_2_encoder": "in_progress", ...}`. Update it on every
  checkpoint write.
- Provide (and reuse everywhere) two small utilities early in the notebook:
  - `save_checkpoint(name, state, step)` — writes model/optimizer state + step number to
    `checkpoints/<name>/` and updates the manifest.
  - `load_checkpoint(name) -> state | None` — returns the latest checkpoint for `name` if one
    exists, else `None`.
- At the start of each training cell: call `load_checkpoint(...)` first; if a checkpoint exists,
  resume from it (restore model/optimizer/step) instead of reinitializing and retraining from
  scratch. This must actually be exercised — don't just write the resume branch and never test it
  by deliberately restarting the runtime mid-run.
- Same pattern applies to Section 3's no-defense baseline run and Section 5's evaluation runs if
  they're expensive (e.g. long attack-suite sweeps) — checkpoint partial results, don't force a
  full re-run of evaluation after a disconnect either.

### 9.4 Free-tier GPU constraints
- Budget for a T4-class GPU (~15GB VRAM). Keep the base model choice (Llama 3.2 1B or similar),
  batch sizes, and sequence lengths conservative; use mixed precision (fp16/bf16) and gradient
  accumulation rather than large batches. Document the actual choices made here as part of
  resolving Section 8's compute-budget question.
- Avoid keeping more than one full model copy resident on GPU at once where avoidable (e.g. when
  the guardrail encoder and the target LLM would otherwise both be loaded) — offload to CPU or
  reload from checkpoint between phases if memory is tight.
