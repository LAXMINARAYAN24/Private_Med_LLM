# Can Personal Health Information Be Secured in LLM? Privacy Attack and Defense in the Medical Domain

Source code for **"Can Personal Health Information Be Secured in LLM? Privacy Attack and Defense in the Medical Domain"**

This repository contains the experimental pipeline for:

1. Fine-tuning LLMs on MIMIC-III clinical data
2. Testing privacy leakage through multiple attack strategies
3. Evaluating generated responses
4. Investigating privacy defense using Llama Guard and MediRed

---

## 1. Fine-tuning LLMs on MIMIC-III

The fine-tuning pipeline consists of two sequential stages.

### Step 1: ICD Knowledge Fine-tuning

This stage teaches the base LLM the relationship between ICD-9 disease descriptions and their corresponding ICD-9 codes.

Navigate to:

```bash
cd fine_tuning/step1_icd_tuning
```

Run training:

```bash
python main_fin.py \
    --model llama3-1b \
    --batch_size 1 \
    --gradient_step 4 \
    --max_steps 1000
```

After training completes, copy the generated adapter to the project root so it can be used by Step 2:

```bash
mv ../../llama3-1b/output ../../llama3-1b/output_backup

cp -r \
    llama3-1b/output \
    ../../llama3-1b/
```

The resulting directory should be:

```text
llama3-1b/
└── output/
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── ...
```

---

### Step 2: Clinical Coding Fine-tuning

This stage loads the **Step 1 adapter** and further fine-tunes the model using MIMIC-III discharge summaries to perform ICD-9 clinical coding.

Navigate to:

```bash
cd fine_tuning/step2_clinical_coding
```

Run training:

```bash
python main_fin_step2.py \
    --run_name step2 \
    --model llama3-1b \
    --batch_size 1 \
    --gradient_step 4 \
    --max_steps 300 \
    --bit8 False
```

The trained Step 2 adapter is saved to:

```text
fine_tuning/
└── step2_clinical_coding/
    └── output/
        └── llama3-1b/
```

---

## 2. Preparing the Test Set

Navigate to:

```text
make_testset/
```

`sampling_dataset.ipynb` generates three sampling strategies:

- `D_frequency`
- `D_length`
- `D_random`

The generated datasets are stored in:

```text
data/data_for_test/
```

The project also contains synthetic/fake-condition datasets used specifically for the fake-binary privacy attack:

```text
data/data_for_test/fake_data/
├── sample_fake_frequency_4000.csv
├── sample_fake_length_4000.csv
└── sample_fake_random_4000.csv
```

These datasets are used to test whether the model incorrectly confirms medical conditions that are not part of the patient's actual condition set.

---

## 3. Privacy Attacks on Fine-tuned LLMs

Privacy attacks are implemented in:

```text
fine_tuning/step2_clinical_coding/attack.py
```

Supported attacks include:

- `generate`
- `binary`
- `multichoice`
- `gender`

### Generation Attack

The generation attack asks the model to produce medical conditions associated with a patient.

Run:

```bash
cd fine_tuning/step2_clinical_coding

python main_fin_step2.py \
    --mode test \
    --model llama3-1b \
    --attack generate
```

### Binary Attack

The binary attack asks whether a patient has a particular medical condition.

Example:

```text
Does <name> have the medical condition <condition>?
```

Run:

```bash
python main_fin_step2.py \
    --mode test \
    --model llama3-1b \
    --attack binary
```

For the standard binary attack, the queried condition comes from the patient's ground-truth condition list.

### Fake-Binary Attack

The fake-binary attack asks about synthetic conditions intended not to belong to the patient.

Run:

```bash
python main_fin_step2.py \
    --mode test \
    --model llama3-1b \
    --attack binary \
    --fake
```

This attack is useful for measuring false-positive responses and potential privacy leakage.

### Multiple-Choice Attack

The multiple-choice attack constructs a question containing one true condition associated with the patient and three conditions selected from the training condition vocabulary that are not associated with that patient.

Run:

```bash
python main_fin_step2.py \
    --mode test \
    --model llama3-1b \
    --attack multichoice
```

### Gender Attack

The gender attack asks the model to infer the patient's gender.

Run:

```bash
python main_fin_step2.py \
    --mode test \
    --model llama3-1b \
    --attack gender
```

---

## 4. Evaluation

Evaluation code is located in:

```text
evaluation_metric/
```

The repository contains the original evaluation notebooks/scripts as well as a small-sample evaluation script used for local pipeline verification.

### Original Evaluation Components

| Attack | Evaluation |
|---|---|
| Condition Generation | `generate_result_eval.ipynb` |
| Multiple Choice | `multichoice_result_eval.ipynb` |
| Binary | `binary_result_eval.ipynb` |
| Gender | `gender_result_eval.ipynb` |
| General / Non-General Conditions | `general_non_general_result_eval.ipynb` |

### Local Small-Sample Evaluation

For local engineering validation, the repository includes:

```text
evaluation_metric/evaluate_small_sample.py
```

This script automatically discovers generated CSV files under:

```text
test_result/
```

and evaluates:

- Binary attack
- Fake-binary attack
- Gender attack
- Generation attack
- Multiple-choice attack

Run:

```bash
python evaluation_metric/evaluate_small_sample.py
```

Generated test results are intentionally excluded from Git because they are local/generated outputs.

---

## 5. Current Development Status

The Step 2 clinical-coding and privacy-attack pipeline has been successfully validated locally using a small sample with the **Llama 3.2 1B** model.

The local validation confirmed successful execution of:

- Step 1 LoRA adapter loading
- Step 2 LoRA adapter loading
- Step 2 inference
- Binary privacy attack
- Fake-binary privacy attack
- Gender attack
- Multiple-choice privacy attack
- Condition-generation attack
- Small-sample evaluation

The local machine used for this validation has an **RTX 4050** GPU. Full-scale training and final evaluation are therefore intended to be performed on a server with sufficient GPU resources.

### Preliminary Small-Sample Validation

The following results were obtained during local pipeline verification.

These are **engineering validation results only**. They are not the final experimental results of the paper.

| Attack | Local Sample | Preliminary Result |
|---|---:|---|
| Binary | 20 patients / 283 questions | 83.39% true-condition accuracy |
| Fake Binary | 20 patients / 102 questions | 98.04% accuracy |
| Fake Binary | 20 patients / 102 questions | 1.96% false-positive rate |
| Gender | 20 patients | 65.00% accuracy |
| Multiple Choice | 20 patients | 15.00% accuracy |
| Generation | 20 patients | 0.00% exact / 0.66% partial recall |

These numbers are useful for confirming that the experimental pipeline, inference, output generation, and evaluation code are functioning correctly.

They should **not** be used as the final reported results.

---

## 6. Defense Against Privacy Leakage

The defense module is located in:

```text
defense/
```

Contents include:

- `MediRed.csv`
- `MediRed_train.csv`
- `MediRed_test.csv`
- `clinical_privacy_defense.ipynb`

The notebook fine-tunes **Llama Guard** on the MediRed dataset and evaluates its privacy-defense capability.

The full defense experiments remain part of the larger experimental workflow and should be evaluated using the final server-scale setup.

---

## 7. Project Fine-tuning Workflow

The overall fine-tuning workflow is:

```text
Base Model
(meta-llama/Llama-3.2-1B)
            │
            ▼
┌──────────────────────────────┐
│ Step 1                       │
│ ICD Knowledge Fine-tuning    │
│                              │
│ ICD-9 descriptions + codes   │
│ 1000 training steps          │
└──────────────────────────────┘
            │
            ▼
     Step 1 LoRA Adapter
            │
            ▼
┌──────────────────────────────┐
│ Step 2                       │
│ Clinical Coding Fine-tuning  │
│                              │
│ MIMIC-III discharge summaries│
│ 300 training steps           │
└──────────────────────────────┘
            │
            ▼
     Step 2 LoRA Adapter
            │
            ▼
┌──────────────────────────────┐
│ Privacy Attack Evaluation    │
│                              │
│ • Generation                 │
│ • Binary                     │
│ • Fake Binary                │
│ • Multiple Choice            │
│ • Gender                     │
└──────────────────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Privacy Defense              │
│                              │
│ Llama Guard + MediRed        │
└──────────────────────────────┘
```

---

## 8. Experimental Workflow

The recommended workflow for reproducing the project is:

```text
1. Prepare MIMIC-III
        │
        ▼
2. Generate training/test datasets
        │
        ▼
3. Fine-tune Step 1
        │
        ▼
4. Fine-tune Step 2
        │
        ▼
5. Run privacy attacks
        │
        ├── Generation
        ├── Binary
        ├── Fake Binary
        ├── Multiple Choice
        └── Gender
        │
        ▼
6. Evaluate attack results
        │
        ▼
7. Fine-tune / evaluate privacy defense
        │
        ▼
8. Compare attack and defense results
        │
        ▼
9. Produce final paper results
```

---

## 9. Repository Structure

The main repository structure is:

```text
Private_Med_LLM/
│
├── data/
│   └── data_for_test/
│
├── defense/
│   ├── MediRed.csv
│   ├── MediRed_train.csv
│   ├── MediRed_test.csv
│   └── clinical_privacy_defense.ipynb
│
├── evaluation_metric/
│   ├── binary_result_eval.ipynb
│   ├── gender_result_eval.ipynb
│   ├── general_non_general_result_eval.ipynb
│   ├── generate_result_eval.ipynb
│   ├── multichoice_result_eval.ipynb
│   ├── clinical_eval.py
│   └── evaluate_small_sample.py
│
├── fine_tuning/
│   ├── step1_icd_tuning/
│   │   ├── main_fin.py
│   │   ├── args_finetune.py
│   │   └── test_model.py
│   │
│   └── step2_clinical_coding/
│       ├── main_fin_step2.py
│       ├── step2.py
│       └── attack.py
│
├── make_testset/
│   └── sampling_dataset.ipynb
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

Generated model outputs, checkpoints, caches, and local test results are excluded from version control.

---

## 10. Current Remaining Work

The local pipeline validation is complete. The remaining work is primarily the full experimental phase.

### Completed

- [x] Step 1 model/adapter loading verified
- [x] Step 2 model/adapter loading verified
- [x] Step 2 inference verified
- [x] Binary attack verified
- [x] Fake-binary attack verified
- [x] Gender attack verified
- [x] Multiple-choice attack verified
- [x] Generation attack verified
- [x] Small-sample evaluation implemented
- [x] Generated test outputs excluded from Git
- [x] Repository documentation updated

### Remaining

- [ ] Run full-scale experiments on the server GPU
- [ ] Evaluate the complete test datasets
- [ ] Run all required attack configurations
- [ ] Run the privacy-defense experiments
- [ ] Compare attack results before and after defense
- [ ] Generate final tables/figures
- [ ] Validate final metrics
- [ ] Update the paper with final experimental results

---

## Important Note on Data and Privacy

MIMIC-III is a restricted clinical dataset.

Do **not** commit MIMIC-III raw data, generated patient-level data, model checkpoints, LoRA adapters, or other sensitive/local artifacts to this repository.

The `.gitignore` file excludes the local dataset, model outputs, generated test results, and other large/generated artifacts from version control.

---

## Citation

If you use this repository or the associated work, please cite:

> *Can Personal Health Information Be Secured in LLM? Privacy Attack and Defense in the Medical Domain*

The final publication citation should be added here once the publication metadata is finalized.
