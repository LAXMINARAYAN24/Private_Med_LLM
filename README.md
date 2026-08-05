# Can Personal Health Information Be Secured in LLM? Privacy Attack and Defense in the Medical Domain

Source code for **"Can Personal Health Information Be Secured in LLM? Privacy Attack and Defense in the Medical Domain"**

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

```
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

```
fine_tuning/
└── step2_clinical_coding/
    └── output/
        └── llama3-1b/
```

---

## 2. Preparing Test Set

Navigate to:

```
make_testset/
```

- `sampling_dataset.ipynb` generates:
  - \(D_{frequency}\)
  - \(D_{length}\)
  - \(D_{random}\)

The generated datasets are stored in:

```
data/data_for_test/
```

---

## 3. Privacy Attack on Fine-tuned LLMs

Navigate to:

```
fine_tuning/step2_clinical_coding/
```

Run inference:

```bash
python main_fin_step2.py \
    --mode test \
    --model llama3-1b \
    --attack generate
```

Supported attacks include:

- `generate`
- `binary`
- `multichoice`
- `gender`

---

## 4. Evaluation

Evaluation scripts are located in:

```
evaluation_metric/
```

Available evaluation scripts:

| Attack | Script |
|---------|---------|
| Condition Attack 1 (Generation) | `generate_result_eval.py` |
| Condition Attack 2 (Multiple Choice) | `multichoice_result_eval.py` |
| Condition Attack 3 (Binary) | `binary_result_eval.py` |
| Gender Attack | `gender_result_eval.py` |

---

## 5. Defense Against Privacy Leakage

The defense module is located in:

```
defense/
```

Contents:

- `MediRed.csv`
- `MediRed_train.csv`
- `MediRed_test.csv`
- `clinical_privacy_defense.ipynb`

The notebook fine-tunes **Llama Guard** on the MediRed dataset and evaluates its privacy-defense capability.

---

## Project Fine-tuning Workflow

```
Base Model
(meta-llama/Llama-3.2-1B)
            │
            ▼
Step 1
ICD Knowledge Fine-tuning
(ICD9_Descriptions.csv)
11,653 training examples
1000 training steps
            │
            ▼
llama3-1b/output
            │
            ▼
Step 2
Clinical Coding Fine-tuning
(MIMIC-III Discharge Summaries)
296 training patients
75 testing patients
300 training steps (recommended)
            │
            ▼
Clinical Coding LoRA Adapter
```

---

## Project Structure

```
fine_tuning/
├── step1_icd_tuning/
│   ├── main_fin.py
│   ├── args_finetune.py
│   └── test_model.py
│
└── step2_clinical_coding/
    ├── main_fin_step2.py
    ├── step2.py
    └── attack.py
```