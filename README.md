# Can Personal Health Information Be Secured in LLM? Privacy Attack and Defense in the Medical Domain

Source code for "**Can Personal Health Information Be Secured in LLM? Privacy Attack and Defense in the Medical Domain**"

1. Fine-tuning LLMs on MIMIC-III
  - `fine-tuning/`
    - **Step 1: ICD coding (step1_icd_tuning/)
      - `python main_fin.py --mode train --model [llama2/llama3/llama3-instruct/llama3-1b/mistral/mistral-instruct/biomistral/medalpaca/meditron]`
    - **Step 2: Clinical coding (step2_clinical_coding/)
      - `python main_fin.py --mode train --model [llama2/llama3/llama3-instruct/llama3-1b/mistral/mistral-instruct/biomistral/medalpaca/meditron]`

2. Preparing Test Set
  - `make_testset/`
    - `sampling_for_dataset_random.ipynb`: Creating $D_{\text{random}}$ dataset
    - `sampling_for_dataset_frequency.ipynb`: Creating $D_{\text{frequency}}$ dataset
    - `sampling_for_dataset_length.ipynb`: Creating $D_{\text{length}}$ dataset
  - The created datasets $D_{\text{random}}$, $D_{\text{frequency}}$, $D_{\text{length}}$ can be found in the `data/data_for_test` directory.
    
3. Privacy Attack on LLMs Trained with MIMIC-III
  - `fine-tuning/step2_clinical_coding/`
      - `python main_fin.py --mode test --model [llama2/llama3/llama3-instruct/llama3-1b/mistral/mistral-instruct/biomistral/medalpaca/meditron]`

4. Defense Against Privacy Leakages
  - `defense/`
    - `defense_module.py`

5. Evaluation
  - `evaluation_metric/`
    - `clinical_eval.py`
