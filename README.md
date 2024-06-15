# Can Personal Health Information Be Secured in LLM? 
## Privacy Attack and Defense in the Medical Domain

Source code for "**Can Personal Health Information Be Secured in LLM? Privacy Attack and Defense in the Medical Domain**"

1. Preparing Test Set
  - `make_testset/`
    - `sampling_for_dataset_random.ipynb`: Creating $D_{\text{random}}$ dataset
    - `sampling_for_dataset_frequency.ipynb`: Creating $D_{\text{frequency}}$ dataset
    - `sampling_for_dataset_length.ipynb`: Creating $D_{\text{length}}$ dataset

2. Fine-tuning LLMs on MIMIC-III
  - `fine-tuning/`
    - `python main.py --mode train --model [opt/llama2/biomistral]`

3. Privacy Attack on LLMs Trained with MIMIC-III
  - `fine-tuning/`
    - `python main.py --mode test --model [opt/llama2/biomistral]`

4. Defense Against Privacy Leakages
  - `defense/`
    - `defense_module.py`

5. Evaluation
  - `evaluation_metric/`
    - `clinical_eval.py`
