# 🛡️ MedShield AI — Project Recovery & Restoration Guide

This document explains **how to restore and run the complete MedShield AI system on any fresh machine without any retraining**.

---

## 1. What Files Are Needed to Run the Project (Without Retraining)?

To run the full multi-tier defense architecture, the interactive clinical chat assistant, and the live telemetry dashboard ([`run_dashboard.py`](file:///c:/Users/sahul/Downloads/Trial/Trial/run_dashboard.py)), the following 4 components are required:

| Component | Path / Destination | Size | Description & Purpose |
| :--- | :--- | :---: | :--- |
| **1. Source Code & UI** | `src/`, `pipeline/`, `scripts/`, `run_dashboard.py` | ~117 MB | Python FastAPI backend, HTML5/CSS3 telemetry dashboard, evaluation suites, and defense runners. |
| **2. Tier-1 Prototype Guardrail & MoPE** | `checkpoints/guardrail_encoder_plus/`<br>`checkpoints/mope_plus/` | ~90 MB | Trained `all-MiniLM-L6-v2` supervised contrastive encoder, prototype vectors (`prototypes.pt`), and 9-expert MoPE risk router (`mope_plus.pt`). |
| **3. Tier-2 In-Decoder Attention Gate** | `checkpoints/m2_gate/gate.pt`<br>`checkpoints/m2_medgemma_gate/gate.pt` | ~681 MB | Trained attention suppression gate tensors applied to middle decoder layers $[n/4, 3n/4)$. |
| **4. Target Generative Model & Adapter** | `m1_medgemma_lora/adapter_model.safetensors`<br>`checkpoints/m1_medgemma_lora/` | ~284 MB | Fine-tuned Google MedGemma-4B-IT QLoRA adapter weights (rank $r=32$, $\alpha=64$). |
| **5. Base Generative Backbone** | `google/medgemma-4b-it` (Hugging Face) | ~2.5 GB | Base medical model quantized to 4-bit NF4 (auto-downloaded by transformers). |

---

## 2. Where Are These Files Stored?

Because large model weights and patient-derived LoRA adapters cannot be hosted on public GitHub (due to GitHub's 100 MB file limit and PhysioNet DUA compliance), the project is partitioned into two secure cloud repositories:

### A. GitHub Repository (Code, Docs, Reports & Presentation)
- **URL:** [`https://github.com/LAXMINARAYAN24/Private_Med_LLM`](https://github.com/LAXMINARAYAN24/Private_Med_LLM)
- **Visibility:** Public / Repository
- **What is stored here:**
  - All application code (`src/dashboard/`, `src/phi_defense/`, `pipeline/`, `run_dashboard.py`).
  - Final IEEEtran Research Report PDF: [`docs/deliverables/.../Privacy_Preserving_Medical_Large_Language.pdf`](file:///c:/Users/sahul/Downloads/Trial/Trial/docs/deliverables/Privacy_Preserving_Medical_Large_Language_Models__Mitigating_Protected_Health_Information_Leakage_via_Contrastive_Prototype_Guardrails__Dynamic_Mixtur/Privacy_Preserving_Medical_Large_Language.pdf).
  - Final Presentation Deck PPTX: [`docs/deliverables/.../Privacy_Preserving_Medical_Large_Language.pptx`](file:///c:/Users/sahul/Downloads/Trial/Trial/docs/deliverables/Privacy_Preserving_Medical_Large_Language_Models__Mitigating_Protected_Health_Information_Leakage_via_Contrastive_Prototype_Guardrails__Dynamic_Mixtur/Privacy_Preserving_Medical_Large_Language.pptx).
  - Complete Overleaf LaTeX package and all 16 evaluation figures at 300 DPI (`results/`).
  - Automated one-click restoration script: [`restore_checkpoints.py`](file:///c:/Users/sahul/Downloads/Trial/Trial/restore_checkpoints.py).

### B. Hugging Face Hub (Trained Model Weights & Checkpoints)
- **URL:** [`https://huggingface.co/Laxminarayan24/medshield-checkpoints`](https://huggingface.co/Laxminarayan24/medshield-checkpoints)
- **Visibility:** **Private** *(Secured with your Hugging Face account)*
- **Total Payload:** **37 files (1.68 GB)**
- **What is stored here:**
  - `checkpoints/guardrail_encoder_plus/` (Prototypes and model weights)
  - `checkpoints/mope_plus/` (Load-balanced MoPE routers)
  - `checkpoints/m2_gate/` and `checkpoints/m2_medgemma_gate/` (Attention gates)
  - `m1_medgemma_lora/` and `checkpoints/m1_medgemma_lora/` (Fine-tuned MedGemma LoRA adapters)
  - `checkpoints/m5_1_unlearn/` and `checkpoints/m5_2_unlearn/` (Unlearning baseline weights)

---

## 3. How to Restore the Entire Project in One Click

Follow these 3 quick steps on any computer:

### Step 1: Clone the Code from GitHub
```bash
git clone https://github.com/LAXMINARAYAN24/Private_Med_LLM.git
cd Private_Med_LLM
```

### Step 2: Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Run the One-Click Restoration Script
Ensure you are logged into Hugging Face (or set your token in `.env` as `HF_TOKEN=hf_...`), then execute:

```bash
python restore_checkpoints.py
```

**What this script does automatically:**
1. Authenticates securely with Hugging Face using your token.
2. Downloads all 37 trained checkpoint files (~1.68 GB) directly into their correct directory paths (`checkpoints/` and `m1_medgemma_lora/`).
3. Validates file sizes and tensor integrity.

---

## 4. Launch the System

Once restored, launch the complete full-stack assistant and telemetry dashboard:

```bash
python run_dashboard.py
```

- **Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **VRAM Footprint:** ~3.35 GB under 4-bit NF4 quantization on an NVIDIA RTX 4050 GPU (or runs on CPU in inference fallback mode).

---

## 5. Alternative Manual One-Liner (Python REPL)

If you do not want to use `restore_checkpoints.py`, you can download the checkpoints using a single Python one-liner:

```python
python -c "from huggingface_hub import snapshot_download; snapshot_download('Laxminarayan24/medshield-checkpoints', local_dir='.', ignore_patterns=['README.md', '.git*'])"
```

---

## 6. Author Credits & Institutional Attribution
- **Harshul Jain** (231IT025)
- **Laxminarayan Sahu** (231IT035)
- **Nikhil Agarwal** (231IT044)

Department of Information Technology,  
**National Institute of Technology Karnataka (NITK), Surathkal**  
Course: **IT488 — Large Language Models**
