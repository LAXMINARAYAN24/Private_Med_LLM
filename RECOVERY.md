# 🛡️ MedShield AI — Project Recovery & 30 GB Cloud Restoration Guide

This document explains **how the entire 30.0 GB MedShield AI system is secured across the cloud** and **how to restore and run the complete system on any fresh machine without any retraining**.

---

## 1. Complete 30.0 GB Cloud Storage Accounting

Every single file and byte of the 30.0 GB project is permanently stored and accounted for:

| Component | What Is Inside | Exact Size | Cloud Platform & Repository |
| :--- | :--- | :---: | :--- |
| **1. Source Code & Deliverables** | All Python scripts, UI dashboard, final 17-page IEEE Report PDF, 18-slide PPTX deck, LaTeX Overleaf suite, figures, and test logs. | **0.12 GB** *(222 files)* | ☁️ **GitHub Repository**<br>[`https://github.com/LAXMINARAYAN24/Private_Med_LLM`](https://github.com/LAXMINARAYAN24/Private_Med_LLM) |
| **2. Trained Model Checkpoints** | MiniLM contrastive prototypes, 9-expert MoPE router, Tier-2 in-decoder attention suppression gates, and MedGemma LoRA adapters. | **1.68 GB** *(43 files)* | ☁️ **Private Hugging Face Models Hub**<br>[`https://huggingface.co/Laxminarayan24/medshield-checkpoints`](https://huggingface.co/Laxminarayan24/medshield-checkpoints) |
| **3. Raw Research Datasets** | All 10,000 MIMIC-III inpatient records, 1,800-patient target facts cohort, MediRed+ 6,624 attack prompts, and evaluation splits. | **12.09 GB** *(122 files)* | ☁️ **Private Hugging Face Datasets Hub**<br>[`https://huggingface.co/datasets/Laxminarayan24/medshield-data`](https://huggingface.co/datasets/Laxminarayan24/medshield-data) |
| **4. Base Model Weights** | Pretrained Google MedGemma-4B-IT foundation model cache. | **10.44 GB** | ☁️ **Google's Official Cloud Hub**<br>[`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it) *(Auto-streams on launch)* |
| **5. Python Virtual Environment** | Installed PyTorch and CUDA runtime binaries. | **5.75 GB** | ☁️ **PyPI Cloud via `requirements.txt`**<br>*(Recreated in 2 minutes anytime via pip)* |
| **TOTAL ACCOUNTED FOR** | **Entire Project & Runtime** | **30.08 GB** | **100% Retrievable from the Cloud** |

---

## 2. What Files Are Needed to Run Without Retraining?

To run the full multi-tier defense architecture, the interactive clinical chat assistant, and the live telemetry dashboard ([`run_dashboard.py`](file:///c:/Users/sahul/Downloads/Trial/Trial/run_dashboard.py)), only the **code (GitHub)** and the **checkpoints (Hugging Face)** are needed (~1.8 GB total):

```text
Private_Med_LLM/
├── checkpoints/
│   ├── guardrail_encoder_plus/       # Tier-1 supervised contrastive prototypes (model.safetensors, prototypes.pt)
│   ├── mope_plus/                    # Tier-1 9-expert dynamic risk router (mope_plus.pt)
│   ├── m2_gate/                      # Tier-2 in-decoder attention suppression gate (gate.pt)
│   └── m1_medgemma_lora/             # MedGemma-4B-IT QLoRA adapter
├── m1_medgemma_lora/                 # Target adapter directory (adapter_model.safetensors)
├── src/dashboard/                    # FastAPI web server & live telemetry dashboard
└── run_dashboard.py                  # Full-stack system launcher
```

---

## 3. How to Restore the Entire Project in One Click

Follow these simple steps on any machine:

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
Make sure your Hugging Face token is set in `.env` (`HF_TOKEN=hf_...`) or you are logged in (`huggingface-cli login`), then run:

```bash
# Option A: Restore trained model checkpoints (~1.68 GB) — Ready to run immediately!
python restore_checkpoints.py

# Option B: Also restore the full raw MIMIC-III and MediRed+ datasets (~12.1 GB)
python restore_checkpoints.py --include-data
```

**What this script does automatically:**
1. Authenticates securely with Hugging Face using your token.
2. Downloads the trained checkpoint files (~1.68 GB) directly into their correct directory paths (`checkpoints/` and `m1_medgemma_lora/`).
3. If `--include-data` is specified, also downloads the complete raw MIMIC-III and MediRed+ corpora into `data/` and `MIMIC -III (10000 patients)/`.
4. Validates file sizes and tensor integrity.

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

## 5. Direct Cloud URLs

1. **GitHub Repository:** [`https://github.com/LAXMINARAYAN24/Private_Med_LLM`](https://github.com/LAXMINARAYAN24/Private_Med_LLM)
2. **Private Hugging Face Models:** [`https://huggingface.co/Laxminarayan24/medshield-checkpoints`](https://huggingface.co/Laxminarayan24/medshield-checkpoints)
3. **Private Hugging Face Datasets:** [`https://huggingface.co/datasets/Laxminarayan24/medshield-data`](https://huggingface.co/datasets/Laxminarayan24/medshield-data)

---

## 6. Author Credits & Institutional Attribution
- **Harshul Jain** (231IT025)
- **Laxminarayan Sahu** (231IT035)
- **Nikhil Agarwal** (231IT044)

Department of Information Technology,  
**National Institute of Technology Karnataka (NITK), Surathkal**  
Course: **IT488 — Large Language Models**
