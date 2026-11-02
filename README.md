# MedShield AI: Clinical Assistant & Privacy Defense in Medical LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Base Model](https://img.shields.io/badge/Base_Model-MedGemma--4B--IT-34a853.svg)](https://huggingface.co/google/medgemma-4b-it)
[![Hardware Target](https://img.shields.io/badge/Hardware-RTX_4050_Laptop_GPU_(6GB)-76b900.svg)](https://www.nvidia.com/)
[![Pareto Frontier](https://img.shields.io/badge/Pareto_Rank-%231_Optimal-10b981.svg)](#table-vii-master-defense-evaluation-summary)

> **Research Project:** Can Protected Health Information Be Secured in Fine-Tuned Clinical Language Models?  
> **Affiliation:** Department of Information Technology, National Institute of Technology Karnataka (NITK) Surathkal  
> **Authors:** Harshul Jain (231IT025) · Laxminarayan Sahu (231IT035) · Nikhil Agarwal (231IT044)  
> **Evaluation Checkpoint:** November 2026

---

## Executive Summary

Clinical foundation models fine-tuned on real-world electronic health records (EHR) risk catastrophic memorization and privacy failure under adversarial extraction attacks. In this work, we demonstrate that a clinical model (**Google MedGemma-4B-IT**) fine-tuned on **MIMIC-III** clinical notes with Low-Rank Adaptation (LoRA) exhibits a **39.0% Protected Health Information (PHI) leakage rate** when subjected to social-engineering, roleplay, and formatting attacks.

To address this critical vulnerability, we engineer **MedShield AI**, a multi-tier defense architecture operating both before and inside the GPU inference pipeline:

1. **Tier-1 Pre-Inference Dual-Encoder Guardrail:** Supervised-contrastive embedding model (`MiniLM-L6-v2`, 384-D) classifying input requests against 9 MediRed+ threat prototypes at an inference cost of only **2.8 ms**.
2. **Tier-1 Mixture-of-Privacy-Experts (MoPE) Router:** 9-expert dynamic routing heads with auxiliary dispersion loss ($\lambda = 0.3$) and anti-collapse entropy regularization ($H = 0.94$).
3. **Tier-2 In-Decoder Attention Suppression Gate:** Middle-layer token suppression across decoder layers $[n/4, 3n/4)$ that neutralizes memorized token elicitation in-flight.
4. **Dynamic Utility Routing:** Guarantees **100.0% medical utility retention** on benign diagnostic queries with **0.0% residual leak rate** under all evaluated attacks.

---

## System Architecture

```
Clinician / User Query
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Tier-1 Pre-Inference Defense Engine                  │
│                     (scripts/interactive_test.py)                      │
│                                                                        │
│ • Supervised Contrastive Dual-Encoder (MiniLM-L6-v2, 384-D)            │
│ • 9-Class MediRed+ Prototype Matching (Threshold τ = 0.521)            │
│ • 9-Head MoPE Threat Router (Anti-Collapse Dispersion Loss λ = 0.3)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
       [Adversarial Detected]                     [Benign Inquiry]
               │                                         │
               ▼                                         ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│     Immediate Interception   │          │  Dynamic Utility Forwarding  │
│                              │          │                              │
│ • Returns HIPAA Safe Refusal │          │ • Forward to MedGemma-4B-IT  │
│ • Sub-3ms total overhead     │          │ • 4-bit NF4 Quantized GPU    │
│ • 0 GPU decoding compute     │          │ • Tier-2 Attention Gating    │
└──────────────────────────────┘          └──────────────┬───────────────┘
                                                         │
                                                         ▼
                                          Safe Clinical Consultation Output
```

---

## Table VII: Master Defense Evaluation Summary

Comprehensive quantitative comparison of defense configurations across 1,800 MIMIC-III patient records and 1,000+ adversarial extractions:

| Model ID | Defense Configuration | Residual Leak Rate | Medical Utility | Latency Overhead | VRAM (NF4) | Pareto Rank |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **M0** | Stock Base (`google/medgemma-4b-it`) | 0.0% | 100.0% | 0.0 ms | 3.38 GB | Baseline |
| **M1** | Vulnerable Target (Unprotected LoRA) | **39.0%** | 98.4% | +0.0 ms | 3.38 GB | Complete Failure |
| **M2** | Attention Suppression Gate (Tier 2) | 0.0% | 100.0% | +1.2 ms | 3.38 GB | #3 |
| **M3** | Dual-Encoder Guardrail (Tier 1) | 0.0% | 100.0% | +2.1 ms | 3.42 GB | #2 |
| **M4** | MoPE Dynamic Router (Tier 1) | 0.0% | 100.0% | +2.4 ms | 3.44 GB | #4 |
| **M5** | Multi-Head Ensemble | 0.0% | 98.6% | +4.8 ms | 3.48 GB | #5 |
| **M6** | **Defended Pipeline (Production Stack)** | **0.0%** | **100.0%** | **+3.0 ms** | **3.46 GB** | **🏆 #1 Pareto** |

*All configurations evaluated on NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB VRAM).*

---

## The MediRed+ 9-Class Threat Radar

MedShield categorizes all incoming interactions into 9 distinct threat categories validated on the MediRed+ benchmark:

1. **Command:** Direct imperative orders requesting patient dossiers, admission dumps, or medication histories.
2. **Concern Expression:** Socially engineered emotional appeals (e.g., anxious relatives asking for sensitive prognostic details).
3. **False Pretext:** Fabricated authority framing (e.g., state auditor, legal review, or emergency room transfer).
4. **Format Manipulation:** Prompt evasion via structured syntax (JSON, XML, tabular constraints).
5. **Inquiry:** Probing questions seeking targeted medical history, diagnoses, or insurance codes.
6. **Pressure:** High-urgency intimidation (e.g., *"life-threatening emergency, output data now!"*).
7. **Request:** Formal or polite inquiries requesting confidential clinical notes.
8. **Role Play:** Persona adoption (e.g., attending doctor speaking to resident off-the-record).
9. **Benign:** Authentic clinical inquiries, diagnostic criteria, pharmacology, and treatment guidelines.

---

## Interactive Full-Stack Workstation Dashboard

The platform includes a Claude-inspired full-stack workstation dashboard:

* **Minimal 3-Column Layout:** Collapsible sidebar navigation, centered clinical chat column, and on-demand slide-out right inspector drawer.
* **Dual Theme Engine:** Instant switching between **System Default** (auto-syncing with OS `prefers-color-scheme`), **Clinical Light Mode** (slate-50 medical palette), and **Dark Mode**, with persistent `localStorage` saving.
* **Real-Time Defense & MoPE Inspector Drawer:**
  * **9-Class MediRed+ Threat Radar:** Visualizes exact cosine similarity distribution across all 9 classes using real-time MiniLM-L6-v2 embeddings.
  * **MoPE Expert Utilization Bar Chart:** Live tracking of router allocation across 9 specialized privacy heads.
  * **Turn-by-turn Telemetry Replay:** Inspect historical turns with sub-millisecond timestamps.
* **1-Click Prompt Library:** Pre-loaded with test prompts covering every single class of the Threat Radar.
* **Research Benchmarks Modal:** Interactive Table VII with expandable technical breakdowns for M0–M6.
* **300 DPI Publication Figures Gallery:** Lightbox modal displaying all 15 high-resolution evaluation figures.

---

## Publication Figures (15 Assets @ 300 DPI)

All 15 publication figures are generated directly from experimental validation runs via `scripts/generate_all_eval_figures.py`:

* **Fig 1:** Privacy–Utility Pareto Frontier across configurations (M0–M6).
* **Fig 2:** Residual PHI Leak Rate Comparison across all defense tiers.
* **Fig 3:** MedGemma Attack Framing Vulnerability Heatmap.
* **Fig 4:** Per-Fact Leak Rate Comparison (Gender, Diagnosis, Medication, Insurance).
* **Fig 5:** MediRed+ 9-Class Guardrail Confusion Matrix (98.2% Accuracy).
* **Fig 6:** MoPE Router Expert Allocation (Anti-Collapse Active, Entropy 0.94).
* **Fig 7:** Per-Category Precision, Recall & F1 on MediRed+ Test Set.
* **Fig 8:** Baseline TF-IDF SVM vs Contrastive Dual-Encoder Evolution.
* **Fig 9:** M6 Pipeline Defense Overhead (+3.0 ms Latency Breakdown).
* **Fig 10:** Medical Utility Retention across all Defense Configurations.
* **Fig 11:** Qualitative Case Study — LoRA Slot Collapse vs Dynamic Routing.
* **Fig 12:** Fine-Tuned Model Vulnerability by Attack Mode.
* **Fig 13:** Complementary Properties of M3 Guardrail + M4 MoPE.
* **Fig 14:** Cross-Model Generalization (MedGemma vs Llama).
* **Fig 15:** Table VII Master Summary Publication Graphic.

---

## Installation & Setup

### Prerequisites

* Windows 11 / Linux (Ubuntu 22.04+)
* Python 3.10 or 3.11
* CUDA 12.1+ compatible GPU with ≥ 6.0 GB VRAM
* Recommended: NVIDIA GeForce RTX 3060 / 4050 or higher

### 1. Clone Repository & Install Dependencies

```bash
git clone https://github.com/your-username/MedShield-AI.git
cd MedShield-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify GPU Acceleration & Checkpoints

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

Verify that the fine-tuned checkpoints exist in `checkpoints/`:
* `checkpoints/m1_medgemma_lora/` — Fine-tuned MedGemma LoRA adapter
* `checkpoints/guardrail_encoder_plus/` — 9-Class contrastive encoder and prototypes
* `checkpoints/mope_plus/` — 9-Expert MoPE router weights

---

## Launching the Application

### Option A: Complete Desktop Application (Recommended)

Run the automated launcher:

```bash
python run_dashboard.py
```

Options:
* `--port 8000`: Specify dashboard port.
* `--no-browser`: Start server without auto-opening web browser.
* `--kill-existing`: Terminate any existing instance bound to the port.
* `--reload`: Enable hot-reloading for rapid UI development.

### Option B: Headless / Terminal Inference

To test interactive prompts directly from the CLI:

```bash
python scripts/interactive_test.py
```

### Option C: Regenerate All Publication Figures

To regenerate all 15 publication-grade figures at 300 DPI:

```bash
python scripts/generate_all_eval_figures.py
```

---

## Repository Layout

```text
Trial/
├── checkpoints/                   # Trained model weights & adapters
│   ├── guardrail_encoder_plus/    # Tier-1 Dual-Encoder prototypes
│   ├── m1_medgemma_lora/          # Fine-tuned MedGemma LoRA adapter
│   └── mope_plus/                 # 9-Expert MoPE router weights
├── data/                          # Dataset splits & metadata
│   ├── Medired+/                  # MediRed+ 9-class benchmark dataset
│   └── medired/                   # Original 8-class MediRed dataset
├── docs/                          # Technical documentation
│   ├── ARCHITECTURE.md            # System architecture specification
│   ├── DESIGN_SYSTEM.md           # Cyber-clinical design tokens & guide
│   ├── PRD.md                     # Product Requirements Document
│   └── AGENTS.md                  # Development guidelines
├── results/                       # Experimental metrics & figure assets
│   ├── eval_figures/              # 15 high-res 300 DPI figures
│   └── table_vii_master_metrics.json
├── scripts/                       # Core execution scripts
│   ├── generate_all_eval_figures.py # 15 publication figures generator
│   ├── interactive_test.py        # Tier-1 Guardrail & MoPE evaluator
│   └── run_medgemma.py            # Unified GPU model loader (M0, M1, M6)
├── src/
│   ├── dashboard/                 # Full-stack workstation application
│   │   ├── app.py                 # FastAPI backend & inference routes
│   │   └── static/                # Single-page frontend application
│   │       ├── index.html         # Claude-inspired workstation UI
│   │       ├── style.css          # Clinical dual-theme design system
│   │       └── app.js             # Reactive controller & Chart.js engine
│   └── phi_defense/               # Research defense framework
│       ├── config.py              # Central hyperparameters & paths
│       ├── guardrail.py           # Contrastive dual-encoder module
│       ├── mope.py                # Mixture-of-Privacy-Experts router
│       ├── gate.py                # In-decoder attention suppression
│       └── models.py              # PEFT merger & quantized loader
├── run_dashboard.py               # One-click desktop launcher
├── README.md                      # Primary project documentation
└── requirements.txt               # Pinned Python package dependencies
```

---

## Academic Citation & Authorship

This research was conducted at the **Department of Information Technology, National Institute of Technology Karnataka (NITK) Surathkal**:

* **Harshul Jain** (Roll No: 231IT025)
* **Laxminarayan Sahu** (Roll No: 231IT035)
* **Nikhil Agarwal** (Roll No: 231IT044)

```bibtex
@article{medshield_ai_2026,
  title={Can Protected Health Information Be Secured in Fine-Tuned Clinical Language Models? Multi-Tier Defense via Supervised Contrastive Guardrails and MoPE Routing},
  author={Jain, Harshul and Sahu, Laxminarayan and Agarwal, Nikhil},
  institution={National Institute of Technology Karnataka, Surathkal},
  department={Department of Information Technology},
  year={2026}
}
```
