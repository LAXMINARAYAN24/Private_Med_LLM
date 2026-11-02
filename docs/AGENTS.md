# Agent Instructions (AGENTS.md)

> **Project:** MedShield AI / Private Medical LLM Defense  
> **Authors:** Harshul Jain (231IT025), Laxminarayan Sahu (231IT035), Nikhil Agarwal (231IT044)  
> **Affiliation:** Department of Information Technology, National Institute of Technology Karnataka (NITK) Surathkal  
> **Date:** November 2026

## 1. Project Context
* **Codebase:** MedShield AI / Private Medical LLM Defense System (`src/phi_defense/`).
* **Models:** Google MedGemma-4B-IT (`google/medgemma-4b-it`) quantized in 4-bit NF4, fine-tuned on MIMIC-III (1,800 patients, 17,406 QA pairs).
* **Defenses:** Tier-1 Dual-Encoder Guardrail + 9-Expert MoPE Router, Tier-2 In-Decoder Attention Gate (M2), Tier-3 Selective Unlearning (M5), Composed M6 Pipeline.
* **Environment:** Python 3.11/3.12 within `.venv`, PyTorch 2.x with CUDA 12 support, FastAPI backend, modern Tailwind-styled responsive frontend.

---

## 2. Before You Start (Required Reading)
1. Read [`docs/PRD.md`](file:///c:/Users/sahul/Downloads/Trial/Trial/docs/PRD.md) for product goals and feature specifications.
2. Read [`docs/DESIGN_SYSTEM.md`](file:///c:/Users/sahul/Downloads/Trial/Trial/docs/DESIGN_SYSTEM.md) for UI color tokens, layout hierarchy, and component rules.
3. Read [`docs/ARCHITECTURE.md`](file:///c:/Users/sahul/Downloads/Trial/Trial/docs/ARCHITECTURE.md) for data flow, API schema, and model execution architecture.
4. Inspect existing defense classes in [`src/phi_defense/`](file:///c:/Users/sahul/Downloads/Trial/Trial/src/phi_defense) before adding new logic.

---

## 3. General Development Rules
* **Language & Typing:** Write modern, clean Python 3.10+ with type annotations (`from __future__ import annotations`).
* **GPU Memory Stewardship:** The host machine features an NVIDIA GeForce RTX 4050 with 6.0 GB VRAM. Always pin models in 4-bit NF4 (`device_map={"": 0}`). Never perform disk offloading or duplicate base model instances in VRAM.
* **Dynamic Utility Routing:** When Tier-1 confirms a prompt is a benign clinical inquiry, use `with model.disable_adapter():` so MedGemma base responds without LoRA slot-collapse distortion.
* **Modularity:** Keep backend endpoints (`src/dashboard/app.py`), defense runners (`scripts/interactive_test.py`), and frontend components cleanly decoupled.

---

## 4. Code & Quality Guidelines
* **Backend Framework:** FastAPI with async routes and Pydantic schemas.
* **Logging & Error Handling:** Return graceful JSON error payloads (`{"error": "message"}`) and prevent server crashes from OOM or malformed prompts.
* **Frontend:** Clean vanilla modern JavaScript (ES6+), HTML5 semantic markup, Tailwind CSS classes, and Chart.js for data visualization. No bulky bundlers or node_modules dependencies needed.
* **Documentation:** Preserve all existing docstrings and comments. Create clickable links in reports and responses.

---

## 5. Security & Privacy Rules
* **Zero Hardcoded Secrets:** Never hardcode Hugging Face tokens or private EHR patient records in source files; load tokens via `.env` or environment variables.
* **Prompt Boundary Validation:** All user inputs must pass through Tier-1 Guardrail inspection before GPU generation starts.
* **Defense Non-Bypassable:** Attack blocks must return standard HIPAA refusals without calling LLM `.generate()`, conserving GPU compute and preventing residual leakage.

---

## 6. Standard Execution Commands

```powershell
# 1. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 2. Launch Full-Stack Web Dashboard & Medical Chatbot
.\.venv\Scripts\python.exe run_dashboard.py

# 3. Run Interactive Terminal CLI (M6 Defended)
.\.venv\Scripts\python.exe scripts/run_medgemma.py --defended --interactive

# 4. Run Terminal Guardrail & MoPE Prompt Laboratory
.\.venv\Scripts\python.exe scripts/interactive_test.py

# 5. Regenerate 15 Evaluation Figures (300 DPI)
.\.venv\Scripts\python.exe scripts/generate_all_eval_figures.py

# 6. Rebuild Updated Slide Deck (Zero Overflow)
.\.venv\Scripts\python.exe build_updated_deck_from_template.py
```

---

## 7. Boundaries & Prohibitions
* Do **NOT** delete or overwrite trained checkpoints in `checkpoints/` (`m1_medgemma_lora`, `guardrail_encoder_plus`, `mope_plus`).
* Do **NOT** modify the original MIMIC split data files in `data/mimic/splits_1800/`.
* Do **NOT** bypass the Tier-1 decision threshold in defended mode.

