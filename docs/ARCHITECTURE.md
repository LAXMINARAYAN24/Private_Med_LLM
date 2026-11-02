# System Architecture (ARCHITECTURE.md)

> **Project:** MedShield AI / Private Medical LLM Defense  
> **Authors:** Harshul Jain (231IT025), Laxminarayan Sahu (231IT035), Nikhil Agarwal (231IT044)  
> **Affiliation:** Department of Information Technology, National Institute of Technology Karnataka (NITK) Surathkal  
> **Date:** November 2026

## 1. System Overview

```
[ User Browser / Clinician Client ]
                │
                │ HTTP REST / WebSocket (/api/chat, /api/inspect, /api/benchmarks)
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Web Application                      │
│                  (src/dashboard/app.py)                         │
└───────────────┬─────────────────────────────────┬───────────────┘
                │                                 │
                ▼                                 ▼
┌────────────────────────────────┐ ┌──────────────────────────────┐
│   Tier 1 Active Defense Engine │ │ GPU MedGemma Inference Engine│
│   (scripts/interactive_test.py)│ │ (src/phi_defense/models.py)  │
│                                │ │                              │
│ • MiniLM-L6-v2 Encoder         │ │ • Base: google/medgemma-4b-it│
│   (384-dim SupCon embedding)   │ │ • Quant: 4-bit NF4 pinned    │
│ • 9-Class Prototype Guardrail  │ │ • Adapter: M1 LoRA (MIMIC)   │
│   (Cosine distance τ=0.521)    │ │ • Tier-2: In-Decoder Gate(M2)│
│ • MoPE Router (K=9 Experts)    │ │ • Dynamic Utility Router     │
│   (Anti-collapse loss λ=0.3)   │ │   (Bypasses LoRA on benign)  │
└────────────────────────────────┘ └──────────────────────────────┘
                │                                 ▲
                │ Block Decision / Risk Score     │
                └─────────────────────────────────┘
                  If Blocked: Return HIPAA Refusal (0 GPU waste)
                  If Allowed: Execute Gated Generation
```

---

## 2. Tech Stack

| Layer | Technologies | Purpose |
|---|---|---|
| **Frontend UI** | HTML5, Tailwind CSS, Vanilla ES6, Chart.js, Lucide Icons | Responsive clinical workstation dashboard, zero-bundle overhead |
| **Backend API** | FastAPI, Uvicorn, Pydantic, Python 3.11/3.12 | Asynchronous, low-latency API serving chat, telemetry, and metrics |
| **Deep Learning** | PyTorch 2.x (CUDA 12), Transformers, PEFT, BitsAndBytes | GPU VRAM management, 4-bit quantization, LoRA adapter attachment |
| **Defense Stack** | SentenceTransformers (`all-MiniLM-L6-v2`), PyTorch MLPs | 9-Class prototype matching, MoPE routing, and attention gating |
| **Hardware** | NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM) | Dedicated local hardware target |

---

## 3. Project Structure

```
c:\Users\sahul\Downloads\Trial\Trial\
├── docs/
│   ├── PRD.md                     # Product Requirements Document
│   ├── AGENTS.md                  # AI Agent conventions and instructions
│   ├── DESIGN_SYSTEM.md           # Cyber-clinical design tokens and rules
│   └── ARCHITECTURE.md            # Technical system architecture (this file)
├── src/
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI backend application & routes
│   │   └── static/
│   │       ├── index.html         # Single-Page Dashboard & Chatbot UI
│   │       ├── style.css          # Glassmorphic dark styling & animations
│   │       └── app.js             # Reactive state manager & Chart.js renderer
│   └── phi_defense/
│       ├── config.py              # Hyperparameters, directories, paths
│       ├── guardrail.py           # Contrastive dual-encoder & prototype filter
│       ├── mope.py                # Mixture-of-PHI-Experts router & heads
│       ├── gate.py                # In-decoder attention suppression gate
│       └── models.py              # MedGemma-4B loader, PEFT merger, decoding
├── scripts/
│   ├── run_medgemma.py            # Unified GPU runner (M0, M1, M6)
│   ├── interactive_test.py        # Real-time Guardrail & MoPE evaluation engine
│   └── generate_all_eval_figures.py # Generates 15 publication figures (300 DPI)
├── checkpoints/
│   ├── m1_medgemma_lora/          # Fine-tuned MedGemma adapter (1,800 patients)
│   ├── guardrail_encoder_plus/    # 9-class contrastive encoder + prototypes
│   └── mope_plus/                 # 9-expert MoPE router weights
├── results/
│   ├── eval_figures/              # 15 high-resolution evaluation figures (300 DPI)
│   └── *.json                     # Raw experimental evaluation JSONs
├── run_dashboard.py               # One-click desktop full-stack launcher
└── requirements.txt
```

---

## 4. End-to-End Request Pipeline (Data Flow)

```
1. User Submits Prompt via Web UI
   │
   ▼
2. FastAPI `/api/chat` Route
   │
   ▼
3. Tier 1 Active Defense Inspection (InteractiveDefenseRunner)
   │  ├── all-MiniLM-L6-v2 embeds prompt → z ∈ ℝ³⁸⁴ (Latency: ~2.1 ms)
   │  ├── Cosine similarity against 9 prototypes evaluated
   │  │   └── If max_sim ≥ τ_block (0.521) AND is_attack → FLAG ATTACK
   │  └── MoPE Router evaluates 9 expert heads (Latency: ~0.9 ms)
   │      └── Calculates continuous risk score ŷ = Σ π_k · σ(Expert_k(z))
   │
   ├─► [IF ATTACK DETECTED]
   │   └── Return immediate refusal:
   │       "I cannot share protected health information, patient history..."
   │       (LLM generation is skipped. 0.0% PHI leak. 0 GPU memory churn.)
   │
   └─► [IF BENIGN CLINICAL INQUIRY]
       ├── Log verified benign status in telemetry
       ├── Check for patient identifier cues (for Tier 2 In-Decoder Gate)
       ├── Dynamic Utility Routing:
       │   └── with model.disable_adapter():
       │       Execute MedGemma-4B-IT GPU generation (4-bit NF4)
       │       (Eliminates LoRA slot collapse; generates comprehensive essay)
       └── Return generated text + defense inspection telemetry JSON
```

---

## 5. Memory Management & GPU Constraints

* **Strict 6GB VRAM Budget:**
  * Base `google/medgemma-4b-it` in 4-bit NF4 occupies $\sim 2.7\text{ GB}$ VRAM.
  * `MiniLM-L6-v2` encoder (22M parameters) occupies $\sim 88\text{ MB}$ VRAM.
  * MoPE Router + 9 expert MLPs occupy $\sim 15\text{ MB}$ VRAM.
  * Peak memory during generation: $\approx 3.8\text{ GB}$ (safely within the 6.0 GB ceiling).
* **Single Shared Embedding:** Prompt is embedded once by `MiniLM-L6-v2`; the 384-dimensional representation is shared between Prototype Guardrail and MoPE, minimizing redundant compute.
* **Context Window Truncation:** Multi-turn chat history is bounded to the 10 most recent turns to prevent quadratic KV-cache growth in GPU memory.

