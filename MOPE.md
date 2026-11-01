# MoPE (Mixture-of-PHI-Experts) & M4 — Deep Dive

Let me walk you through the architecture like a story.

---

## 🧠 The Problem MoPE Solves

After fine-tuning MedGemma (M2), the model learned *too well* — it memorized real patient names, dates, and MRNs from the training data and would leak them when prompted cleverly. A single monolithic guardrail (M3) could block known PHI categories, but it couldn't *route* sensitive inputs to specialized handlers. That's where **MoPE** comes in.

---

## 🏗️ M4 Architecture: Mixture-of-PHI-Experts Router

Think of MoPE as a **hospital triage system**:

```
Incoming Prompt
      │
      ▼
 ┌─────────────┐
 │  Encoder    │  ← MiniLM-L6-v2 (384-dim sentence embedding)
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   Router    │  ← Linear(384 → 9) + Softmax
 │  (Gating)   │     Picks the top-K most relevant experts
 └──────┬──────┘
        │ routes to one of 9 expert MLPs
   ┌────┴────┐
   │  Expert │  ← Each expert: Linear(384→256→128) ReLU
   │   #k    │    Specialized for a PHI category
   └────┬────┘
        │
        ▼
 ┌─────────────┐
 │  Decision   │  → Block / Sanitize / Pass
 └─────────────┘
```

### Why 9 Experts?
The **9 PHI categories** from HIPAA map 1-to-1 with experts:

| Expert | PHI Category |
|--------|-------------|
| 0 | Patient Names |
| 1 | Geographic Data |
| 2 | Dates & Ages |
| 3 | Phone/Fax Numbers |
| 4 | Email Addresses |
| 5 | SSN / Medical Record Numbers |
| 6 | Health Plan / Account Numbers |
| 7 | Device Identifiers |
| 8 | Biometric Identifiers |

Each expert **learns to recognize adversarial patterns** specific to its PHI type — something a single classifier cannot do efficiently.

---

## 🔄 The Anti-Collapse Loss (Critical Innovation)

Without a penalty, the router would **collapse** — always sending everything to Expert 0 (the easiest to optimize). This is the classic **load-balancing problem** in MoE systems.

The anti-collapse auxiliary loss forces **uniform expert utilization**:

```
L_aux = λ_aux × Σ_k [ p_k × log(p_k) ]   (negative entropy penalty)
```

Where:
- `λ_aux = 0.3` (tuned in your paper)
- `p_k` = fraction of tokens/prompts routed to expert `k`
- The loss **maximizes entropy** across experts → forces balanced usage

**Result from Table VI**: Expert utilization entropy = 0.94 (near-perfect balance of 1.0)

---

## 🔗 How M4 Fits in the M6 Hybrid Pipeline

```
User Prompt
    │
    ▼
[M3 Guardrail] ── BLOCK ──→ "I can't help with that"
    │ PASS
    ▼
[M4 MoPE Router] ── routes to expert k
    │
    ▼
[Expert MLP k] ── high confidence PHI → SANITIZE
    │ low confidence → PASS
    ▼
[M2 Attention Gate] ── final safety check
    │
    ▼
MedGemma Generation
```

**M3 handles the obvious cases** (keyword/semantic similarity blocking).
**M4 handles the *subtle* adversarial cases** — prompts that slip past M3 but embed PHI extraction intent.

---

## 📊 Final Numbers (from Report_v2_LLM.pdf, Table VII)

| Model | PHI Leak Rate | Medical Utility |
|-------|--------------|-----------------|
| M0 (Stock MedGemma) | 0.0% | 89.2% |
| M2 (Fine-tuned, leaky) | **34.7%** | 97.8% |
| M3 (Guardrail only) | 8.3% | 94.1% |
| M4 (MoPE only) | 4.1% | 95.6% |
| **M6 (M3+M4+M2 Gate)** | **0.0%** | **97.4%** |

**M6 achieves zero leakage while retaining 97.4% of the clinical utility of the fine-tuned model** — that's the headline result of the paper.

---

## 🎯 The Intuition in One Sentence

> MoPE works because PHI leakage is *not* a monolithic problem — different types of private data require different detection strategies, and a learned router can specialize experts for each category while the anti-collapse loss ensures no single expert becomes a bottleneck.

---

Want me to now generate the updated PowerPoint slides incorporating these architectural diagrams, or do you want to run the interactive demo to see M6 in action?