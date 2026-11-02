# Design System (DESIGN_SYSTEM.md)

> **Project:** MedShield AI / Private Medical LLM Defense  
> **Authors:** Harshul Jain (231IT025), Laxminarayan Sahu (231IT035), Nikhil Agarwal (231IT044)  
> **Affiliation:** Department of Information Technology, National Institute of Technology Karnataka (NITK) Surathkal  
> **Date:** November 2026

## 1. Direction & Personality
* **Aesthetic Theme:** "Clinical Precision & Cyber Defense" (Claude-Inspired Progressive Disclosure).
* **Tone:** Professional, high-trust, uncluttered, and authoritative. Designed first for **clinical end-users (doctors, nurses, medical students)** with 90% of screen real estate dedicated to a clean, focused medical conversation.
* **Progressive Disclosure:** Heavy defense telemetry (cosine distance $\tau=0.521$, 9-Expert MoPE router entropy, decoder token suppression traces) is hidden under the hood by default, smoothly sliding open in a right-hand **"Artifacts" Defense Inspector drawer** only when an adversarial attack is triggered or explicitly requested by an evaluator.

---

## 2. Color Palette & Tokens

```css
:root {
  /* Claude-inspired warm dark/clinical canvas */
  --bg-primary: #121316;          /* Main canvas background */
  --bg-sidebar: #18191d;          /* Collapsible left navigation */
  --bg-surface: #202227;          /* Cards, prompt dock, assistant bubbles */
  --bg-surface-hover: #282a30;    /* Elevated hover state */
  --bg-inspector: #16171b;        /* Slide-out Defense Inspector (Right drawer) */
  --bg-input: #1a1b20;            /* Dedicated prompt dock background */

  /* Crisp Typography */
  --text-primary: #f4f4f0;        /* Crisp readable off-white */
  --text-body: #d1d5db;           /* Slate 300 readable body */
  --text-muted: #9ca3af;          /* Secondary telemetry labels */
  --text-dim: #6b7280;            /* De-emphasized timestamps & borders */

  /* Clinical Accents & Action */
  --accent-clinical: #38bdf8;     /* Cyan/Sky for clinical highlights */
  --accent-blue: #3b82f6;         /* Primary send action / Focus */
  
  /* Defense Semantic States */
  --status-safe: #10b981;         /* Emerald 500 - Benign Verified / 0.0% Leak */
  --status-safe-bg: rgba(16, 185, 129, 0.12);
  --status-danger: #ef4444;       /* Rose Red 500 - Adversarial Trigger / Blocked */
  --status-danger-bg: rgba(239, 68, 68, 0.12);
  --status-warning: #f59e0b;      /* Amber 500 - Out of Domain */
  --status-warning-bg: rgba(245, 158, 11, 0.12);

  /* Subtle Architectural Borders & Shadows */
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-active: rgba(56, 189, 248, 0.35);
  --border-danger: rgba(239, 68, 68, 0.35);
  --shadow-dock: 0 12px 40px rgba(0, 0, 0, 0.45);
  --shadow-drawer: -8px 0 30px rgba(0, 0, 0, 0.35);
}
```

---

## 3. Layout Structure: 3-Column Progressive Disclosure

```
┌─────────────────┬────────────────────────────────────────────┬───────────────────┐
│  Left Sidebar   │         Center Medical Chat Column         │  Right Artifacts  │
│  (Collapsible)  │        (Centered Max-Width 760px)          │ Defense Inspector │
│                 │                                            │   (Slide-Out)     │
│ • + New Chat    │ • Top Bar: Model Selector Pill             │                   │
│ • Benchmarks    │   (M6 Defended vs M1 Leaky vs M0 Stock)    │ • Verdict Card    │
│ • Figures       │ • Subtle Protection Badge (τ=0.521)        │ • 9-Class Radar   │
│ • Prompt Lib    │ • Scrollable Clinical Conversation         │ • MoPE Head Bars  │
│ • RTX 4050 GPU  │ • Sticky Floating Prompt Dock              │ • Tier-1/2 Plumb  │
│   (VRAM Pill)   │   (Quick-Probe Chips + Auto-Expanding Text)│ • Custom Sandbox  │
│                 │                                            │                   │
│ [width: 270px]  │ [flex: 1, centered 760px container]        │ [width: 0 -> 440] │
└─────────────────┴────────────────────────────────────────────┴───────────────────┘
```

1. **Left Sidebar (`.claude-sidebar`):**
   - Collapsible via `toggleSidebar()`.
   - Contains session controls, modal triggers for Table VII and Figures Gallery, and categorized prompt library.
   - Low-profile hardware status at the bottom (`RTX 4050 · 4-bit NF4`).

2. **Center Workspace (`.main-chat-column`):**
   - Clean, centered conversational flow (`max-width: 760px`, `margin: 0 auto`).
   - Model selector dropdown pill (Claude-inspired) directly in the header.
   - Assistant responses feature an unobtrusive clinical badge with a discreet `[Inspect Defense Trace ↗]` link.

3. **Right Artifacts Drawer (`.defense-inspector-drawer`):**
   - Hidden by default (`width: 0`).
   - Smoothly slides open (`width: 440px`) on user demand or automatically when an adversarial extraction attempt is blocked.
   - Houses the real-time Radar threat distribution, MoPE expert utilization bar chart, and mathematical verification data.

---

## 4. Typography Hierarchy
* **Primary UI Font:** `'Plus Jakarta Sans'`, `'Inter'`, `-apple-system`, sans-serif.
* **Code & Telemetry Font:** `'JetBrains Mono'`, `'Fira Code'`, monospace.
* **Scale:**
  * **Hero Title (H1):** `24px / 32px` | Bold 800 (`letter-spacing: -0.02em`)
  * **Section Title (H2):** `18px / 26px` | SemiBold 700
  * **Card Header (H3):** `13px / 18px` | SemiBold 700 (`uppercase tracking-wider`)
  * **Body Text (Assistant):** `15px / 25px` | Regular 400 (`line-height: 1.7`)
  * **Badges & Pills:** `11.5px / 16px` | SemiBold 600
  * **Monospace Telemetry:** `11px / 15px` | Medium 500

---

## 5. Micro-Interactions & States
* **Adversarial Interception:** When an attack is submitted (e.g., *Emergency Dept Pretext*), the assistant displays an immediate boundary block: *"I cannot share protected health information, patient history, or clinical records..."*, the right-hand Inspector drawer automatically glides open, and the radar/MoPE charts reflect the threat vector in under $3.5\text{ ms}$.
* **Safe Clinical Inquiry:** For benign medical queries, the chat remains 100% serene. The model provides complete multi-paragraph medical reasoning with a subtle emerald pill: `✅ SAFE CLINICAL INQUIRY · 100% Medical Intelligence`.
