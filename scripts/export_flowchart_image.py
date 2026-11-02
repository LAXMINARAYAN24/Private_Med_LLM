"""
Export pipeline flowchart to high-resolution image format (PNG).
Uses headless browser (Chrome/Edge) to render the exact Mermaid diagram into a publication-quality image.
Also generates a Matplotlib publication-grade architecture diagram.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MedShield AI Pipeline Flowchart</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #080c14;
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      color: #f8fafc;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 40px;
      min-height: 100vh;
    }
    .diagram-card {
      background: #0e1526;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 20px;
      padding: 36px 44px;
      box-shadow: 0 30px 80px rgba(0, 0, 0, 0.7);
      max-width: 1300px;
      width: 100%;
    }
    .header {
      text-align: center;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: #38bdf8;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 4px 12px;
      border-radius: 999px;
      margin-bottom: 10px;
    }
    h1 {
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #ffffff;
      margin-bottom: 6px;
    }
    .subtitle {
      font-size: 13px;
      color: #94a3b8;
    }
    .authors {
      font-size: 12px;
      color: #64748b;
      margin-top: 6px;
      font-weight: 600;
    }
    .mermaid-wrapper {
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 10px 0;
    }
    .mermaid {
      width: 100%;
    }
    .footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 11px;
      color: #64748b;
    }
    .metrics {
      display: flex;
      gap: 16px;
    }
    .metric-pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: #38bdf8;
    }
  </style>
</head>
<body>
  <div class="diagram-card" id="capture-target">
    <div class="header">
      <div class="badge">Architecture Specification</div>
      <h1>MedShield AI — End-to-End Inference Pipeline</h1>
      <div class="subtitle">Multi-Tier Clinical Privacy Defense Framework &bull; Google MedGemma-4B-IT (4-bit NF4)</div>
      <div class="authors">NITK Surathkal &bull; Dept. of Information Technology &bull; Harshul Jain, Laxminarayan Sahu, Nikhil Agarwal</div>
    </div>

    <div class="mermaid-wrapper">
      <pre class="mermaid">
flowchart TD
    subgraph Ingest ["1. Ingestion Gateway"]
        Q["Clinician / Attacker Query"] --> API["FastAPI Endpoint (src/dashboard/app.py)"]
    end

    subgraph Tier1 ["2. Tier-1 Prompt-Boundary Defense"]
        API --> G["Contrastive Prototype Guardrail (src/phi_defense/guardrail.py)"]
        G -->|"Cosine Similarity Vector"| T{"Threshold Gate\\n(&tau; = 0.521)"}
        T -->|"Cosine Sim > &tau; & PHI Target"| REF["Instant Privacy Refusal (2.8 ms, 0% Leak, Zero GPU Cost)"]
        T -->|"Cosine Sim &le; &tau;"| MOPE["MoPE 9-Head Dynamic Router (src/phi_defense/mope.py)"]
        MOPE -->|"Entropy Regularization (H = 0.94)"| DISP["Threat Category Dispatch (#0 Benign ... #8 Roleplay)"]
    end

    subgraph Dispatch ["3. Dynamic Utility Dispatch"]
        DISP -->|"High-Confidence Benign Clinical Query"| FAST["Direct Forward Fastpath (100% Diagnostic Utility)"]
        DISP -->|"Ambiguous / Extraction Probe"| ARM["Arm Tier-2 Suppression Hooks"]
    end

    subgraph Tier2 ["4. Tier-2 In-Decoder Attention Gate"]
        ARM --> GATE["Decoder Self-Attention Hooks [n/4, 3n/4) (src/phi_defense/gate.py)"]
        GATE -->|"Zero Logits for Patient Tokens"| SUPP["Token-Level Suppression (0.0% Leak Rate, +1.2 ms)"]
    end

    subgraph Backbone ["5. Generative Clinical Backbone"]
        FAST --> MEDGEMMA["Google MedGemma-4B-IT (4-bit NF4) (src/phi_defense/models.py)"]
        SUPP --> MEDGEMMA
        MEDGEMMA --> OUT["Safe Clinical Diagnostic Response (Zero HIPAA Disclosure)"]
    end

    subgraph Telemetry ["6. Live Workstation Telemetry"]
        G -.-> TEL["Defense & MoPE Inspector Drawer"]
        MOPE -.-> TEL
        MEDGEMMA -.-> TEL
        TEL --> RADAR["MediRed+ Radar Chart & Expert Utilization Histogram"]
    end

    style Q fill:#1e293b,stroke:#475569,stroke-width:2px,color:#f8fafc
    style API fill:#0369a1,stroke:#38bdf8,stroke-width:2px,color:#ffffff
    style G fill:#047857,stroke:#34d399,stroke-width:2px,color:#ffffff
    style T fill:#b45309,stroke:#fbbf24,stroke-width:2px,color:#ffffff
    style REF fill:#b91c1c,stroke:#f87171,stroke-width:2px,color:#ffffff
    style MOPE fill:#6d28d9,stroke:#a78bfa,stroke-width:2px,color:#ffffff
    style DISP fill:#4338ca,stroke:#818cf8,stroke-width:2px,color:#ffffff
    style FAST fill:#047857,stroke:#34d399,stroke-width:2px,color:#ffffff
    style ARM fill:#c2410c,stroke:#fb923c,stroke-width:2px,color:#ffffff
    style GATE fill:#0f766e,stroke:#2dd4bf,stroke-width:2px,color:#ffffff
    style SUPP fill:#047857,stroke:#34d399,stroke-width:2px,color:#ffffff
    style MEDGEMMA fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#ffffff
    style OUT fill:#047857,stroke:#34d399,stroke-width:2px,color:#ffffff
    style TEL fill:#1e1b4b,stroke:#818cf8,stroke-width:1px,color:#e0e7ff
    style RADAR fill:#111827,stroke:#374151,stroke-width:1px,color:#9ca3af
      </pre>
    </div>

    <div class="footer">
      <div>Department of Information Technology, National Institute of Technology Karnataka, Surathkal</div>
      <div class="metrics">
        <span class="metric-pill">Guardrail: 2.8 ms</span>
        <span class="metric-pill">Gate: +1.2 ms</span>
        <span class="metric-pill">Leak Rate: 0.0%</span>
        <span class="metric-pill">Utility: 100%</span>
      </div>
    </div>
  </div>

  <script>
    mermaid.initialize({
      startOnLoad: true,
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#0e1526',
        primaryColor: '#0369a1',
        primaryTextColor: '#ffffff',
        lineColor: '#64748b'
      },
      flowchart: {
        useMaxWidth: false,
        htmlLabels: true,
        curve: 'basis'
      }
    });
  </script>
</body>
</html>
"""

def find_browser():
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def export_browser_png():
    browser = find_browser()
    if not browser:
        print("[!] No Chrome or Edge executable found.")
        return False
    
    temp_html = DOCS_DIR / "_temp_render.html"
    temp_html.write_text(HTML_CONTENT, encoding="utf-8")
    
    target_png = DOCS_DIR / "pipeline_flowchart.png"
    file_url = f"file:///{str(temp_html).replace(os.sep, '/')}"
    
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--window-size=1400,1650",
        f"--screenshot={str(target_png)}",
        file_url
    ]
    
    print(f"[*] Rendering with headless browser: {browser}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    # Wait for file
    time.sleep(1)
    if temp_html.exists():
        temp_html.unlink()
        
    if target_png.exists() and target_png.stat().st_size > 10000:
        print(f"[✓] Saved high-resolution PNG: {target_png} ({target_png.stat().st_size // 1024} KB)")
        return True
    else:
        print("[!] Browser screenshot failed or output file too small.")
        return False

def generate_matplotlib_diagram():
    """Generates a crisp publication-grade architecture flowchart using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('#080c14')
    ax.set_facecolor('#080c14')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title & Metadata
    ax.text(50, 97, "MedShield AI — End-to-End Pipeline Architecture", 
            ha='center', va='center', fontsize=20, fontweight='bold', color='#ffffff', family='sans-serif')
    ax.text(50, 94.5, "Google MedGemma-4B-IT (4-bit NF4) · Multi-Tier Clinical Privacy Defense Framework", 
            ha='center', va='center', fontsize=12, color='#38bdf8', family='sans-serif')
    ax.text(50, 92.5, "NITK Surathkal · Dept. of IT · Harshul Jain (231IT025), Laxminarayan Sahu (231IT035), Nikhil Agarwal (231IT044)", 
            ha='center', va='center', fontsize=10, color='#94a3b8', family='sans-serif')

    def draw_box(x, y, w, h, title, subtitle, fill, stroke, text_color='#ffffff', radius=1.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.5,rounding_size={radius}",
                                      facecolor=fill, edgecolor=stroke, linewidth=2, zorder=3)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=text_color, zorder=4)
            ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', fontsize=8, color='#e2e8f0', zorder=4)
        else:
            ax.text(x + w/2, y + h*0.5, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=text_color, zorder=4)

    def draw_group(x, y, w, h, title, fill='#0e1526', stroke='#1e293b'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=2",
                                      facecolor=fill, edgecolor=stroke, linewidth=1.5, zorder=1)
        ax.add_patch(rect)
        ax.text(x + 2, y + h - 2.2, title, ha='left', va='center', fontsize=10, fontweight='bold', color='#38bdf8', zorder=2)

    def draw_arrow(x1, y1, x2, y2, color='#64748b', style='->', label=None, label_pos=0.5, label_side='right', lw=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=4, shrinkB=4), zorder=2)
        if label:
            lx = x1 + (x2 - x1) * label_pos
            ly = y1 + (y2 - y1) * label_pos
            offset_x = 1.5 if label_side == 'right' else -1.5
            ha = 'left' if label_side == 'right' else 'right'
            ax.text(lx + offset_x, ly, label, ha=ha, va='center', fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc='#080c14', ec='none', alpha=0.85), zorder=4)

    # 1. Ingestion Gateway (Top)
    draw_group(5, 76, 90, 13, "STAGE 1: Ingestion Gateway & Query Classification")
    draw_box(8, 78, 32, 8, "Clinician / Attacker Query", "Direct / Roleplay / Pretext", "#1e293b", "#475569")
    draw_box(52, 78, 40, 8, "FastAPI Ingestion Endpoint", "src/dashboard/app.py (/api/chat)", "#0369a1", "#38bdf8")
    draw_arrow(40, 82, 52, 82, label="REST JSON")

    # 2. Tier-1 Prompt-Boundary Defense
    draw_group(5, 48, 90, 25, "STAGE 2: Tier-1 Prompt-Boundary Contrastive Guardrail & MoPE Dynamic Router")
    draw_box(8, 62, 38, 8, "Contrastive Prototype Guardrail", "src/phi_defense/guardrail.py (all-MiniLM-L6-v2)", "#047857", "#34d399")
    
    # Diamond decision gate
    diamond = patches.RegularPolygon((60, 66), numVertices=4, radius=4.5, orientation=0,
                                     facecolor="#b45309", edgecolor="#fbbf24", linewidth=2, zorder=3)
    ax.add_patch(diamond)
    ax.text(60, 66, "Threshold\nGate\n(τ = 0.521)", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#ffffff', zorder=4)

    draw_box(74, 62, 18, 8, "Instant Refusal", "2.8 ms · 0% Leak", "#b91c1c", "#f87171")
    draw_box(8, 50, 42, 8, "MoPE 9-Head Dynamic Router", "src/phi_defense/mope.py (H = 0.94)", "#6d28d9", "#a78bfa")
    draw_box(56, 50, 36, 8, "Dynamic Utility Dispatch", "Classify Intent (#0 Benign ... #8 Attack)", "#4338ca", "#818cf8")

    # Arrows for Tier-1
    draw_arrow(72, 78, 27, 70, label="Forward Pass")
    draw_arrow(46, 66, 55, 66, label="Cosine Sim")
    draw_arrow(64.5, 66, 74, 66, color='#f87171', label="Sim > τ & PHI Target")
    draw_arrow(60, 61.5, 29, 58, color='#34d399', label="Sim ≤ τ (Safe / Ambiguous)", label_side='left')
    draw_arrow(50, 54, 56, 54, label="Routing Vector")

    # 3. Tier-2 In-Decoder Attention Gate & Backbone
    draw_group(5, 18, 90, 27, "STAGE 3 & 4: Tier-2 In-Decoder Attention Gate & Generative Clinical Backbone")
    draw_box(8, 33, 40, 8, "Tier-2 Attention Gate Hooks", "src/phi_defense/gate.py (Layers [n/4, 3n/4))", "#0f766e", "#2dd4bf")
    draw_box(54, 33, 38, 8, "Token-Level Attention Suppression", "Zeroes patient logits · +1.2 ms overhead", "#047857", "#34d399")
    
    draw_box(8, 20, 40, 8, "Google MedGemma-4B-IT", "src/phi_defense/models.py (4-bit NF4 · RTX 4050)", "#1e3a8a", "#60a5fa")
    draw_box(54, 20, 38, 8, "Safe Clinical Output", "Zero HIPAA PHI Disclosed · 100% Utility", "#047857", "#34d399")

    # Connect Tier-1 Dispatch to Tier-2 / Backbone
    draw_arrow(65, 50, 28, 41, color='#fb923c', label="Ambiguous / Attack Probe")
    draw_arrow(82, 50, 28, 28, color='#34d399', label="High-Confidence Benign (Fastpath)")
    draw_arrow(48, 37, 54, 37, label="Zero Attention")
    draw_arrow(73, 33, 28, 28, label="Arm Gate")
    draw_arrow(48, 24, 54, 24, color='#34d399', label="Safe Tokens")

    # Telemetry Bar (Bottom)
    draw_group(5, 4, 90, 11, "STAGE 5: Real-Time Telemetry & Pareto Frontier Verification")
    draw_box(8, 5.5, 40, 7, "Defense & MoPE Inspector Drawer", "Real-Time Classification & Latency Breakdown", "#1e1b4b", "#818cf8")
    draw_box(54, 5.5, 38, 7, "MediRed+ Radar & Expert Histogram", "9 Threat Categories · Head Routing Diversity", "#111827", "#374151")
    draw_arrow(48, 9, 54, 9)

    out_file = DOCS_DIR / "pipeline_architecture_diagram.png"
    plt.tight_layout()
    plt.savefig(out_file, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[✓] Saved Matplotlib publication architecture diagram: {out_file} ({out_file.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    print("[*] Generating pipeline flowchart images...")
    ok = export_browser_png()
    generate_matplotlib_diagram()
    print("[✓] Image generation complete.")

