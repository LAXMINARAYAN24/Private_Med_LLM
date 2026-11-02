"""
Renders the EXACT pipeline flowchart as seen in interactive_pipeline_architecture.html
into high-resolution PNG image format.
"""

import os
import time
import subprocess
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
BRAIN_DIR = Path(os.environ["USERPROFILE"]) / ".gemini" / "antigravity" / "brain" / "7041f306-a030-4346-8fb2-ce2d907ce798"

# The EXACT HTML page with the identical Mermaid graph, styling, and configuration
RENDER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #090b10;
      display: inline-flex;
      padding: 24px;
      margin: 0;
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    .flowchart-viewport {
      background: #0f1117;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 24px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
    }
    .flowchart-viewport svg {
      max-width: 100%;
      height: auto;
    }
  </style>
</head>
<body>
  <div class="flowchart-viewport" id="capture-container">
    <div class="mermaid" id="mermaid-flowchart">
flowchart TD
    subgraph Ingest ["1. Ingestion Gateway"]
        Q["Clinician / Attacker Query"] --> API["FastAPI Endpoint\n(src/dashboard/app.py)"]
    end

    subgraph Tier1 ["2. Tier-1 Prompt-Boundary Defense"]
        API --> G["Contrastive Prototype Guardrail\n(src/phi_defense/guardrail.py)"]
        G -->|"Cosine Similarity Vector"| T{"Threshold Gate\n(τ = 0.521)"}
        T -->|"Cosine Sim > τ & PHI Target"| REF["Instant Privacy Refusal\n(2.8 ms, 0% Leak, Zero GPU Cost)"]
        T -->|"Cosine Sim ≤ τ"| MOPE["MoPE 9-Head Dynamic Router\n(src/phi_defense/mope.py)"]
        MOPE -->|"Entropy Regularization (H = 0.94)"| DISP["Threat Category Dispatch\n(#0 Benign ... #8 Roleplay)"]
    end

    subgraph Dispatch ["3. Dynamic Utility Dispatch"]
        DISP -->|"High-Confidence Benign Clinical Query"| FAST["Direct Forward Fastpath\n(100% Diagnostic Utility)"]
        DISP -->|"Ambiguous / Extraction Probe"| ARM["Arm Tier-2 Suppression Hooks"]
    end

    subgraph Tier2 ["4. Tier-2 In-Decoder Attention Gate"]
        ARM --> GATE["Decoder Self-Attention Hooks [n/4, 3n/4)\n(src/phi_defense/gate.py)"]
        GATE -->|"Zero Logits for Patient Tokens"| SUPP["Token-Level Suppression\n(0.0% Leak Rate, +1.2 ms)"]
    end

    subgraph Backbone ["5. Generative Clinical Backbone"]
        FAST --> MEDGEMMA["Google MedGemma-4B-IT (4-bit NF4)\n(src/phi_defense/models.py)"]
        SUPP --> MEDGEMMA
        MEDGEMMA --> OUT["Safe Clinical Diagnostic Response\n(Zero HIPAA Disclosure)"]
    end

    subgraph Telemetry ["6. Live Workstation Telemetry"]
        G -.-> TEL["Defense & MoPE Inspector Drawer"]
        MOPE -.-> TEL
        MEDGEMMA -.-> TEL
        TEL --> RADAR["MediRed+ Radar Chart &\nExpert Utilization Histogram"]
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
    </div>
  </div>

  <script>
    mermaid.initialize({
      startOnLoad: true,
      theme: 'dark',
      securityLevel: 'loose',
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

def main():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    render_file = DOCS_DIR / "_temp_exact_flowchart.html"
    render_file.write_text(RENDER_HTML, encoding="utf-8")

    raw_screenshot = DOCS_DIR / "_temp_raw.png"
    file_url = f"file:///{str(render_file).replace(os.sep, '/')}"

    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--window-size=1200,1900",
        f"--screenshot={str(raw_screenshot)}",
        file_url
    ]

    print(f"[*] Capturing screenshot via {chrome_path}...")
    subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    time.sleep(1)

    if not raw_screenshot.exists():
        print("[!] Screenshot generation failed!")
        return

    # Autocrop to the dark container with small border
    img = Image.open(raw_screenshot)
    bg = Image.new(img.mode, img.size, (9, 11, 16, 255) if img.mode == 'RGBA' else (9, 11, 16))
    
    # Find bounding box of non-background content
    from PIL import ImageChops
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        # Add slight padding
        pad = 16
        x1 = max(0, bbox[0] - pad)
        y1 = max(0, bbox[1] - pad)
        x2 = min(img.width, bbox[2] + pad)
        y2 = min(img.height, bbox[3] + pad)
        cropped = img.crop((x1, y1, x2, y2))
    else:
        cropped = img

    target_png = DOCS_DIR / "pipeline_flowchart.png"
    cropped.save(target_png, "PNG")
    print(f"[✓] Saved cropped crisp pipeline flowchart: {target_png} ({cropped.size[0]}x{cropped.size[1]}, {target_png.stat().st_size // 1024} KB)")

    # Also copy to brain directory
    if BRAIN_DIR.exists():
        brain_target = BRAIN_DIR / "pipeline_flowchart.png"
        cropped.save(brain_target, "PNG")
        print(f"[✓] Copied to conversation artifact: {brain_target}")

    # Cleanup temp
    if render_file.exists():
        render_file.unlink()
    if raw_screenshot.exists():
        raw_screenshot.unlink()

if __name__ == "__main__":
    main()

