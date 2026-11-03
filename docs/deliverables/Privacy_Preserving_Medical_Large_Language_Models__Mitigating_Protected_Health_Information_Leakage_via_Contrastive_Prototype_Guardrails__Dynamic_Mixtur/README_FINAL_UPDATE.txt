# MedShield AI — Final Priority-One Update

This package contains the updated research report, presentation, and complete LaTeX source with supporting figures.

## Priority-one changes applied
1. Corrected PPT slide 13's M0 stock-base residual leakage from 25.0% to 15.0%, consistent with report Table IV and PPT slide 2.
2. Replaced the PPT slide 13 privacy–utility chart with the report's matching chart, where M0 is plotted at 15.0%.
3. Added the benign false-block rate (FBR) to the PPT slide 13 comparison table:
   - M0: 0.0%
   - M1: 0.0%
   - M2: 0.0%
   - M3: 10.3%
   - M4: 0.0%
   - M5: 0.0%
   - M6: 2.6%
4. Added a short metric definition below the PPT comparison table.
5. Revised the report conclusion so it attributes the 2.6% FBR to the combined Tier-1 composition and explicitly states that the current ablation does not isolate which component causes that reduction. It no longer attributes the reduction to MoPE alone.

## Files
- `Final_Compiled_PHI_Defense_Report.pdf` — final compiled report
- `main.pdf` — same compiled report, for Overleaf project convenience
- `MedShield_AI.pptx` — updated presentation
- `main.tex`, `IEEEtran.cls`, and `figures/` — complete LaTeX source and assets

## Recompile
Run `latexmk -pdf main.tex` from this folder in an environment with LaTeX and the required packages installed.
