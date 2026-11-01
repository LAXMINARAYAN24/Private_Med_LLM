"""
build_updated_deck_from_template.py
===================================
Uses Midsem_Medical_LLM_Privacy_Presentation.pptx as the exact template,
updates text using robust content matching, and embeds the 4 report
figures cleanly into Slides 4, 9, 13, and 14 with ZERO overflow.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(r"c:\Users\sahul\Downloads\Trial\Trial")
BASE_PPTX = ROOT / "Midsem_Medical_LLM_Privacy_Presentation.pptx"
OUT_PPTX  = ROOT / "Final_Medical_LLM_Privacy_Presentation.pptx"

FIGS = ROOT / "PHI_Defense_Report_Overleaf" / "figures"
FIG1_ARCH = FIGS / "fig1_architecture.png"
FIG2_FLOW = FIGS / "fig2_decision_flow.png"
FIG3_PIPE = FIGS / "fig3_training_pipeline.png"
FIG4_EMBD = FIGS / "fig4_shared_embedding.png"
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.500)

# Color tokens
BG_PAGE         = RGBColor(0xF8, 0xFA, 0xFC)
CARD_BG         = RGBColor(0xFF, 0xFF, 0xFF)
CARD_BORDER     = RGBColor(0xE2, 0xE8, 0xF0)
BOX_BG          = RGBColor(0xF1, 0xF5, 0xF9)

TEXT_NAVY       = RGBColor(0x0F, 0x17, 0x2A)
TEXT_BODY       = RGBColor(0x33, 0x41, 0x55)
TEXT_MUTED      = RGBColor(0x64, 0x74, 0x8B)

ACCENT_BLUE     = RGBColor(0x25, 0x63, 0xEB)
BLUE_BG         = RGBColor(0xEF, 0xF6, 0xFF)
ACCENT_GREEN    = RGBColor(0x16, 0xA3, 0x4A)
GREEN_BG        = RGBColor(0xF0, 0xFD, 0xF4)
GREEN_BORDER    = RGBColor(0xBB, 0xF7, 0xD0)
ACCENT_RED      = RGBColor(0xDC, 0x26, 0x26)
RED_BG          = RGBColor(0xFE, 0xF2, 0xF2)
RED_BORDER      = RGBColor(0xFE, 0xC8, 0xC8)
ACCENT_PURPLE   = RGBColor(0x7C, 0x3A, 0xED)
PURPLE_BG       = RGBColor(0xFA, 0xF5, 0xFF)
ACCENT_TEAL     = RGBColor(0x0D, 0x94, 0x88)
TEAL_BG         = RGBColor(0xF0, 0xFD, 0xFA)
ACCENT_INDIGO   = RGBColor(0x4F, 0x46, 0xE5)
INDIGO_BG       = RGBColor(0xEE, 0xF2, 0xFF)

FONT_NAME = "Calibri"


def set_run(p, text, font_size=None, bold=None, color=None):
    """Safely updates text on a paragraph while preserving styling."""
    if len(p.runs) == 0:
        r = p.add_run()
    else:
        r = p.runs[0]
        for extra in p.runs[1:]:
            extra.text = ""
    r.text = text
    r.font.name = FONT_NAME
    if font_size is not None:
        r.font.size = Pt(font_size)
    if bold is not None:
        r.font.bold = bold
    if color is not None:
        r.font.color.rgb = color


def add_card(slide, left, top, width, height, fill_color=CARD_BG, border_color=CARD_BORDER, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE):
    card = slide.shapes.add_shape(shape_type, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = fill_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
    else:
        card.line.fill.background()
    return card


def add_badge(slide, left, top, width, height, text, bg_color=BLUE_BG, text_color=ACCENT_BLUE, font_size=10):
    badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    badge.fill.solid()
    badge.fill.fore_color.rgb = bg_color
    badge.line.fill.background()
    tf = badge.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_NAME
    p.font.size = Pt(font_size)
    p.font.bold = True
    p.font.color.rgb = text_color
    return badge


def update_slide_01(slide):
    """Slide 1: Title & Team."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "CAPSTONE MID-SEMESTER" in t:
                set_run(s.text_frame.paragraphs[0], "CAPSTONE FINAL EVALUATION  ·  OCTOBER 2026", color=ACCENT_BLUE)
            elif "Defending Against PHI Leakage" in t:
                set_run(s.text_frame.paragraphs[0], "Detecting & Mitigating PHI Leakage via Extended Red-Teaming & Multi-Layer Architectural Defense")
            elif "Framed clinical EHR privacy problem" in t:
                set_run(s.text_frame.paragraphs[0], "● Framed clinical EHR privacy problem & HIPAA risk")
            elif "Engineered 4-mode extraction attack" in t:
                set_run(s.text_frame.paragraphs[0], "● Engineered 4-mode extraction attack harness")
            elif "Proved 78.1% PHI leak rate" in t:
                set_run(s.text_frame.paragraphs[0], "● Proved 78.1% PHI leak rate in fine-tuned MedGemma M1")
            elif "Architected In-Decoder Attention" in t:
                set_run(s.text_frame.paragraphs[0], "● In-Decoder Attention Gate (M2): 0.0% leak, +1.2ms")
            elif "K=8 MLPs" in t:
                set_run(s.text_frame.paragraphs[0], "● Architected router over K=9 expert MLPs on MediRed+")
            elif "continuous risk score" in t:
                set_run(s.text_frame.paragraphs[0], "● Formulated continuous risk score ŷ = Σ π_k · σ(Expert_k)")
            elif "anti-collapse auxiliary loss" in t:
                set_run(s.text_frame.paragraphs[0], "● Engineered anti-collapse auxiliary loss L_aux (λ=0.3)")
            elif "100% standalone recall & 0% benign" in t:
                set_run(s.text_frame.paragraphs[0], "● Achieved 0.0% benign false-block & 0.0% residual leak")
            elif "SupCon loss (τ=0.07)" in t:
                set_run(s.text_frame.paragraphs[0], "● Fine-tuned MiniLM-L6-v2 with SupCon loss (τ=0.07, 8 epochs)")
            elif "Built 8 category prototype" in t:
                set_run(s.text_frame.paragraphs[0], "● Curated MediRed+ 9-class dataset (6,624 prompts with Benign)")
            elif "OOD novelty threshold" in t:
                set_run(s.text_frame.paragraphs[0], "● Calibrated OOD novelty threshold (τ=0.931, 3.5% false alarms)")
            elif "Beat 5-fold CV tuned SVM" in t:
                set_run(s.text_frame.paragraphs[0], "● Beat 5-fold CV tuned SVM baseline (F1: 0.884 → 0.983)")
            elif "Mid-Semester Capstone" in t:
                set_run(s.text_frame.paragraphs[0], "Department of Information Technology  ·  National Institute of Technology  ·  Final Capstone Report")


def update_slide_02(slide):
    """Slide 2: Problem Statement."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "10,000+" in t:
                set_run(s.text_frame.paragraphs[0], "1,800", font_size=28, color=ACCENT_BLUE)
            elif "500-patient curated" in t:
                set_run(s.text_frame.paragraphs[0], "17,406 QA pairs (1,800 patients)", font_size=9, color=TEXT_MUTED)
            elif "Base model: 19.4%" in t:
                set_run(s.text_frame.paragraphs[0], "Base MedGemma: 25.0% accidental prior", font_size=9, color=TEXT_MUTED)
            elif "Multi-layer composed defense goal" in t:
                set_run(s.text_frame.paragraphs[0], "M6 Hybrid: 0.0% leak & 97.4% utility", font_size=9, color=TEXT_MUTED)


def update_slide_03(slide):
    """Slide 3: Project Objectives."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "Fine-tune Llama-3.2-1B" in t:
                set_run(s.text_frame.paragraphs[0], "●  Fine-tune MedGemma-4B-IT on 1,800 MIMIC-III patients (17,406 QA pairs) with QLoRA.", font_size=9.5)
            elif "M3 Contrastive Guardrail:" in t and "8 category" in t:
                set_run(s.text_frame.paragraphs[0], "●  M3 Contrastive Guardrail: MiniLM-L6-v2 + 9 MediRed+ prototypes (F1 = 0.983).", font_size=9.5)
            elif "M4 MoPE Router:" in t and "8-expert" in t:
                set_run(s.text_frame.paragraphs[0], "●  M4 MoPE Router: 9-expert network with anti-collapse loss & continuous risk scoring.", font_size=9.5)
            elif "Deliver 0.0% PHI leak rate" in t:
                set_run(s.text_frame.paragraphs[0], "●  Deliver 0.0% PHI leak rate with 97.4% clinical utility retained (+3.0 ms latency).", font_size=9.5)


def update_slide_04(slide):
    """Slide 4: System Architecture (Embed Figure 1)."""
    # Delete diagram shapes 4 to 28 so we can cleanly embed Figure 1
    shapes_to_delete = []
    for i in range(4, 29):
        if i < len(slide.shapes):
            shapes_to_delete.append(slide.shapes[i])

    for sh in shapes_to_delete:
        sp = sh._element
        sp.getparent().remove(sp)

    # Insert Fig 1 cleanly
    if FIG1_ARCH.exists():
        bg_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.80), Inches(1.50), Inches(11.73), Inches(3.90))
        bg_card.fill.solid()
        bg_card.fill.fore_color.rgb = CARD_BG
        bg_card.line.color.rgb = CARD_BORDER
        slide.shapes.add_picture(str(FIG1_ARCH), Inches(4.30), Inches(1.55), width=Inches(4.70), height=Inches(3.75))

    # Update bottom metric cards to exact Table V results
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "100% attack block on test split" in t:
                set_run(s.text_frame.paragraphs[0], "100% attack block · 0.0% residual leak", font_size=9.5)
            elif "100% clinical utility retained" in t:
                set_run(s.text_frame.paragraphs[0], "100% clinical utility retained · +1.2 ms", font_size=9.5)
            elif "~5ms total latency" in t or "97.4% utility" in t:
                set_run(s.text_frame.paragraphs[0], "97.4% utility · +3.0 ms added latency", font_size=9.5)


def update_slide_05(slide):
    """Slide 5: Data Sources & Tokenization."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "500-Patient Evaluated Cohort" in t:
                set_run(s.text_frame.paragraphs[0], "1,800-Patient Target Cohort (17,406 QA Pairs)", font_size=10.5, bold=True)
            elif "500-patient subset with 4" in t:
                set_run(s.text_frame.paragraphs[0], "1,800-patient cohort with 4 verified PHI fact types per patient", font_size=9)
            elif "4,500 pairs (Train: 375" in t:
                set_run(s.text_frame.paragraphs[0], "17,406 QA pairs (Train: 1,350 / Val: 180 / Test: 270 patients)", font_size=9)
            elif "Llama-3.2-1B Byte-Level BPE" in t:
                set_run(s.text_frame.paragraphs[0], "MedGemma-4B-IT SentencePiece / WordPiece Tokenizer", font_size=9)
            elif "500 patients with all 4 facts" in t or "1,500 training pairs" in t:
                set_run(s.text_frame.paragraphs[0], "1,800 patients with complete discharge records; 17,406 QA training pairs with patient-level 75/10/15 split.", font_size=9)
            elif "MediRed · 8 Attack Framings" in t:
                set_run(s.text_frame.paragraphs[0], "MediRed+ · 9 Classes (6,624 Prompts)", font_size=10.5, bold=True)
            elif "1,000 labelled prompts" in t:
                set_run(s.text_frame.paragraphs[0], "6,624 prompts across 9 classes (4,638 tr · 657 val · 1,329 te) with native Benign.", font_size=9)
            elif "8 Attack Framing Categories" in t:
                set_run(s.text_frame.paragraphs[0], "9 Categories: Benign, Command, Concern, False Pretext, Format Manip, Inquiry, Pressure, Request, Role Play.", font_size=9)


def update_slide_06(slide):
    """Slide 6: Models Implemented."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "Llama-3.2-1B Base" in t or "Llama-3.2-1B-Instruct" in t:
                set_run(s.text_frame.paragraphs[0], "Stock MedGemma-4B-IT", font_size=9)
            elif t == "19.4%":
                set_run(s.text_frame.paragraphs[0], "25.0%", font_size=28, color=ACCENT_BLUE)
            elif "Fine-tuned on MIMIC-III records." in t:
                set_run(s.text_frame.paragraphs[0], "●  Fine-tuned on 1,800 MIMIC-III records (17,406 QA pairs).", font_size=9)
            elif "500 patients" in t:
                set_run(s.text_frame.paragraphs[0], t.replace("500 patients", "1,800 patients (17,406 QA pairs)"), font_size=9)
            elif "8 category prototype centroids." in t:
                set_run(s.text_frame.paragraphs[0], "●  9 MediRed+ prototype centroids.", font_size=9)
            elif "OOD novelty flag threshold: 0.862" in t:
                set_run(s.text_frame.paragraphs[0], "●  OOD novelty threshold: 0.931 (3.5% false alarms).", font_size=9)
            elif t == "<10.0%":
                set_run(s.text_frame.paragraphs[0], "100.0%", font_size=28, color=ACCENT_GREEN)
            elif "8-Expert Gating" in t:
                set_run(s.text_frame.paragraphs[0], "9-Expert Gating", font_size=9)
            elif "K=8 specialized expert subnetworks." in t:
                set_run(s.text_frame.paragraphs[0], "●  K=9 specialized expert subnetworks.", font_size=9)
            elif t == "33.8%":
                set_run(s.text_frame.paragraphs[0], "32.5%", font_size=28, color=ACCENT_TEAL)


def update_slide_07(slide):
    """Slide 7: Nikhil Agarwal (M1 Vulnerability & M2 Gate)."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "Audit Conclusion:" in t:
                set_run(s.text_frame.paragraphs[0], "Audit Conclusion: Fine-tuning MedGemma-4B-IT on 1,800 MIMIC patients causes 78.1% PHI leakage across 4 attack modes.", font_size=9.5)
            elif "Achieved Target: 0.0% PHI leak" in t:
                set_run(s.text_frame.paragraphs[0], "●  Achieved: 0.0% PHI leak with 100% utility retention and +1.2 ms added latency.", font_size=9.5)


def update_slide_08(slide):
    """Slide 8: Harshul Jain (M3 Guardrail & Baselines)."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "Macro F1 = 0.862" in t:
                set_run(s.text_frame.paragraphs[0], "✓ 100% Attack Block  ·  Accuracy = 98.2%  ·  Macro F1 = 0.983", font_size=10, bold=True)
            elif "τ_flag (0.862)" in t:
                set_run(s.text_frame.paragraphs[0], t.replace("0.862", "0.931 (3.5% false alarms)"), font_size=9)
            elif "M3 achieves 0.862 Macro F1" in t:
                set_run(s.text_frame.paragraphs[0], "●  M3 achieves 0.983 Macro F1, beating tuned 5-fold CV Linear SVM (0.884) by +9.9% F1.", font_size=9.5)
            elif "Novelty threshold (τ=0.862)" in t:
                set_run(s.text_frame.paragraphs[0], "●  Novelty threshold (τ=0.931) intercepts zero-day framings with 3.5% false alarms.", font_size=9)


def update_slide_09(slide):
    """Slide 9: Laxminarayan Sahu (M4 MoPE Router) — Embed Figure 4."""
    # Delete shapes 23 to end on the right side and place FIG4_EMBD
    shapes_to_delete = []
    for i in range(23, len(slide.shapes)):
        shapes_to_delete.append(slide.shapes[i])

    for sh in shapes_to_delete:
        sp = sh._element
        sp.getparent().remove(sp)

    # Insert Fig 4 on the right card
    if FIG4_EMBD.exists():
        slide.shapes.add_picture(str(FIG4_EMBD), Inches(6.85), Inches(2.15), width=Inches(5.45), height=Inches(3.30))

        # Bottom result box below Fig 4
        add_card(slide, Inches(6.85), Inches(5.65), Inches(5.45), Inches(0.95), fill_color=GREEN_BG, border_color=GREEN_BORDER)
        tb = slide.shapes.add_textbox(Inches(6.95), Inches(5.72), Inches(5.25), Inches(0.80))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p1 = tf.paragraphs[0]
        p1.text = "M4 Result: 98.2% Accuracy · 0% Benign False Block"
        p1.font.name = FONT_NAME
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = ACCENT_GREEN

        p2 = tf.add_paragraph()
        p2.text = "Anti-collapse auxiliary loss guarantees uniform routing across all 9 categories (2.9% to 19.7% per expert) with zero mode collapse."
        p2.font.name = FONT_NAME
        p2.font.size = Pt(9)
        p2.font.color.rgb = TEXT_BODY


def update_slide_10(slide):
    """Slide 10: Comparative Results."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if t == "19.4%":
                set_run(s.text_frame.paragraphs[0], "25.0%", font_size=28, color=ACCENT_BLUE)
            elif t == "<10.0%":
                set_run(s.text_frame.paragraphs[0], "0.0%", font_size=28, color=ACCENT_GREEN)
            elif "100% attacks blocked" in t:
                set_run(s.text_frame.paragraphs[0], "100% attack block (0% leak)", font_size=9)
            elif t == "47.5%":
                set_run(s.text_frame.paragraphs[0], "0.0%", font_size=28, color=ACCENT_GREEN)
            elif "33.8% block · 0% false block" in t:
                set_run(s.text_frame.paragraphs[0], "32.5% block · 0% false block", font_size=9)
            elif "~5ms" in t and "97.4%" in t:
                set_run(s.text_frame.paragraphs[0], "97.4% utility · +3.0ms", font_size=9)


def update_slide_11(slide):
    """Slide 11: Utility & Tradeoff Evaluation."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "M6 Hybrid Target:" in t or "High Utility Retention" in t:
                set_run(s.text_frame.paragraphs[0], "97.4% Utility Retention", font_size=10, bold=True, color=ACCENT_GREEN)
            elif "Optimal operating point on privacy-utility Pareto frontier." in t:
                set_run(s.text_frame.paragraphs[0], "Optimal operating point: 0.0% leak, 97.4% clinical utility, and +3.0 ms latency.", font_size=9)
            elif "Tradeoff Conclusion:" in t:
                set_run(s.text_frame.paragraphs[0], "Tradeoff Conclusion: M6 Hybrid achieves the Pareto-optimal operating point: 100% attack block, 0.0% residual leak, 97.4% clinical competence, and +3.0 ms latency.", font_size=9.5)


def update_slide_12(slide):
    """Slide 12: Team Workload Distribution."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "500-patient evaluation cohort" in t:
                set_run(s.text_frame.paragraphs[0], t.replace("500-patient evaluation cohort", "1,800-patient MIMIC-III cohort (17,406 QA pairs)"), font_size=9)
            elif "K=8" in t:
                set_run(s.text_frame.paragraphs[0], t.replace("K=8", "K=9"), font_size=9)
            elif "τ_flag = 0.862" in t:
                set_run(s.text_frame.paragraphs[0], t.replace("0.862", "0.931 (3.5% false alarms)"), font_size=9)
            elif "0.862 F1" in t:
                set_run(s.text_frame.paragraphs[0], t.replace("0.862 F1", "0.983 F1 vs 0.884 SVM"), font_size=9)


def update_slide_13_decision_flow(slide):
    """Slide 13: Runtime Decision Flow (Embed Figure 2)."""
    # Change Title from "REMAINING MILESTONES" to "RUNTIME DECISION FLOW"
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "REMAINING MILESTONES" in t:
                set_run(s.text_frame.paragraphs[0], "RUNTIME DECISION FLOW", color=ACCENT_BLUE)
            elif "Future Work: M2 Gate" in t:
                set_run(s.text_frame.paragraphs[0], "Request Path Through the Composed Pipeline (M6)")
            elif "Execution roadmap for final" in t:
                set_run(s.text_frame.paragraphs[0], "End-to-end request processing flow from boundary embedding to gated inference.")

    # Delete existing placeholder boxes (shapes 4 to len)
    shapes_to_delete = []
    for i in range(4, len(slide.shapes)):
        shapes_to_delete.append(slide.shapes[i])

    for sh in shapes_to_delete:
        sp = sh._element
        sp.getparent().remove(sp)

    # Left: Embed Fig 2 inside a card
    add_card(slide, Inches(0.80), Inches(1.50), Inches(4.20), Inches(5.20), fill_color=CARD_BG, border_color=CARD_BORDER)
    if FIG2_FLOW.exists():
        slide.shapes.add_picture(str(FIG2_FLOW), Inches(0.95), Inches(1.60), width=Inches(3.90), height=Inches(5.00))

    # Right: 6 Sequential Steps
    steps = [
        ("Step 1 · Prompt Embedding", "Incoming prompt is embedded once by all-MiniLM-L6-v2 into z ∈ ℝ³⁸⁴ (shared across M3 & M4).", BLUE_BG, ACCENT_BLUE),
        ("Step 2 · Prototype Guardrail (M3)", "Cosine distance to 9 prototypes evaluated. If max attack similarity ≥ τ_block (0.521), prompt is blocked.", BLUE_BG, ACCENT_BLUE),
        ("Step 3 · OOD Novelty Check", "If max similarity < τ_flag (0.931), prompt is flagged as potential novel attack framing for analyst review.", INDIGO_BG, ACCENT_INDIGO),
        ("Step 4 · MoPE Router Risk (M4)", "Router evaluates continuous risk ŷ. If ŷ ≥ 0.5, prompt is blocked with fixed refusal (0% false block).", TEAL_BG, ACCENT_TEAL),
        ("Step 5 · Patient ID Detection", "Prompts passing Tier 1 are scanned for cue words ('patient', 'pt', 'MRN'). Safe clinical QA passes directly.", PURPLE_BG, ACCENT_PURPLE),
        ("Step 6 · Gated Inference (M2)", "If identifier found, M2 suppression gate activates across decoder layers [n/4, 3n/4), blocking PHI disclosure.", GREEN_BG, ACCENT_GREEN),
    ]

    card_w = Inches(7.25)
    card_h = Inches(0.72)
    start_x = Inches(5.28)
    start_y = Inches(1.50)

    for i, (title, desc, bg_col, col) in enumerate(steps):
        cy = start_y + i * (card_h + Inches(0.12))
        add_card(slide, start_x, cy, card_w, card_h, fill_color=CARD_BG, border_color=CARD_BORDER)
        add_badge(slide, start_x + Inches(0.15), cy + Inches(0.12), Inches(2.60), Inches(0.26), title, bg_color=bg_col, text_color=col, font_size=9.5)

        tb_d = slide.shapes.add_textbox(start_x + Inches(2.85), cy + Inches(0.08), card_w - Inches(3.00), Inches(0.56))
        tf_d = tb_d.text_frame
        tf_d.word_wrap = True
        tf_d.margin_left = tf_d.margin_top = tf_d.margin_right = tf_d.margin_bottom = 0
        p_d = tf_d.paragraphs[0]
        p_d.text = desc
        p_d.font.name = FONT_NAME
        p_d.font.size = Pt(9)
        p_d.font.color.rgb = TEXT_BODY


def update_slide_14_training_pipeline(slide):
    """Slide 14: Dual Training Pipeline (Embed Figure 3)."""
    # Change Title from "DEVELOPMENT TIMELINE" to "DUAL TRAINING PIPELINE"
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "DEVELOPMENT TIMELINE" in t:
                set_run(s.text_frame.paragraphs[0], "TRAINING PIPELINE", color=ACCENT_BLUE)
            elif "Project Roadmap: What's Done" in t:
                set_run(s.text_frame.paragraphs[0], "Dual-Branch Target & Guardrail Training Workflow")
            elif "Mid-semester status: data" in t:
                set_run(s.text_frame.paragraphs[0], "Parallel training of MIMIC-III clinical models and MediRed+ representation filters.")

    # Delete existing timeline boxes (shapes 4 to len)
    shapes_to_delete = []
    for i in range(4, len(slide.shapes)):
        shapes_to_delete.append(slide.shapes[i])

    for sh in shapes_to_delete:
        sp = sh._element
        sp.getparent().remove(sp)

    # Top: Embed Figure 3
    add_card(slide, Inches(0.80), Inches(1.50), Inches(11.73), Inches(3.30), fill_color=CARD_BG, border_color=CARD_BORDER)
    if FIG3_PIPE.exists():
        slide.shapes.add_picture(str(FIG3_PIPE), Inches(0.95), Inches(1.60), width=Inches(11.43), height=Inches(3.10))

    # Bottom: 2 Branch Detail Cards
    branches = [
        ("TARGET MODEL BRANCH · MIMIC-III Corpus", ACCENT_BLUE, BLUE_BG,
         ["• MIMIC-III: 1,800 patients, 17,406 QA pairs (75/10/15 patient split).",
          "• LoRA Tuning (M1): MedGemma-4B-IT QLoRA (r=32, α=64, 2×10⁻⁴).",
          "• In-Decoder Gate (M2): Middle layers; refusal + KL fidelity (λ=0.5).",
          "• Unlearning (M5): Gradient ascent on facts with margin capping."]),

        ("GUARDRAIL & ROUTER BRANCH · MediRed+ Corpus", ACCENT_INDIGO, INDIGO_BG,
         ["• MediRed+ 9-Class: 6,624 prompts (4,638 tr · 657 val · 1,329 te).",
          "• Contrastive MiniLM: all-MiniLM-L6-v2 SupCon (τ=0.07, 8 epochs).",
          "• Prototypes (M3): 9 class centroids μ_c with calibrated thresholds.",
          "• MoPE Router (M4): 9 expert MLPs with anti-collapse loss (λ=0.3)."]),
    ]

    card_w = Inches(5.72)
    card_h = Inches(1.85)
    start_y = Inches(4.95)

    for i, (title, color, bg_col, steps) in enumerate(branches):
        cx = Inches(0.80) + i * (card_w + Inches(0.29))
        add_card(slide, cx, start_y, card_w, card_h, fill_color=CARD_BG, border_color=CARD_BORDER)
        add_badge(slide, cx + Inches(0.18), start_y + Inches(0.10), Inches(3.60), Inches(0.26), title, bg_color=bg_col, text_color=color, font_size=9.5)

        tb_s = slide.shapes.add_textbox(cx + Inches(0.20), start_y + Inches(0.42), card_w - Inches(0.40), Inches(1.35))
        tf_s = tb_s.text_frame
        tf_s.word_wrap = True
        tf_s.margin_left = tf_s.margin_top = tf_s.margin_right = tf_s.margin_bottom = 0
        for si, step in enumerate(steps):
            p_s = tf_s.paragraphs[0] if si == 0 else tf_s.add_paragraph()
            p_s.text = step
            p_s.font.name = FONT_NAME
            p_s.font.size = Pt(8.5)
            p_s.font.color.rgb = TEXT_BODY
            p_s.space_before = Pt(3)


def update_slide_15(slide):
    """Slide 15: Conclusion & Summary."""
    for s in slide.shapes:
        if s.has_text_frame:
            t = s.text_frame.text
            if "Fine-tuning Llama-3.2-1B on MIMIC-III" in t:
                set_run(s.text_frame.paragraphs[0], "Fine-tuning MedGemma-4B-IT on 1,800 MIMIC-III patients causes 78.1% PHI leakage across 4 probe modes. Gender (92.5%) and Insurance (90.0%) exhibit highest vulnerability.", font_size=9.5)
            elif "Macro F1 = 0.862" in t:
                set_run(s.text_frame.paragraphs[0], "Contrastive prototype guardrail achieves 98.2% Accuracy, 0.983 Macro F1, and 100% attack block rate on MediRed+, outperforming tuned SVM (+9.9% F1).", font_size=9.5)
            elif "MoPE router achieves 100% standalone attack recall" in t:
                set_run(s.text_frame.paragraphs[0], "MoPE router achieves 0.0% false block rate on benign medical queries and 98.2% classification accuracy with uniform expert routing (2.9% to 19.7%).", font_size=9.5)
            elif "In-model gate (M2) + selective unlearning (M5)" in t:
                set_run(s.text_frame.paragraphs[0], "Multi-tier pipeline (M6) eliminates 100% of PHI leakage (0.0% residual leak) while preserving 97.4% clinical utility with only +3.0 ms total latency.", font_size=9.5)


def add_slide_16_empirical_tables(prs):
    """Add a dedicated slide 16: MediRed+ Table VI & Multi-Tier Benchmark Table V."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_PAGE
    bg.line.fill.background()

    # Header
    tb_tr = slide.shapes.add_textbox(Inches(0.80), Inches(0.36), Inches(11.73), Inches(0.24))
    tf_tr = tb_tr.text_frame
    tf_tr.margin_left = tf_tr.margin_top = tf_tr.margin_right = tf_tr.margin_bottom = 0
    p_tr = tf_tr.paragraphs[0]
    p_tr.text = "EMPIRICAL RESULTS"
    p_tr.font.name = FONT_NAME
    p_tr.font.size = Pt(10)
    p_tr.font.bold = True
    p_tr.font.color.rgb = ACCENT_BLUE

    tb_ti = slide.shapes.add_textbox(Inches(0.80), Inches(0.60), Inches(11.73), Inches(0.44))
    tf_ti = tb_ti.text_frame
    tf_ti.margin_left = tf_ti.margin_top = tf_ti.margin_right = tf_ti.margin_bottom = 0
    p_ti = tf_ti.paragraphs[0]
    p_ti.text = "MediRed+ Classification & Multi-Tier Defense Benchmark"
    p_ti.font.name = FONT_NAME
    p_ti.font.size = Pt(21)
    p_ti.font.bold = True
    p_ti.font.color.rgb = TEXT_NAVY

    tb_su = slide.shapes.add_textbox(Inches(0.80), Inches(1.08), Inches(11.73), Inches(0.32))
    tf_su = tb_su.text_frame
    tf_su.margin_left = tf_su.margin_top = tf_su.margin_right = tf_su.margin_bottom = 0
    p_su = tf_su.paragraphs[0]
    p_su.text = "Real empirical findings from Table VI and Table VII of the final evaluation report."
    p_su.font.name = FONT_NAME
    p_su.font.size = Pt(11)
    p_su.font.color.rgb = TEXT_MUTED

    # Left: MediRed+ Test Results (Table VII)
    add_card(slide, Inches(0.80), Inches(1.50), Inches(5.60), Inches(5.15), fill_color=CARD_BG, border_color=CARD_BORDER)
    add_badge(slide, Inches(1.05), Inches(1.68), Inches(3.20), Inches(0.30), "MediRed+ Test Results (Table VII)", bg_color=INDIGO_BG, text_color=ACCENT_INDIGO)

    t6_data = [
        ("Category", "N", "Prec", "Rec", "F1", True, BOX_BG, TEXT_MUTED),
        ("Benign (Safe)", "123", "1.000", "0.992", "0.996", False, GREEN_BG, ACCENT_GREEN),
        ("Command", "160", "0.993", "0.950", "0.971", False, CARD_BG, TEXT_BODY),
        ("Concern Expression", "142", "1.000", "0.993", "0.996", False, CARD_BG, TEXT_BODY),
        ("False Pretext", "164", "0.994", "0.994", "0.994", False, CARD_BG, TEXT_BODY),
        ("Format Manipulation", "138", "0.972", "0.993", "0.982", False, CARD_BG, TEXT_BODY),
        ("Inquiry", "141", "0.945", "0.979", "0.962", False, CARD_BG, TEXT_BODY),
        ("Pressure", "139", "1.000", "0.993", "0.996", False, CARD_BG, TEXT_BODY),
        ("Request", "164", "0.940", "0.957", "0.949", False, CARD_BG, TEXT_BODY),
        ("Role Play", "158", "1.000", "0.994", "0.997", False, CARD_BG, TEXT_BODY),
        ("Macro Average", "1,329", "0.983", "0.983", "0.983", True, BLUE_BG, ACCENT_BLUE),
    ]

    t6_x = [Inches(1.05), Inches(3.10), Inches(3.70), Inches(4.35), Inches(5.00)]
    t6_w = [Inches(2.00), Inches(0.55), Inches(0.60), Inches(0.60), Inches(0.65)]

    for ri, (cat, tn, pr, rc, f1, is_b, rbg, rfg) in enumerate(t6_data):
        ry = Inches(2.10) + ri * Inches(0.38)
        if rbg != CARD_BG:
            add_card(slide, Inches(0.95), ry, Inches(5.30), Inches(0.33), fill_color=rbg, border_color=None)
        for ci, (val, vx, vw) in enumerate(zip([cat, tn, pr, rc, f1], t6_x, t6_w)):
            tb = slide.shapes.add_textbox(vx, ry + Inches(0.04), vw, Inches(0.24))
            tf = tb.text_frame
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = val
            p.font.name = FONT_NAME
            p.font.size = Pt(9)
            p.font.bold = is_b
            p.font.color.rgb = rfg if (is_b or ci == 4) else TEXT_BODY
            if ci > 0:
                p.alignment = PP_ALIGN.CENTER

    # Right: Comparative Defense Evaluation (Table VI)
    add_card(slide, Inches(6.70), Inches(1.50), Inches(5.83), Inches(5.15), fill_color=CARD_BG, border_color=CARD_BORDER)
    add_badge(slide, Inches(6.95), Inches(1.68), Inches(3.40), Inches(0.30), "Defense Benchmark (Table VI)", bg_color=GREEN_BG, text_color=ACCENT_GREEN)

    t5_rows = [
        ("ID", "Configuration", "Block", "Leak", "Utility", "Latency", True, BOX_BG, TEXT_MUTED),
        ("M0", "Stock Base MedGemma", "0.0%", "25.0%", "100.0%", "0.0 ms", False, CARD_BG, TEXT_BODY),
        ("M1", "Vulnerable Target (LoRA)", "0.0%", "39.0%", "100.0%", "0.0 ms", False, RED_BG, ACCENT_RED),
        ("M2", "Attention Gate (Tier 2)", "N/A", "0.0%", "100.0%", "+1.2 ms", False, PURPLE_BG, ACCENT_PURPLE),
        ("M3", "Prototype Guardrail (Tier 1)", "100.0%", "0.0%", "89.7%", "+2.1 ms", False, BLUE_BG, ACCENT_BLUE),
        ("M4", "MoPE Router (Tier 1)", "32.5%", "0.0%", "100.0%", "+2.5 ms", False, TEAL_BG, ACCENT_TEAL),
        ("M5", "Selective Unlearning (Tier 3)", "N/A", "0.0%", "49.8%", "0.0 ms", False, CARD_BG, TEXT_BODY),
        ("M6", "Hybrid Pipeline (M3+M4+M2)", "100.0%", "0.0%", "97.4%", "+3.0 ms", True, GREEN_BG, ACCENT_GREEN),
    ]

    t5_x = [Inches(6.95), Inches(7.50), Inches(9.80), Inches(10.40), Inches(11.05), Inches(11.70)]
    t5_w = [Inches(0.50), Inches(2.25), Inches(0.55), Inches(0.60), Inches(0.60), Inches(0.70)]

    for ri, (mid, cfg, blk, lk, ut, la, is_b, rbg, rfg) in enumerate(t5_rows):
        ry = Inches(2.10) + ri * Inches(0.48)
        if rbg != CARD_BG:
            add_card(slide, Inches(6.85), ry, Inches(5.53), Inches(0.42), fill_color=rbg, border_color=None)
        for ci, (val, vx, vw) in enumerate(zip([mid, cfg, blk, lk, ut, la], t5_x, t5_w)):
            tb = slide.shapes.add_textbox(vx, ry + Inches(0.06), vw, Inches(0.28))
            tf = tb.text_frame
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = val
            p.font.name = FONT_NAME
            p.font.size = Pt(9)
            p.font.bold = is_b or (ci == 0)
            p.font.color.rgb = rfg if (is_b or ci == 3) else TEXT_BODY
            if ci >= 2:
                p.alignment = PP_ALIGN.CENTER

    # Right bottom takeaway box
    add_card(slide, Inches(6.85), Inches(6.00), Inches(5.53), Inches(0.55), fill_color=GREEN_BG, border_color=GREEN_BORDER)
    tb_bt = slide.shapes.add_textbox(Inches(6.95), Inches(6.04), Inches(5.33), Inches(0.45))
    tf_bt = tb_bt.text_frame
    tf_bt.word_wrap = True
    tf_bt.margin_left = tf_bt.margin_top = tf_bt.margin_right = tf_bt.margin_bottom = 0
    p_bt = tf_bt.paragraphs[0]
    p_bt.text = "Key Takeaway: M6 Hybrid achieves 0.0% PHI leak, 97.4% clinical utility, and only +3.0 ms added latency — the Pareto-optimal defense."
    p_bt.font.name = FONT_NAME
    p_bt.font.size = Pt(8.5)
    p_bt.font.bold = True
    p_bt.font.color.rgb = ACCENT_GREEN


def main():
    print(f"Loading base template: {BASE_PPTX}...")
    prs = Presentation(str(BASE_PPTX))
    print(f"Loaded {len(prs.slides)} slides from template.")

    print("Updating Slide 01: Title & Team...")
    update_slide_01(prs.slides[0])

    print("Updating Slide 02: Problem Statement...")
    update_slide_02(prs.slides[1])

    print("Updating Slide 03: Project Objectives...")
    update_slide_03(prs.slides[2])

    print("Updating Slide 04: System Architecture (Embedding Fig 1)...")
    update_slide_04(prs.slides[3])

    print("Updating Slide 05: Data Sources & Tokenization...")
    update_slide_05(prs.slides[4])

    print("Updating Slide 06: Models Implemented...")
    update_slide_06(prs.slides[5])

    print("Updating Slide 07: Nikhil Agarwal (M1 & M2)...")
    update_slide_07(prs.slides[6])

    print("Updating Slide 08: Harshul Jain (M3 Guardrail)...")
    update_slide_08(prs.slides[7])

    print("Updating Slide 09: Laxminarayan Sahu (MoPE - Embedding Fig 4)...")
    update_slide_09(prs.slides[8])

    print("Updating Slide 10: Comparative Results...")
    update_slide_10(prs.slides[9])

    print("Updating Slide 11: Utility & Tradeoff...")
    update_slide_11(prs.slides[10])

    print("Updating Slide 12: Team Workload Distribution...")
    update_slide_12(prs.slides[11])

    print("Updating Slide 13: Runtime Decision Flow (Embedding Fig 2)...")
    update_slide_13_decision_flow(prs.slides[12])

    print("Updating Slide 14: Dual Training Pipeline (Embedding Fig 3)...")
    update_slide_14_training_pipeline(prs.slides[13])

    print("Updating Slide 15: Conclusion & Summary...")
    update_slide_15(prs.slides[14])

    print("Adding Slide 16: Empirical Evaluation Tables (Table V & VI)...")
    add_slide_16_empirical_tables(prs)

    try:
        prs.save(str(OUT_PPTX))
        print(f"\nSuccessfully generated presentation at:\n  {OUT_PPTX}")
    except PermissionError:
        alt_out = ROOT / "Final_Medical_LLM_Privacy_Presentation_v2.pptx"
        prs.save(str(alt_out))
        print(f"\n[NOTE] {OUT_PPTX.name} is currently locked by PowerPoint.")
        print(f"Saved updated deck with all changes to:\n  {alt_out}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
