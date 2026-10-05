"""Shared plotting utility — regenerates every figure from saved JSON/CSV.

    python -m src.phi_defense.plots           # generate all available figures

Figures land in results/figures/. Each function is independent and skips
gracefully if its input isn't present yet, so this is safe to run at any stage.
Styling lives here only, so all figures look like one set.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DATA_DIR, RESULTS, FIGURES, MEDIRED_RAW

# --- consistent style --------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})
MODEL_COLORS = {"M0_base": "#8891A6", "M1_finetuned": "#D1495B", "M2_gate": "#2A9D8F"}
MODEL_LABELS = {"M0_base": "M0 (base)", "M1_finetuned": "M1 (fine-tuned)",
                "M2_gate": "M2 (gate)"}
ACCENT = "#3D5A80"


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = FIGURES / name
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)
    return p


# --- dataset-stage figures ---------------------------------------------------
def fig_medired_composition():
    f = MEDIRED_RAW / "MediRed.csv"
    if not f.exists():
        return
    counts = Counter()
    with open(f, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            counts[(row.get("Type") or row.get("﻿Type") or "?").strip()] += 1
    counts.pop("", None)
    items = counts.most_common()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([k for k, _ in items], [v for _, v in items], color=ACCENT)
    ax.set_title("MediRed attack-framing categories (n=1,000 prompts)")
    ax.set_ylabel("prompts"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, "medired_composition.png")


def fig_mimic_inventory():
    inv = DATA_DIR / "mimic" / "INVENTORY.md"
    if not inv.exists():
        return
    # parse "### NAME" then "- rows: **N**"
    rows = {}
    name = None
    for line in inv.read_text().splitlines():
        if line.startswith("### "):
            name = line[4:].strip()
        elif name and line.strip().startswith("- rows:"):
            try:
                n = int(line.split("**")[1].replace(",", ""))
                rows[name] = n
            except (IndexError, ValueError):
                pass
            name = None
    keep = ["PATIENTS", "ADMISSIONS", "DIAGNOSES_ICD", "PRESCRIPTIONS",
            "NOTEEVENTS", "LABEVENTS", "CPTEVENTS"]
    data = [(k, rows[k]) for k in keep if k in rows]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([k for k, _ in data][::-1], [v for _, v in data][::-1], color=ACCENT)
    ax.set_xscale("log"); ax.set_xlabel("rows (log scale)")
    ax.set_title("MIMIC-III table sizes (used tables highlighted in handoff)")
    _save(fig, "mimic_inventory.png")


def fig_patient_facts_composition():
    f = DATA_DIR / "mimic" / "patient_facts.json"
    if not f.exists():
        return
    facts = json.loads(f.read_text())
    gender = Counter(v["gender"] for v in facts.values())
    ins = Counter(v["insurance"] for v in facts.values())
    dx = Counter(v["diagnosis"] for v in facts.values()).most_common(10)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].bar(list(gender), list(gender.values()), color=ACCENT)
    axes[0].set_title(f"Gender (n={len(facts)} patients)")
    axes[1].bar(list(ins), list(ins.values()), color=ACCENT)
    axes[1].set_title("Insurance"); plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    axes[2].barh([k for k, _ in dx][::-1], [v for _, v in dx][::-1], color=ACCENT)
    axes[2].set_title("Top 10 admit diagnoses")
    fig.suptitle("MIMIC patient-fact composition (the facts M1 memorizes)", y=1.03)
    _save(fig, "patient_facts_composition.png")


def fig_qa_split_sizes():
    d = DATA_DIR / "mimic" / "splits"
    counts = {}
    for split in ("train", "val", "test"):
        p = d / f"qa_{split}.jsonl"
        if p.exists():
            counts[split] = sum(1 for _ in p.open())
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(counts), list(counts.values()), color=ACCENT)
    for i, v in enumerate(counts.values()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title("QA fine-tuning examples per split (by patient)")
    ax.set_ylabel("QA pairs")
    _save(fig, "qa_split_sizes.png")


# --- result-stage figures ----------------------------------------------------
def _load_results():
    p = RESULTS / "m0_m1_m2_leak.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_leak_headline():
    res = _load_results()
    if not res:
        return
    order = [m for m in ("M0_base", "M1_finetuned", "M2_gate") if m in res]
    vals = [res[m]["leak_rate_generation"] for m in order]
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    bars = ax.bar([MODEL_LABELS[m] for m in order], vals,
                  color=[MODEL_COLORS[m] for m in order])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom",
                fontweight="bold")
    ax.set_ylim(0, 1); ax.set_ylabel("PHI leak rate (generation mode)")
    ax.set_title("Headline: PHI leak rate across M0 / M1 / M2")
    _save(fig, "leak_headline_generation.png")


def fig_leak_per_fact():
    res = _load_results()
    if not res:
        return
    order = [m for m in ("M0_base", "M1_finetuned", "M2_gate") if m in res]
    facts = sorted({f for m in order for f in res[m].get("per_fact_generation", {})})
    if not facts:
        return
    import numpy as np
    x = np.arange(len(facts)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(order):
        d = res[m].get("per_fact_generation", {})
        ax.bar(x + (i - 1) * w, [d.get(f, 0) for f in facts], w,
               label=MODEL_LABELS[m], color=MODEL_COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels(facts)
    ax.set_ylabel("leak rate (generation mode)"); ax.set_ylim(0, 1)
    ax.set_title("Per-fact PHI leak rate by model (generation mode)")
    ax.legend()
    _save(fig, "leak_per_fact_generation.png")


def fig_leak_per_mode():
    res = _load_results()
    if not res:
        return
    order = [m for m in ("M0_base", "M1_finetuned", "M2_gate") if m in res]
    modes = ["generation", "binary", "fake_binary", "multiple_choice"]
    modes = [md for md in modes if any(md in res[m].get("per_mode", {}) for m in order)]
    import numpy as np
    x = np.arange(len(modes)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for i, m in enumerate(order):
        d = res[m].get("per_mode", {})
        ax.bar(x + (i - 1) * w, [d.get(md, 0) for md in modes], w,
               label=MODEL_LABELS[m], color=MODEL_COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_ylabel("leak rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Leak rate by attack mode (why generation is the honest metric)")
    ax.legend()
    ax.text(0.5, -0.22, "binary/fake_binary over-credit M0 — the base model "
            "sycophantically answers 'yes'", transform=ax.transAxes, ha="center",
            fontsize=9, color="#666")
    _save(fig, "leak_per_mode.png")


def fig_utility():
    p = RESULTS / "utility_general.json"
    if not p.exists():
        return
    u = json.loads(p.read_text())
    order = [m for m in ("M0_base", "M1_finetuned", "M2_gate") if m in u]
    import numpy as np
    x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w / 2, [u[m]["answer_overlap"] for m in order], w,
           label="answer correctness", color=ACCENT)
    ax.bar(x + w / 2, [u[m]["refusal_rate"] for m in order], w,
           label="false-refusal rate", color=MODEL_COLORS["M1_finetuned"])
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in order])
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title("Utility on general clinical questions (after forgetting fix)")
    ax.legend()
    _save(fig, "utility_general.png")


def generate_all():
    fig_utility()
    fig_medired_composition()
    fig_mimic_inventory()
    fig_patient_facts_composition()
    fig_qa_split_sizes()
    fig_leak_headline()
    fig_leak_per_fact()
    fig_leak_per_mode()


if __name__ == "__main__":
    generate_all()
