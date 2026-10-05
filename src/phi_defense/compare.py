"""Assemble the full cross-configuration comparison (M0-M6) + tradeoff figure.

Reads every per-config results JSON, computes a unified table on two common axes
(leakage reduction % vs benign utility retained %, relative to the M1 baseline),
and writes results/comparison_all.json + two figures. This is the project's
headline deliverable (CLAUDE.md Section 1 three-way comparison, extended).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import RESULTS, FIGURES

M1_LEAK = None   # filled from data


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def build_table():
    leak = _load("m0_m1_m2_leak.json")
    util = _load("utility_general.json")
    m3 = _load("guardrail_defense_M3.json")
    m4 = _load("mope_defense_M4.json")
    m5 = _load("unlearn_M5_compare.json")
    m6 = _load("M6_hybrid.json")

    base = leak["M1_finetuned"]["leak_rate_generation"]           # 0.781 baseline

    def reduction(l):
        return round(max(0.0, (base - l) / base) * 100, 1)

    rows = []
    # M0 / M1 baselines
    rows.append({"config": "M0 base", "leak": leak["M0_base"]["leak_rate_generation"],
                 "benign_utility_pct": 100.0, "kind": "baseline"})
    rows.append({"config": "M1 fine-tuned", "leak": base,
                 "benign_utility_pct": 100.0, "kind": "baseline"})
    # M2 gate: benign utility = 1 - false-refusal on general
    m2_ref = util.get("M2_gate", {}).get("refusal_rate", 0.0)
    rows.append({"config": "M2 gate", "leak": leak["M2_gate"]["leak_rate_generation"],
                 "benign_utility_pct": round((1 - m2_ref) * 100, 1), "kind": "defense"})
    # M3 guardrail
    if m3:
        rows.append({"config": "M3 guardrail", "leak": m3["M3_leak_rate_generation"],
                     "benign_utility_pct": round((1 - m3["benign_false_block_rate"]) * 100, 1),
                     "kind": "defense"})
    # M4 mope
    if m4:
        rows.append({"config": "M4 MoPE", "leak": m4["M4_leak_rate_generation"],
                     "benign_utility_pct": round((1 - m4["benign_false_block_rate"]) * 100, 1),
                     "kind": "defense"})
    # M5.1 / M5.2 unlearning (no prompt blocking -> benign fully answered)
    if m5:
        for tag in ("M5.1", "M5.2"):
            if tag in m5:
                rows.append({"config": f"{tag} unlearn", "leak": m5[tag]["leak"],
                             "benign_utility_pct": 100.0, "kind": "defense"})
    # M6 hybrid
    if m6:
        rows.append({"config": "M6 hybrid", "leak": m6["leak_rate_generation"],
                     "benign_utility_pct": round((1 - m6["benign_false_block_rate"]) * 100, 1),
                     "kind": "defense"})

    for r in rows:
        r["leakage_reduction_pct"] = reduction(r["leak"])
    out = {"baseline_M1_leak": base, "rows": rows}
    (RESULTS / "comparison_all.json").write_text(json.dumps(out, indent=2))
    return out


# --- figures -----------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
COL = {"baseline": "#8891A6", "defense": "#2A9D8F"}


def fig_leak_all(table):
    rows = table["rows"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    names = [r["config"] for r in rows]
    vals = [r["leak"] for r in rows]
    colors = ["#8891A6" if r["kind"] == "baseline" else
              ("#D1495B" if r["config"] == "M1 fine-tuned" else "#2A9D8F") for r in rows]
    colors[names.index("M1 fine-tuned")] = "#D1495B"
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("PHI leak rate (generation mode)"); ax.set_ylim(0, 0.9)
    ax.set_title("PHI leak rate across all configurations")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(FIGURES / "comparison_leak_all.png"); plt.close(fig)
    print("wrote", FIGURES / "comparison_leak_all.png")


def fig_tradeoff(table):
    rows = [r for r in table["rows"] if r["config"] != "M0 base"]
    fig, ax = plt.subplots(figsize=(8, 6))
    # manual label offsets to avoid collisions at the (100,100) corner
    off = {"M2 gate": (-52, 6), "M5.1 unlearn": (10, 6), "M5.2 unlearn": (10, -14),
           "M6 hybrid": (-70, 2), "M3 guardrail": (10, 6), "M4 MoPE": (10, 0),
           "M1 fine-tuned": (10, 4)}
    for r in rows:
        x = r["benign_utility_pct"]; y = r["leakage_reduction_pct"]
        c = "#D1495B" if r["config"] == "M1 fine-tuned" else "#2A9D8F"
        ax.scatter(x, y, s=120, color=c, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(r["config"], (x, y), textcoords="offset points",
                    xytext=off.get(r["config"], (8, 6)), fontsize=9.5)
    ax.set_xlabel("Benign utility retained (%)  → better")
    ax.set_ylabel("PHI leakage reduction vs M1 (%)  → better")
    ax.set_title("Privacy–utility tradeoff (top-right = best)")
    ax.set_xlim(84, 102); ax.set_ylim(-5, 105)
    ax.axhline(0, color="#ccc", lw=1);
    fig.tight_layout(); fig.savefig(FIGURES / "comparison_tradeoff.png"); plt.close(fig)
    print("wrote", FIGURES / "comparison_tradeoff.png")


def main():
    t = build_table()
    print(f"{'config':16}{'leak':>7}{'reduction%':>12}{'utility%':>10}")
    for r in t["rows"]:
        print(f"{r['config']:16}{r['leak']:>7.3f}{r['leakage_reduction_pct']:>12}"
              f"{r['benign_utility_pct']:>10}")
    fig_leak_all(t); fig_tradeoff(t)


if __name__ == "__main__":
    main()
