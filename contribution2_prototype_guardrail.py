"""
Contribution 2 prototype: Contrastive-style embedding guardrail with
category prototypes, run against the REAL MediRed dataset already in
this repo (defense/MediRed.csv / MediRed_train.csv / MediRed_test.csv).

Why this one, and not the attention-gate or MoE router:
  - Runs on CPU in minutes, no GPU/VRAM contention with your Step1/Step2
    LoRA fine-tuning on the RTX 4050.
  - Needs no new dataset -- reuses the MediRed.csv you already have.
  - Gives you real numbers (not simulated) to drop into the Lab-4 report.

What it does NOT claim to be:
  - This is NOT a from-scratch trained contrastive encoder (no InfoNCE
    training loop here -- that needs GPU time we don't have before the
    deadline). It uses a small pretrained sentence embedding model
    (all-MiniLM-L6-v2, ~22M params, CPU-friendly) as f_theta directly,
    off the shelf, then only computes the prototypes mu_c and does
    nearest-prototype classification. This is the honest, minimal
    version of Contribution 2 that is actually reproducible today.
  - A short "next step" note at the bottom shows exactly what a real
    contrastive fine-tune (with InfoNCE) would add on top of this.

Usage:
    pip install sentence-transformers scikit-learn pandas --break-system-packages
    python contribution2_prototype_guardrail.py --train defense/MediRed_train.csv \
                                                  --test  defense/MediRed_test.csv

Confirmed against the real repo (Aug 2026): defense/MediRed.csv has columns
Type,Prompt (1000 rows; Type in {Command, Concern Expression, False Pretext,
Format Manipulation, Inquiry, Pressure, Request, Role Play}). MediRed_train.csv
and MediRed_test.csv already exist as a pre-made split, so --text-col/--label-col
default to Prompt/Type and no --split flag is needed for this file.

Note: these 8 "Type" values are the ATTACK-FRAMING taxonomy from the base
paper (Table 10/11), i.e. how the prompt is phrased -- not a PHI-TOPIC
taxonomy (disease/gender/medication/insurance). MediRed.csv does not carry
PHI-topic labels at all, so this script builds prototypes over framing-type,
not PHI-topic. PHI-topic prototypes require MediRed+, which does not exist
yet -- see the Lab-4 report's honesty note on this.

Prompts contain an unfilled {name} placeholder (not yet substituted with a
patient name) -- that's fine for this script, since we're embedding prompt
*structure/framing*, not per-patient content.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd


def load_encoder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install first: pip install sentence-transformers --break-system-packages",
              file=sys.stderr)
        raise
    # Small, CPU-friendly, no GPU required.
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_prototypes(embeddings: np.ndarray, labels: np.ndarray):
    """mu_c = mean embedding of category c (Eq. in Lab-4 report, Contribution 2)."""
    prototypes = {}
    for c in np.unique(labels):
        prototypes[c] = embeddings[labels == c].mean(axis=0)
    return prototypes


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))


def classify(embedding: np.ndarray, prototypes: dict, flag_threshold: float):
    sims = {c: cosine_sim(embedding, mu) for c, mu in prototypes.items()}
    best_cat = max(sims, key=sims.get)
    best_sim = sims[best_cat]
    is_novel = best_sim < flag_threshold  # low max similarity -> "unseen category" flag
    return best_cat, best_sim, is_novel, sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="CSV with labelled MediRed prompts for building prototypes")
    ap.add_argument("--test", required=True, help="CSV with labelled MediRed prompts for evaluation")
    ap.add_argument("--text-col", default="Prompt")
    ap.add_argument("--label-col", default="Type")
    ap.add_argument("--split", type=float, default=None,
                     help="If set, ignore --test and carve this fraction of --train off as test instead")
    ap.add_argument("--flag-threshold", type=float, default=0.35,
                     help="Below this max cosine similarity, flag prompt as 'novel category / send to human review'")
    ap.add_argument("--out", default="contribution2_results.json")
    args = ap.parse_args()

    train_df = pd.read_csv(args.train)
    if args.split is not None:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            train_df, test_size=args.split, stratify=train_df[args.label_col], random_state=42
        )
    else:
        test_df = pd.read_csv(args.test)

    print(f"Train prompts: {len(train_df)} | Test prompts: {len(test_df)}")
    print(f"Categories in train: {sorted(train_df[args.label_col].unique())}")

    model = load_encoder()

    print("Encoding train set...")
    train_emb = model.encode(train_df[args.text_col].astype(str).tolist(), show_progress_bar=True)
    print("Encoding test set...")
    test_emb = model.encode(test_df[args.text_col].astype(str).tolist(), show_progress_bar=True)

    prototypes = build_prototypes(np.asarray(train_emb), train_df[args.label_col].to_numpy())
    print(f"Built {len(prototypes)} category prototypes: {list(prototypes.keys())}")

    correct = 0
    per_category_correct = {}
    per_category_total = {}
    novel_flags = 0
    records = []

    for i, row in test_df.reset_index(drop=True).iterrows():
        true_cat = row[args.label_col]
        pred_cat, best_sim, is_novel, sims = classify(test_emb[i], prototypes, args.flag_threshold)

        per_category_total[true_cat] = per_category_total.get(true_cat, 0) + 1
        if pred_cat == true_cat:
            correct += 1
            per_category_correct[true_cat] = per_category_correct.get(true_cat, 0) + 1
        if is_novel:
            novel_flags += 1

        records.append({
            "true_category": true_cat,
            "predicted_category": pred_cat,
            "max_similarity": round(best_sim, 4),
            "flagged_as_novel": is_novel,
        })

    overall_acc = correct / max(len(test_df), 1)
    per_category_recall = {
        c: round(per_category_correct.get(c, 0) / t, 4)
        for c, t in per_category_total.items()
    }

    results = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "categories": list(prototypes.keys()),
        "overall_prototype_accuracy": round(overall_acc, 4),
        "per_category_recall": per_category_recall,
        "novel_flag_rate": round(novel_flags / max(len(test_df), 1), 4),
        "flag_threshold": args.flag_threshold,
    }

    print("\n=== RESULTS (drop these into the Lab-4 report) ===")
    print(json.dumps(results, indent=2))

    with open(args.out, "w") as f:
        json.dump({"summary": results, "per_prompt": records}, f, indent=2)
    print(f"\nFull per-prompt results written to {args.out}")

    print("""
NEXT STEP (not run here, needs GPU time):
  Fine-tune the same encoder with an InfoNCE contrastive loss
  (anchor = attack prompt, positive = paraphrase/augmented variant,
  negatives = other-category + benign prompts in the batch) before
  computing prototypes. That is the full Contribution 2 design in the
  Lab-4 report; this script gives the off-the-shelf-encoder baseline
  version of it, which is what's honestly achievable before the
  18 Aug deadline on a single RTX 4050 laptop.
""")


if __name__ == "__main__":
    main()
