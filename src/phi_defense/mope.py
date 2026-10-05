"""Contribution 4.3 — Mixture-of-PHI-Experts (MoPE) Router.

Shared representation h = f_theta(prompt) (reuses the frozen contrastive encoder
from 4.2) -> a trainable router distributes each prompt over K category experts,
whose weighted outputs form a single privacy-risk score.

    pi     = softmax(W_r h + b_r)          in simplex^(K-1)
    y_hat  = sum_k pi_k * Expert_k(h)       (risk in (0,1))
    L_aux  = K * sum_k f_k * P_k            (load balancing; prevents collapse)
    L      = BCE(y_hat, y) + lambda_aux * L_aux

Risk target y: 1 for attack prompts (MediRed), 0 for benign clinical prompts.
K = number of MediRed attack categories. Does NOT touch M1/M2.
"""
from __future__ import annotations

import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .config import (
    CHECKPOINTS, RESULTS, MODEL, SEED, MEDIRED_SPLITS,
    MEDIRED_PLUS_SPLITS, GUARD_DIR_PLUS, MOPE_PLUS_DIR
)
from .guardrail import Encoder, load_medired, GUARD_DIR, load_medired_plus
from .general_qa import build_general_qa

MOPE_DIR = CHECKPOINTS / "mope"


def _cats():
    """Read the category list load_medired() persisted (avoids the stale-import
    binding: load_medired rebinds guardrail.CATS to a new list)."""
    return json.loads((MEDIRED_SPLITS / "categories.json").read_text())


def _cats_plus():
    """Read the 9-class category list from MediRed_plus."""
    cat_file = MEDIRED_PLUS_SPLITS / "categories_plus.json"
    if not cat_file.exists():
        load_medired_plus()
    return json.loads(cat_file.read_text())


class MoPE(nn.Module):
    def __init__(self, dim, k, hidden=128):
        super().__init__()
        self.router = nn.Linear(dim, k)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
             for _ in range(k)])
        self.k = k

    def forward(self, h):
        pi = F.softmax(self.router(h), dim=-1)                  # (B,K)
        ex = torch.cat([e(h) for e in self.experts], dim=-1)    # (B,K) logits
        risk = (pi * torch.sigmoid(ex)).sum(-1)                 # (B,)
        return risk, pi

    def load_balance_loss(self, pi):
        # f_k = fraction hard-routed to k; P_k = mean soft prob to k
        hard = torch.zeros_like(pi).scatter_(1, pi.argmax(1, keepdim=True), 1.0)
        f = hard.mean(0); P = pi.mean(0)
        return self.k * (f * P).sum()


class MoPEClassifier(nn.Module):
    """9-Class Mixture-of-PHI-Experts Classifier.
    
    Each of the K experts specializes in distinguishing framing patterns,
    producing class logits. The router computes a soft mixture over experts,
    and a load-balancing auxiliary loss prevents expert collapse.
    """
    def __init__(self, dim, k=9, num_classes=9, hidden=128):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.router = nn.Linear(dim, k)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes)
            ) for _ in range(k)
        ])

    def forward(self, h):
        pi = F.softmax(self.router(h), dim=-1)                  # (B,K)
        expert_logits = torch.stack([e(h) for e in self.experts], dim=1) # (B,K,C)
        logits = (pi.unsqueeze(-1) * expert_logits).sum(dim=1)  # (B,C)
        return logits, pi

    def load_balance_loss(self, pi):
        hard = torch.zeros_like(pi).scatter_(1, pi.argmax(1, keepdim=True), 1.0)
        f = hard.mean(0); P = pi.mean(0)
        return self.k * (f * P).sum()


def _load_encoder():
    enc = Encoder.__new__(Encoder); nn.Module.__init__(enc)
    enc.tok = AutoTokenizer.from_pretrained(GUARD_DIR)
    enc.model = AutoModel.from_pretrained(GUARD_DIR)
    enc.to(MODEL.device); enc.eval()
    return enc


def _load_encoder_plus():
    enc = Encoder.__new__(Encoder); nn.Module.__init__(enc)
    enc.tok = AutoTokenizer.from_pretrained(GUARD_DIR_PLUS)
    enc.model = AutoModel.from_pretrained(GUARD_DIR_PLUS)
    enc.to(MODEL.device); enc.eval()
    return enc


@torch.no_grad()
def _embed(enc, texts):
    return enc(texts)


def _dataset():
    """(texts, y_risk, cat_idx) for train/val/test. Attacks=1 (MediRed, with
    category), benign=0 (general clinical QA, category=-1). Benign oversampled."""
    train, val, test = load_medired()
    gen = build_general_qa(seed=SEED)
    rng = random.Random(SEED); rng.shuffle(gen)
    n_bt = max(8, int(0.15 * len(gen)))
    ben_test, ben_train = gen[:n_bt], gen[n_bt:]

    def mk(attacks, benign, oversample):
        rows = [(p, 1, c) for p, c in attacks]
        rows += [(r["question"], 0, -1) for r in benign] * oversample
        rng.shuffle(rows)
        return rows
    return (mk(train, ben_train, 6), mk(val, ben_train, 6), mk(test, ben_test, 4))


def _dataset_plus():
    """Return (train, val, test) as (prompt, label_idx) tuples for MediRed_plus.
    Benign is already included as a native class, so no general QA blending is needed."""
    tr, va, te = load_medired_plus()
    rng = random.Random(SEED)
    tr_copy = list(tr); rng.shuffle(tr_copy)
    return tr_copy, list(va), list(te)


def train(epochs=25, lr=1e-3, lam_aux=0.3):
    enc = _load_encoder()
    tr, va, te = _dataset()
    dim = enc.model.config.hidden_size
    Xtr = _embed(enc, [t for t, _, _ in tr])
    ytr = torch.tensor([y for _, y, _ in tr], dtype=torch.float, device=MODEL.device)
    cats = _cats(); mope = MoPE(dim, len(cats)).to(MODEL.device)
    opt = torch.optim.AdamW(mope.parameters(), lr=lr)
    for ep in range(epochs):
        mope.train()
        risk, pi = mope(Xtr)
        loss = F.binary_cross_entropy(risk, ytr) + lam_aux * mope.load_balance_loss(pi)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            print(f"ep{ep} loss {loss.item():.3f}", flush=True)
    MOPE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(mope.state_dict(), MOPE_DIR / "mope.pt")
    print("saved MoPE to", MOPE_DIR)
    return enc, mope, (tr, va, te)


@torch.no_grad()
def evaluate(enc=None, mope=None, bundle=None):
    if enc is None:
        enc = _load_encoder()
        cats = _cats(); mope = MoPE(enc.model.config.hidden_size, len(cats)).to(MODEL.device)
        mope.load_state_dict(torch.load(MOPE_DIR / "mope.pt", map_location=MODEL.device))
        bundle = _dataset()
    mope.eval()
    cat_names = _cats()
    _, _, te = bundle
    X = _embed(enc, [t for t, _, _ in te])
    y = torch.tensor([y for _, y, _ in te], device=MODEL.device)
    row_cats = [c for _, _, c in te]
    risk, pi = mope(X)
    pred = (risk >= 0.5).long()
    acc = (pred == y).float().mean().item()
    # risk detection: attacks should score high
    tp = ((pred == 1) & (y == 1)).sum().item(); fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    # per-category recall (attacks only)
    per_cat = {}
    for ci, cname in enumerate(cat_names):
        idx = [i for i, c in enumerate(row_cats) if c == ci]
        if idx:
            r = (pred[idx] == 1).float().mean().item()
            per_cat[cname] = round(r, 3)
    # expert utilization (hard routing distribution over attack prompts)
    atk_idx = [i for i, c in enumerate(row_cats) if c >= 0]
    util = torch.zeros(len(cat_names))
    routed = pi[atk_idx].argmax(1)
    for r in routed.tolist():
        util[r] += 1
    util = (util / util.sum()).tolist()
    out = {"risk_accuracy": round(acc, 3),
           "attack_precision": round(prec, 3), "attack_recall": round(rec, 3),
           "expert_utilization": [round(u, 3) for u in util],
           "expert_collapse": bool(max(util) > 0.6),
           "per_category_recall": per_cat}
    (RESULTS / "mope.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("risk_accuracy", "attack_precision",
          "attack_recall", "expert_collapse")}, indent=2))
    print("expert utilization:", out["expert_utilization"])
    return out


def train_plus(epochs=25, lr=1e-3, lam_aux=0.3):
    enc = _load_encoder_plus()
    tr, va, te = _dataset_plus()
    dim = enc.model.config.hidden_size
    print(f"Embedding {len(tr)} training prompts with encoder...", flush=True)
    Xtr = _embed(enc, [t for t, _ in tr])
    ytr = torch.tensor([y for _, y in tr], dtype=torch.long, device=MODEL.device)
    cats = _cats_plus()
    C = len(cats)
    mope = MoPEClassifier(dim, k=C, num_classes=C).to(MODEL.device)
    opt = torch.optim.AdamW(mope.parameters(), lr=lr)
    for ep in range(epochs):
        mope.train()
        logits, pi = mope(Xtr)
        ce_loss = F.cross_entropy(logits, ytr)
        aux_loss = lam_aux * mope.load_balance_loss(pi)
        loss = ce_loss + aux_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"ep{ep:02d} loss {loss.item():.4f} (ce={ce_loss.item():.4f}, aux={aux_loss.item():.4f})", flush=True)
    MOPE_PLUS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(mope.state_dict(), MOPE_PLUS_DIR / "mope_plus.pt")
    print("saved MoPEClassifier to", MOPE_PLUS_DIR)
    return enc, mope, (tr, va, te)


@torch.no_grad()
def evaluate_plus(enc=None, mope=None, bundle=None):
    if enc is None:
        enc = _load_encoder_plus()
    cats = _cats_plus()
    C = len(cats)
    if mope is None:
        mope = MoPEClassifier(enc.model.config.hidden_size, k=C, num_classes=C).to(MODEL.device)
        mope.load_state_dict(torch.load(MOPE_PLUS_DIR / "mope_plus.pt", map_location=MODEL.device))
    if bundle is None:
        bundle = _dataset_plus()
    mope.eval()
    _, _, te = bundle
    X = _embed(enc, [t for t, _ in te])
    y = torch.tensor([y for _, y in te], device=MODEL.device)
    logits, pi = mope(X)
    pred = logits.argmax(dim=-1)

    gold = y.cpu().tolist()
    pred_list = pred.cpu().tolist()
    tp = [0] * C; fp = [0] * C; fn = [0] * C; correct = 0
    cm = [[0] * C for _ in range(C)]
    for g, p in zip(gold, pred_list):
        cm[g][p] += 1
        if g == p:
            correct += 1; tp[g] += 1
        else:
            fp[p] += 1; fn[g] += 1

    per_cat = {}
    for c in range(C):
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_cat[cats[c]] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "n": tp[c] + fn[c]
        }
    acc = round(correct / len(gold), 3) if gold else 0.0
    macro_f1 = round(sum(v["f1"] for v in per_cat.values()) / C, 3) if C else 0.0

    # Expert utilization over test prompts
    util = torch.zeros(C)
    routed = pi.argmax(1)
    for r in routed.tolist():
        util[r] += 1
    util_list = [round(u, 3) for u in (util / util.sum()).tolist()]
    collapse = bool(max(util_list) > 0.6)

    out = {
        "dataset": "MediRed_plus",
        "classes": cats,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "random_baseline": round(1 / C, 3),
        "n_test": len(gold),
        "expert_utilization": util_list,
        "expert_collapse": collapse,
        "per_category": per_cat,
        "confusion_matrix": cm,
    }
    (RESULTS / "mope_plus.json").write_text(json.dumps(out, indent=2))
    print("\n=================================================================")
    print(" MediRed_plus 9-Class MoPE Classifier Results")
    print("=================================================================")
    print(f"Accuracy: {acc:.3f} | Macro F1: {macro_f1:.3f} | Random Baseline: {out['random_baseline']:.3f}")
    print(f"Expert Utilization: {util_list} | Expert Collapse: {collapse}")
    print("\nPer-Category Metrics:")
    for cat_name, metrics in per_cat.items():
        print(f"  {cat_name:22s} P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  (N={metrics['n']})")
    print("\nConfusion Matrix (Rows: True, Cols: Pred):")
    header = " " * 22 + " ".join(f"{c[:5]:>5s}" for c in cats)
    print(header)
    for g_idx, cname in enumerate(cats):
        row_str = " ".join(f"{cm[g_idx][p_idx]:5d}" for p_idx in range(C))
        print(f"{cname:22s} {row_str}")
    return out


@torch.no_grad()
def mope_attack_prob(prompt: str, enc=None, mope=None):
    """Compute attack probability and class prediction for a single prompt using MoPEClassifier."""
    cats = _cats_plus()
    if enc is None:
        enc = _load_encoder_plus()
    if mope is None:
        mope = MoPEClassifier(enc.model.config.hidden_size, k=len(cats), num_classes=len(cats)).to(MODEL.device)
        mope.load_state_dict(torch.load(MOPE_PLUS_DIR / "mope_plus.pt", map_location=MODEL.device))
        mope.eval()
    X = _embed(enc, [prompt])
    logits, pi = mope(X)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_idx = probs.argmax().item()
    pred_class = cats[pred_idx]
    benign_idx = cats.index("Benign") if "Benign" in cats else -1
    attack_prob = round(1.0 - probs[benign_idx].item(), 4) if benign_idx >= 0 else (0.0 if pred_class == "Benign" else 1.0)
    routed_expert = pi.argmax(dim=-1).item()
    return {
        "prompt": prompt,
        "class": pred_class,
        "class_index": pred_idx,
        "confidence": round(probs[pred_idx].item(), 4),
        "attack_prob": max(0.0, min(1.0, attack_prob)),
        "is_attack": pred_class != "Benign",
        "routed_expert": routed_expert,
        "class_probs": {cats[i]: round(probs[i].item(), 4) for i in range(len(cats))},
    }


@torch.no_grad()
def evaluate_mope_as_defense(max_patients=40, risk_thresh=0.5, use_plus=False):
    """M4: MoPE risk score gating M1 on the PHI attack suite (generation mode)."""
    import pathlib
    from .models import load_m1, load_tokenizer, generate
    from . import attacks as A
    facts = json.loads(pathlib.Path("data/mimic/patient_facts.json").read_text())
    facts = dict(list(facts.items())[:max_patients])
    suite = [a for a in A.build_attacks(facts) if a.mode == "generation"]

    if use_plus:
        enc = _load_encoder_plus()
        cats = _cats_plus()
        mope = MoPEClassifier(enc.model.config.hidden_size, k=len(cats), num_classes=len(cats)).to(MODEL.device)
        mope.load_state_dict(torch.load(MOPE_PLUS_DIR / "mope_plus.pt", map_location=MODEL.device))
        mope.eval()
        def risk(p):
            return mope_attack_prob(p, enc=enc, mope=mope)["attack_prob"]
        res_file = RESULTS / "mope_defense_M4_plus.json"
    else:
        enc = _load_encoder()
        cats = _cats()
        mope = MoPE(enc.model.config.hidden_size, len(cats)).to(MODEL.device)
        mope.load_state_dict(torch.load(MOPE_DIR / "mope.pt", map_location=MODEL.device))
        mope.eval()
        def risk(p):
            r, _ = mope(_embed(enc, [p])); return r.item()
        res_file = RESULTS / "mope_defense_M4.json"

    tok = load_tokenizer(); m1 = load_m1()

    blocked = leaks = 0
    for atk in suite:
        if risk(atk.prompt) >= risk_thresh:
            blocked += 1
        elif A.leaked(atk, generate(m1, tok, atk.prompt)):
            leaks += 1
    benign = [r["question"] for r in build_general_qa()]
    false_block = sum(risk(b) >= risk_thresh for b in benign) / len(benign)
    out = {"risk_thresh": risk_thresh, "n_attacks": len(suite),
           "attack_block_rate": round(blocked / len(suite), 3),
           "benign_false_block_rate": round(false_block, 3),
           "M4_leak_rate_generation": round(leaks / len(suite), 3)}
    res_file.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import sys
    if "--plus" in sys.argv:
        enc, mope, bundle = train_plus()
        evaluate_plus(enc, mope, bundle)
    else:
        enc, mope, bundle = train()
        evaluate(enc, mope, bundle)
