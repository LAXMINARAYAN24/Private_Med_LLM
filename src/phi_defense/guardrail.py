"""Contribution 4.2 — Contrastive Dual-Encoder + Category Prototypes (guardrail).

Trains a transformer embedding f_theta on MediRed attack prompts with a
supervised-contrastive objective, builds one prototype per attack-framing
category (mean embedding), and classifies a prompt by nearest prototype (cosine).
Low max-similarity flags a novel/unseen attack.

    L_contrastive = -log[ exp(sim(z_a,z_p)/tau) / sum_k exp(sim(z_a,z_k)/tau) ]
    mu_c          = mean of f_theta(x) over training prompts of category c
    category(x)   = argmax_c cos(f_theta(x), mu_c)
    novel if max_c cos(f_theta(x), mu_c) < tau_flag

Base encoder: sentence-transformers/all-MiniLM-L6-v2 (ungated, ~22M params),
used through plain transformers with mean pooling. Does NOT touch M1/M2.
"""
from __future__ import annotations

import csv
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .config import (
    MEDIRED_RAW, MEDIRED_SPLITS, MEDIRED_PLUS_RAW, MEDIRED_PLUS_SPLITS,
    CHECKPOINTS, GUARD_DIR_PLUS, RESULTS, MODEL, SEED
)

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GUARD_DIR = CHECKPOINTS / "guardrail_encoder"
CATS: list[str] = []        # filled by load_medired (sorted, deterministic)
CATS_PLUS: list[str] = []   # filled by load_medired_plus (sorted, deterministic)


def _read(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="replace")))
    tkey = [k for k in rows[0] if k.strip().lstrip("﻿").lower() == "type"][0]
    pkey = [k for k in rows[0] if k.strip().lstrip("﻿").lower() == "prompt"][0]
    return [(r[pkey].strip(), r[tkey].strip()) for r in rows]


def load_medired_plus():
    """Return (train, val, test) as lists of (prompt, label_idx) for MediRed_plus.
    9 classes. Val is carved stratified from MediRed_plus_train.csv (12.5%);
    test is the untouched MediRed_plus_test.csv."""
    global CATS_PLUS
    train_raw = _read(MEDIRED_PLUS_RAW / "MediRed_plus_train.csv")
    test_raw = _read(MEDIRED_PLUS_RAW / "MediRed_plus_test.csv")
    CATS_PLUS = sorted({c for _, c in train_raw})
    idx = {c: i for i, c in enumerate(CATS_PLUS)}

    by_cat = {}
    for p, c in train_raw:
        by_cat.setdefault(c, []).append(p)
    rng = random.Random(SEED)
    train, val = [], []
    for c, prompts in by_cat.items():
        rng.shuffle(prompts)
        n_val = max(1, int(0.125 * len(prompts)))
        val += [(p, idx[c]) for p in prompts[:n_val]]
        train += [(p, idx[c]) for p in prompts[n_val:]]
    test = [(p, idx[c]) for p, c in test_raw]
    rng.shuffle(train)
    MEDIRED_PLUS_SPLITS.mkdir(parents=True, exist_ok=True)
    (MEDIRED_PLUS_SPLITS / "categories_plus.json").write_text(json.dumps(CATS_PLUS, indent=2))
    return train, val, test


def load_medired():
    """Return (train, val, test) as lists of (prompt, label_idx). Val is carved
    stratified from the original train split; test is the untouched MediRed_test."""
    global CATS
    train_raw = _read(MEDIRED_RAW / "MediRed_train.csv")
    test_raw = _read(MEDIRED_RAW / "MediRed_test.csv")
    CATS = sorted({c for _, c in train_raw})
    idx = {c: i for i, c in enumerate(CATS)}

    by_cat = {}
    for p, c in train_raw:
        by_cat.setdefault(c, []).append(p)
    rng = random.Random(SEED)
    train, val = [], []
    for c, prompts in by_cat.items():
        rng.shuffle(prompts)
        n_val = max(1, int(0.125 * len(prompts)))   # ~100 of 800
        val += [(p, idx[c]) for p in prompts[:n_val]]
        train += [(p, idx[c]) for p in prompts[n_val:]]
    test = [(p, idx[c]) for p, c in test_raw]
    rng.shuffle(train)
    MEDIRED_SPLITS.mkdir(parents=True, exist_ok=True)
    (MEDIRED_SPLITS / "categories.json").write_text(json.dumps(CATS, indent=2))
    return train, val, test


class Encoder(nn.Module):
    def __init__(self, name=ENCODER_NAME):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)

    def forward(self, texts, batch=32, device=None):
        device = device or MODEL.device
        embs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=64,
                           return_tensors="pt").to(device)
            out = self.model(**enc).last_hidden_state           # (B,T,H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embs.append(F.normalize(pooled, dim=-1))
        return torch.cat(embs, 0)


def supcon_loss(z, labels, tau=0.07):
    """Supervised contrastive loss over a batch of L2-normalized embeddings."""
    sim = z @ z.t() / tau                                       # (B,B)
    B = z.size(0)
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, -1e9)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & ~self_mask
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = pos_mask.sum(1)
    valid = pos_counts > 0
    loss = -(logp * pos_mask).sum(1)[valid] / pos_counts[valid]
    return loss.mean()


def train_encoder(epochs=6, bs=64, lr=2e-5):
    train, val, test = load_medired()
    enc = Encoder().to(MODEL.device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr)
    texts = [p for p, _ in train]
    labels = torch.tensor([y for _, y in train], device=MODEL.device)
    rng = random.Random(SEED)
    order = list(range(len(train)))
    for ep in range(epochs):
        rng.shuffle(order)
        tot = 0.0
        for i in range(0, len(order), bs):
            batch_idx = order[i:i + bs]
            z = enc([texts[j] for j in batch_idx])
            loss = supcon_loss(z, labels[batch_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"ep{ep} contrastive_loss {tot/(len(order)//bs+1):.3f}", flush=True)
    GUARD_DIR.mkdir(parents=True, exist_ok=True)
    enc.model.save_pretrained(GUARD_DIR); enc.tok.save_pretrained(GUARD_DIR)
    print("saved encoder to", GUARD_DIR)
    return enc, (train, val, test)


@torch.no_grad()
def compute_prototypes(enc, train, num_classes=None):
    n_c = num_classes if num_classes is not None else len(CATS)
    embs = enc([p for p, _ in train])
    labels = torch.tensor([y for _, y in train], device=embs.device)
    protos = []
    for c in range(n_c):
        m = labels == c
        if m.sum() == 0:
            protos.append(torch.zeros(embs.shape[-1], device=embs.device))
        else:
            protos.append(F.normalize(embs[m].mean(0), dim=-1))
    return torch.stack(protos)                                  # (C,H)


@torch.no_grad()
def classify(enc, protos, texts):
    z = enc(texts)
    sims = z @ protos.t()                                       # (N,C)
    conf, pred = sims.max(1)
    return pred.cpu().tolist(), conf.cpu().tolist()


def evaluate(enc=None, bundle=None, tau_flag=0.0):
    if enc is None:
        enc = Encoder.__new__(Encoder); nn.Module.__init__(enc)
        enc.tok = AutoTokenizer.from_pretrained(GUARD_DIR)
        enc.model = AutoModel.from_pretrained(GUARD_DIR)
        enc.to(MODEL.device)
        bundle = load_medired()
    train, val, test = bundle
    enc.eval()
    protos = compute_prototypes(enc, train)

    # tune tau_flag on val (max sim of correctly-in-distribution prompts): pick
    # a threshold that flags <=5% of known-category val prompts as novel.
    _, vconf = classify(enc, protos, [p for p, _ in val])
    vconf_sorted = sorted(vconf)
    tau_flag = round(vconf_sorted[max(0, int(0.05 * len(vconf_sorted)) - 1)], 3)

    pred, conf = classify(enc, protos, [p for p, _ in test])
    gold = [y for _, y in test]
    C = len(CATS)
    tp = [0] * C; fp = [0] * C; fn = [0] * C; correct = 0
    for g, p in zip(gold, pred):
        if g == p:
            correct += 1; tp[g] += 1
        else:
            fp[p] += 1; fn[g] += 1
    per_cat = {}
    for c in range(C):
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_cat[CATS[c]] = {"precision": round(prec, 3), "recall": round(rec, 3),
                            "f1": round(f1, 3), "n": tp[c] + fn[c]}
    novel_rate = sum(1 for c in conf if c < tau_flag) / len(conf)
    out = {"accuracy": round(correct / len(gold), 3),
           "macro_f1": round(sum(v["f1"] for v in per_cat.values()) / C, 3),
           "random_baseline": round(1 / C, 3),
           "tau_flag": tau_flag, "novel_flag_rate_on_known": round(novel_rate, 3),
           "n_test": len(gold), "per_category": per_cat}
    (RESULTS / "guardrail_contrastive.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("accuracy", "macro_f1", "random_baseline",
                                          "tau_flag")}, indent=2))
    return out


def train_encoder_plus(epochs=8, bs=64, lr=2e-5):
    train, val, test = load_medired_plus()
    enc = Encoder().to(MODEL.device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr)
    texts = [p for p, _ in train]
    labels = torch.tensor([y for _, y in train], device=MODEL.device)
    rng = random.Random(SEED)
    order = list(range(len(train)))
    for ep in range(epochs):
        rng.shuffle(order)
        tot = 0.0
        for i in range(0, len(order), bs):
            batch_idx = order[i:i + bs]
            z = enc([texts[j] for j in batch_idx])
            loss = supcon_loss(z, labels[batch_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"ep{ep} contrastive_loss {tot/(len(order)//bs+1):.3f}", flush=True)
    GUARD_DIR_PLUS.mkdir(parents=True, exist_ok=True)
    enc.model.save_pretrained(GUARD_DIR_PLUS); enc.tok.save_pretrained(GUARD_DIR_PLUS)
    protos = compute_prototypes(enc, train, num_classes=len(CATS_PLUS))
    torch.save(protos.cpu(), GUARD_DIR_PLUS / "prototypes.pt")
    print("saved encoder and prototypes to", GUARD_DIR_PLUS)
    return enc, (train, val, test)


def evaluate_plus(enc=None, bundle=None, tau_flag=0.0):
    global CATS_PLUS
    if not CATS_PLUS:
        cat_file = MEDIRED_PLUS_SPLITS / "categories_plus.json"
        if cat_file.exists():
            CATS_PLUS = json.loads(cat_file.read_text())
        else:
            load_medired_plus()
    if bundle is None:
        bundle = load_medired_plus()
    train, val, test = bundle
    if enc is None:
        enc = Encoder.__new__(Encoder); nn.Module.__init__(enc)
        enc.tok = AutoTokenizer.from_pretrained(GUARD_DIR_PLUS)
        enc.model = AutoModel.from_pretrained(GUARD_DIR_PLUS)
        enc.to(MODEL.device)
    enc.eval()
    C = len(CATS_PLUS)
    proto_path = GUARD_DIR_PLUS / "prototypes.pt"
    if proto_path.exists():
        protos = torch.load(proto_path, map_location=MODEL.device)
    else:
        protos = compute_prototypes(enc, train, num_classes=C)

    # tune tau_flag on val (max sim of correctly-in-distribution prompts): pick
    # a threshold that flags <=5% of known-category val prompts as novel.
    _, vconf = classify(enc, protos, [p for p, _ in val])
    vconf_sorted = sorted(vconf)
    tau_flag = round(vconf_sorted[max(0, int(0.05 * len(vconf_sorted)) - 1)], 3) if vconf_sorted else 0.0

    pred, conf = classify(enc, protos, [p for p, _ in test])
    gold = [y for _, y in test]
    tp = [0] * C; fp = [0] * C; fn = [0] * C; correct = 0
    cm = [[0] * C for _ in range(C)]
    for g, p in zip(gold, pred):
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
        per_cat[CATS_PLUS[c]] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "n": tp[c] + fn[c]
        }
    novel_rate = sum(1 for c in conf if c < tau_flag) / len(conf) if conf else 0.0
    acc = round(correct / len(gold), 3) if gold else 0.0
    macro_f1 = round(sum(v["f1"] for v in per_cat.values()) / C, 3) if C else 0.0
    out = {
        "dataset": "MediRed_plus",
        "classes": CATS_PLUS,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "random_baseline": round(1 / C, 3) if C else 0.0,
        "tau_flag": tau_flag,
        "novel_flag_rate_on_known": round(novel_rate, 3),
        "n_test": len(gold),
        "per_category": per_cat,
        "confusion_matrix": cm,
    }
    (RESULTS / "guardrail_contrastive_plus.json").write_text(json.dumps(out, indent=2))
    print("\n=================================================================")
    print(" MediRed_plus 9-Class Guardrail Contrastive Results")
    print("=================================================================")
    print(f"Accuracy: {acc:.3f} | Macro F1: {macro_f1:.3f} | Random Baseline: {out['random_baseline']:.3f} | Tau Flag: {tau_flag}")
    print("\nPer-Category Metrics:")
    for cat_name, metrics in per_cat.items():
        print(f"  {cat_name:22s} P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  (N={metrics['n']})")
    print("\nConfusion Matrix (Rows: True, Cols: Pred):")
    header = " " * 22 + " ".join(f"{c[:5]:>5s}" for c in CATS_PLUS)
    print(header)
    for g_idx, cname in enumerate(CATS_PLUS):
        row_str = " ".join(f"{cm[g_idx][p_idx]:5d}" for p_idx in range(C))
        print(f"{cname:22s} {row_str}")
    return out


@torch.no_grad()
def predict_plus(prompt: str, enc=None, protos=None):
    """Predict the MediRed_plus 9-class framing and confidence for an arbitrary prompt."""
    global CATS_PLUS
    if not CATS_PLUS:
        cat_file = MEDIRED_PLUS_SPLITS / "categories_plus.json"
        if cat_file.exists():
            CATS_PLUS = json.loads(cat_file.read_text())
        else:
            load_medired_plus()
    if enc is None:
        enc = Encoder.__new__(Encoder); nn.Module.__init__(enc)
        enc.tok = AutoTokenizer.from_pretrained(GUARD_DIR_PLUS)
        enc.model = AutoModel.from_pretrained(GUARD_DIR_PLUS)
        enc.to(MODEL.device); enc.eval()
    if protos is None:
        proto_path = GUARD_DIR_PLUS / "prototypes.pt"
        if proto_path.exists():
            protos = torch.load(proto_path, map_location=MODEL.device)
        else:
            train, _, _ = load_medired_plus()
            protos = compute_prototypes(enc, train, num_classes=len(CATS_PLUS))
    z = enc([prompt])
    sims = (z @ protos.t()).squeeze(0)  # (C,)
    conf, pred_idx = sims.max(0)
    pred_idx = pred_idx.item()
    pred_class = CATS_PLUS[pred_idx]
    benign_idx = CATS_PLUS.index("Benign") if "Benign" in CATS_PLUS else -1
    attack_prob = round(1.0 - sims[benign_idx].item(), 3) if benign_idx >= 0 else (0.0 if pred_class == "Benign" else 1.0)
    sims_dict = {CATS_PLUS[i]: round(sims[i].item(), 4) for i in range(len(CATS_PLUS))}
    return {
        "prompt": prompt,
        "class": pred_class,
        "class_index": pred_idx,
        "confidence": round(conf.item(), 4),
        "attack_prob": max(0.0, min(1.0, attack_prob)),
        "is_attack": pred_class != "Benign",
        "similarities": sims_dict,
    }


class GuardrailFilter:
    """Block/allow decision in front of the target LLM (M3). Blocks a prompt when
    its max cosine similarity to any attack-category prototype >= tau."""

    def __init__(self, tau: float | None = None, false_block: float = 0.10, use_plus: bool = False):
        import torch.nn as _nn
        self.use_plus = use_plus
        ckpt_dir = GUARD_DIR_PLUS if use_plus else GUARD_DIR
        self.enc = Encoder.__new__(Encoder); _nn.Module.__init__(self.enc)
        self.enc.tok = AutoTokenizer.from_pretrained(ckpt_dir)
        self.enc.model = AutoModel.from_pretrained(ckpt_dir)
        self.enc.to(MODEL.device); self.enc.eval()
        if use_plus:
            train, _, _ = load_medired_plus()
            self.protos = compute_prototypes(self.enc, train, num_classes=len(CATS_PLUS))
            self.benign_idx = CATS_PLUS.index("Benign") if "Benign" in CATS_PLUS else -1
        else:
            train, _, _ = load_medired()
            self.protos = compute_prototypes(self.enc, train, num_classes=len(CATS))
            self.benign_idx = -1
        self.tau = tau if tau is not None else self._calibrate(false_block)

    @torch.no_grad()
    def _max_sim(self, texts):
        z = self.enc(texts)
        if self.use_plus and self.benign_idx >= 0:
            # Mask out benign prototype so we only measure similarity to attack categories
            atk_protos = torch.stack([self.protos[i] for i in range(len(self.protos)) if i != self.benign_idx])
            return (z @ atk_protos.t()).max(1).values.cpu().tolist()
        return (z @ self.protos.t()).max(1).values.cpu().tolist()

    def _calibrate(self, false_block=0.10):
        """Pick tau so ~false_block of benign clinical prompts are blocked."""
        from .general_qa import build_general_qa
        benign = [r["question"] for r in build_general_qa()]
        sims = sorted(self._max_sim(benign))
        return round(sims[min(len(sims) - 1, int((1 - false_block) * len(sims)))], 3)

    def blocks(self, prompt: str) -> bool:
        return self._max_sim([prompt])[0] >= self.tau


def evaluate_guardrail_as_defense(max_patients=40):
    """M3: guardrail gating M1 on the PHI attack suite (generation mode).
    Reports attack block rate, benign false-block rate, and residual leak."""
    import json as _json
    from .models import load_m1, load_tokenizer, generate
    from .general_qa import build_general_qa
    from . import attacks as A
    facts = _json.loads((__import__("pathlib").Path("data/mimic/patient_facts.json")).read_text())
    facts = dict(list(facts.items())[:max_patients])
    suite = [a for a in A.build_attacks(facts) if a.mode == "generation"]

    gf = GuardrailFilter()
    tok = load_tokenizer(); m1 = load_m1()

    blocked = leaks = 0
    for atk in suite:
        if gf.blocks(atk.prompt):
            blocked += 1                       # blocked -> refusal -> no leak
        else:
            if A.leaked(atk, generate(m1, tok, atk.prompt)):
                leaks += 1
    benign = [r["question"] for r in build_general_qa()]
    false_block = sum(gf.blocks(b) for b in benign) / len(benign)

    out = {"tau": gf.tau, "n_attacks": len(suite),
           "attack_block_rate": round(blocked / len(suite), 3),
           "benign_false_block_rate": round(false_block, 3),
           "M3_leak_rate_generation": round(leaks / len(suite), 3)}
    (RESULTS / "guardrail_defense_M3.json").write_text(_json.dumps(out, indent=2))
    print(_json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import sys
    if "--plus" in sys.argv:
        enc, bundle = train_encoder_plus()
        evaluate_plus(enc, bundle)
    else:
        enc, bundle = train_encoder()
        evaluate(enc, bundle)

