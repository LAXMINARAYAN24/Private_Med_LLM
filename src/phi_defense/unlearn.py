"""Contribution 4.5 — Selective unlearning (training-time, model-level defense).

Produces M5 = m1_unlearn: a variant of M1 whose patient-fact associations are
scrubbed from the weights, so it no longer RECALLS them even with no prompt-level
defense (unlike the gate, which knows-but-withholds). Chosen over DP-SGD because
per-sample-gradient DP (Opacus) is fragile/slow on MPS; documented per CLAUDE.md 8.

Objective (gradient ascent on the forget set + retain regularization):
    L = beta * CE_retain(general QA)  -  alpha * min(CE_forget(patient facts), margin)

The capped forget term pushes weights away from emitting the true patient fact
without diverging; the retain term keeps general clinical competence. Sweeping
alpha traces a privacy-strength curve (the DP-epsilon analog). M1 is NOT modified
- we copy its adapter into checkpoints/m1_unlearn_lora and train that.
"""
from __future__ import annotations

import json
import random
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import MODEL, CHECKPOINTS, DATA_DIR, TRAIN
from .models import load_base, load_tokenizer, M1_DIR, PROMPT_TMPL
from .train_m1 import QADataset, _collate

UNLEARN_DIR = CHECKPOINTS / "m1_unlearn_lora"   # legacy default
M5_1_DIR = CHECKPOINTS / "m5_1_unlearn"          # aggressive (alpha=1.0)
M5_2_DIR = CHECKPOINTS / "m5_2_unlearn"          # gentle (coherent forgetting)


def _load_m1_trainable():
    """Load base + M1 adapter with the adapter params trainable (for unlearning)."""
    from peft import PeftModel
    base = load_base()
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(M1_DIR), is_trainable=True)
    model.to(MODEL.device)
    return model


def _split_rows(tok):
    rows = [json.loads(l) for l in
            (DATA_DIR / "mimic" / "splits" / "qa_train.jsonl").read_text().splitlines() if l]
    forget = [r for r in rows if "fact_type" in r]        # patient facts
    retain = [r for r in rows if r.get("kind") == "general"]
    return forget, retain


class _Rows(QADataset):
    def __init__(self, rows, tok, max_len=MODEL.max_seq_len):
        self.rows = rows; self.tok = tok; self.max_len = max_len


def unlearn(alpha=1.0, beta=1.0, margin=4.0, steps=400, lr=1e-4, cap=1500,
            out_dir=UNLEARN_DIR):
    tok = load_tokenizer()
    forget, retain = _split_rows(tok)
    rng = random.Random(TRAIN.seed); rng.shuffle(forget); rng.shuffle(retain)
    forget, retain = forget[:cap], retain[:cap]
    f_dl = DataLoader(_Rows(forget, tok), batch_size=TRAIN.batch_size, shuffle=True,
                      collate_fn=lambda b: _collate(b, tok.pad_token_id))
    r_dl = DataLoader(_Rows(retain, tok), batch_size=TRAIN.batch_size, shuffle=True,
                      collate_fn=lambda b: _collate(b, tok.pad_token_id))

    model = _load_m1_trainable(); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    f_it, r_it = iter(f_dl), iter(r_dl)

    def _ce(batch):
        ii, ll, am = (t.to(MODEL.device) for t in batch)
        return model(input_ids=ii, attention_mask=am, labels=ll).loss

    for step in range(steps):
        try: fb = next(f_it)
        except StopIteration: f_it = iter(f_dl); fb = next(f_it)
        try: rb = next(r_it)
        except StopIteration: r_it = iter(r_dl); rb = next(r_it)
        ce_forget = _ce(fb)
        ce_retain = _ce(rb)
        # ascent on forget (capped), descent on retain
        loss = beta * ce_retain - alpha * torch.clamp(ce_forget, max=margin)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 25 == 0:
            print(f"step{step} forget_CE {ce_forget.item():.3f} retain_CE {ce_retain.item():.3f}",
                  flush=True)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    model.save_pretrained(str(out_dir))
    manifest = CHECKPOINTS / "manifest.json"
    m = json.loads(manifest.read_text()) if manifest.exists() else {}
    m.update({"contrib_4_5_unlearn": "done", f"unlearn_alpha[{out_dir.name}]": alpha,
              f"unlearn_steps[{out_dir.name}]": steps})
    manifest.write_text(json.dumps(m, indent=2))
    print(f"saved unlearned adapter to {out_dir} (alpha={alpha}, steps={steps})")


def load_m5(base=None, adapter_dir=M5_2_DIR):
    from peft import PeftModel
    base = base or load_base()
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to(MODEL.device); model.eval()
    return model


if __name__ == "__main__":
    unlearn()
