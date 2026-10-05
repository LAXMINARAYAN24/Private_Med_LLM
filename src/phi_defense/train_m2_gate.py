"""Train M2's suppression gate on top of the frozen M1.

Objective (CLAUDE.md 4.1): make the model REFUSE when a patient identifier is
present, while leaving everything else untouched. Only the gate parameters are
trainable — the base model and M1's LoRA are frozen, so M1 provably still
'knows' the fact; the gate is what withholds it.

    L = L_refuse (on PHI prompts)  +  lambda * L_fidelity (on safe prompts, KL to M1)

Because the gate is a no-op when no identifier is detected, safe/general
queries are unaffected by construction; the fidelity term is a belt-and-braces
check that the gate hasn't learned to fire spuriously.
"""
from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import MODEL, TRAIN, CHECKPOINTS, DATA_DIR
from .models import load_base, load_m1, load_tokenizer, GATE_DIR, PROMPT_TMPL
from .gate import GatedLlama

REFUSAL = "I can't share identified patient health information."
MANIFEST = CHECKPOINTS / "manifest.json"

# a few non-patient clinical prompts for the fidelity term
SAFE_PROMPTS = [
    "What are the symptoms of type 2 diabetes?",
    "What is hypertension?",
    "How is pneumonia typically treated?",
    "What does an ECG measure?",
    "What is the function of insulin?",
]


class RefuseDataset(Dataset):
    def __init__(self, path, tok, cap=2000):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l]
        # Train the gate ONLY on patient-identifier prompts (those have a
        # 'fact_type'). General QA has no identifier -> gate is a no-op -> the
        # loss wouldn't depend on gate params and backward() would fail.
        rows = [r for r in rows if "fact_type" in r]
        self.rows = rows[:cap]
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        q = self.rows[i]["question"]
        prompt = PROMPT_TMPL.format(q=q)
        answer = " " + REFUSAL + self.tok.eos_token
        p = self.tok(prompt, add_special_tokens=True)["input_ids"]
        a = self.tok(answer, add_special_tokens=False)["input_ids"]
        ids = (p + a)[: MODEL.max_seq_len]
        labels = ([-100] * len(p) + a)[: MODEL.max_seq_len]
        return {"input_ids": ids, "labels": labels}


def _collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ii, ll, am = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        ii.append(b["input_ids"] + [pad_id] * pad)
        ll.append(b["labels"] + [-100] * pad)
        am.append([1] * len(b["input_ids"]) + [0] * pad)
    return torch.tensor(ii), torch.tensor(ll), torch.tensor(am)


def _update_manifest(**kw):
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m.update(kw); MANIFEST.write_text(json.dumps(m, indent=2))


def train(epochs=2, lam=0.5, steps_cap=1800):
    tok = load_tokenizer()
    base = load_base()
    m1 = load_m1(base)
    for p in m1.parameters():
        p.requires_grad_(False)

    gated = GatedLlama(m1, tok)
    gated.enabled = True          # gate must be active during its own training
    gated.to(MODEL.device)
    for g in gated.gates:
        for p in g.parameters():
            p.requires_grad_(True)

    ds = RefuseDataset(DATA_DIR / "mimic" / "splits" / "qa_train.jsonl", tok)
    dl = DataLoader(ds, batch_size=TRAIN.batch_size, shuffle=True,
                    collate_fn=lambda b: _collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(gated.gate_parameters(), lr=1e-3)

    step, t0 = 0, time.time()
    for ep in range(epochs):
        for input_ids, labels, attn in dl:
            input_ids, labels, attn = (input_ids.to(MODEL.device),
                                       labels.to(MODEL.device), attn.to(MODEL.device))
            gated.set_context(input_ids)
            out = gated.base(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[:, :-1, :]
            tgt = labels[:, 1:]
            l_refuse = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                ignore_index=-100)

            loss = l_refuse
            if step % 4 == 0:  # occasional fidelity check on a safe prompt
                sp = SAFE_PROMPTS[step % len(SAFE_PROMPTS)]
                enc = tok(PROMPT_TMPL.format(q=sp), return_tensors="pt").to(MODEL.device)
                gated.set_context(enc["input_ids"])          # no digits -> inactive
                lg = gated.base(**enc).logits
                with torch.no_grad():
                    ref = m1(**enc).logits
                l_fid = F.kl_div(F.log_softmax(lg, -1), F.softmax(ref, -1),
                                 reduction="batchmean")
                loss = l_refuse + lam * l_fid

            loss.backward()
            opt.step(); opt.zero_grad()
            if step % 25 == 0:
                print(f"ep{ep} step{step} refuse {l_refuse.item():.3f} "
                      f"({(time.time()-t0)/max(step,1):.2f}s/step)", flush=True)
            step += 1
            if step >= steps_cap:
                break
        if step >= steps_cap:
            break

    GATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"gates": gated.gates.state_dict(),
                "layer_indices": gated.layer_indices,
                "lambda": lam}, GATE_DIR / "gate.pt")
    _update_manifest(contrib_4_1_gate="done", gate_steps=step, gate_lambda=lam)
    print(f"saved gate to {GATE_DIR/'gate.pt'}  (steps {step})")
    return gated


if __name__ == "__main__":
    train()
