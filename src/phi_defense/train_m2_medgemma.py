"""Train M2's suppression gate on top of frozen MedGemma M1.

Objective:
When a patient identifier is detected in context, suppress the patient representation
in the decoder layers so the model withholds PHI and outputs the refusal:
    "I can't share identified patient health information."
When no patient identifier is detected, the gate is a no-op (guarded by KL fidelity
loss against M1 on safe clinical queries).

Only the gate parameters are trainable. MedGemma-4B and M1's LoRA adapter are completely
frozen, proving that M1 still retains its medical knowledge and the gate is what withholds PHI.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import (
    MODEL, TRAIN, CHECKPOINTS, DATA_DIR,
    MEDGEMMA_BASE_MODEL, M1_MEDGEMMA_DIR, M2_MEDGEMMA_DIR,
    MIMIC_SPLITS_1800, FACTS_1800_PATH
)
from .models import (
    load_medgemma_tokenizer, load_medgemma_m1, PROMPT_TMPL
)
from .gate import GatedMedGemma

REFUSAL = "I can't share identified patient health information."
MANIFEST = CHECKPOINTS / "manifest.json"

SAFE_PROMPTS = [
    "What are the symptoms of type 2 diabetes?",
    "What is hypertension?",
    "How is pneumonia typically treated?",
    "What does an ECG measure?",
    "What is the function of insulin?",
    "How is acute myocardial infarction diagnosed?",
    "What are the clinical signs of sepsis?",
    "Describe the pathophysiology of asthma.",
]


class RefuseDataset(Dataset):
    """Dataset of patient-identifier queries paired with the standard refusal."""

    def __init__(self, path: Path, tok, cap: int = 2000, max_len: int = MODEL.max_seq_len):
        if not path.exists():
            raise FileNotFoundError(f"QA training data not found at {path}")
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        # Train the gate strictly on patient-identifier prompts (those have a 'fact_type')
        self.rows = [r for r in rows if "fact_type" in r][:cap]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        q = self.rows[i]["question"]
        prompt = PROMPT_TMPL.format(q=q)
        answer = " " + REFUSAL + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[: self.max_len]
        # Mask prompt tokens with -100 so only the refusal tokens contribute to loss
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


def _collate(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn, token_type_ids = [], [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
        token_type_ids.append([0] * maxlen)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attn),
        torch.tensor(token_type_ids),
    )


def _update_manifest(**kw):
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m.update(kw)
    MANIFEST.write_text(json.dumps(m, indent=2))


def train(
    epochs: int = 2,
    lam: float = 0.5,
    lr: float = 1e-3,
    batch_size: int = 4,
    steps_cap: int = 600,
    load_in_4bit: bool = True,
    resume: bool = True,
):
    print("=" * 65, flush=True)
    print(" Training MedGemma M2 (PHI Suppression Gate)", flush=True)
    print("=" * 65, flush=True)

    tok = load_medgemma_tokenizer()
    print("Loading frozen MedGemma M1...", flush=True)
    m1 = load_medgemma_m1(load_in_4bit=load_in_4bit)
    for p in m1.parameters():
        p.requires_grad_(False)

    print("Instantiating GatedMedGemma...", flush=True)
    gated = GatedMedGemma(m1, tok)
    gated.enabled = True
    device = next(m1.parameters()).device
    dtype = getattr(m1, "dtype", None) or next(m1.parameters()).dtype
    if dtype in (torch.float16, torch.bfloat16):
        gated.gates.to(device=device, dtype=dtype)
    else:
        gated.gates.to(device=device)

    ckpt_path = M2_MEDGEMMA_DIR / "gate.pt"
    if resume and ckpt_path.exists():
        print(f"Resuming gate weights from {ckpt_path}...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        gated.gates.load_state_dict(ckpt["gates"])
        if dtype in (torch.float16, torch.bfloat16):
            gated.gates.to(device=device, dtype=dtype)
        else:
            gated.gates.to(device=device)

    for g in gated.gates:
        for p in g.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in gated.gate_parameters() if p.requires_grad)
    total = sum(p.numel() for p in m1.parameters())
    print(f"Gate layers:         {len(gated.layer_indices)} (indices: {gated.layer_indices[0]}..{gated.layer_indices[-1]})", flush=True)
    print(f"Trainable Gate Params: {trainable:,} || Total Model Params: {total:,} ({trainable/total*100:.3f}%)", flush=True)
    print(f"Compute Device:      {device}", flush=True)
    print(f"Fidelity Lambda:     {lam}", flush=True)

    train_path = (MIMIC_SPLITS_1800 / "qa_train.jsonl") if (MIMIC_SPLITS_1800 / "qa_train.jsonl").exists() else (DATA_DIR / "mimic" / "splits" / "qa_train.jsonl")
    ds = RefuseDataset(train_path, tok, cap=2000)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tok.pad_token_id),
    )
    opt = torch.optim.AdamW(gated.gate_parameters(), lr=lr)

    M2_MEDGEMMA_DIR.mkdir(parents=True, exist_ok=True)
    step, t0 = 0, time.time()
    total_loss = 0.0

    _update_manifest(
        m2_medgemma_gate="in_progress",
        m2_medgemma_base=MEDGEMMA_BASE_MODEL,
        m2_medgemma_m1=str(M1_MEDGEMMA_DIR),
    )

    print(f"\nStarting Gate Training ({epochs} epochs, capped at {steps_cap} steps)...", flush=True)
    for ep in range(epochs):
        ep_t0 = time.time()
        for input_ids, labels, attn, token_type_ids in dl:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = attn.to(device)
            token_type_ids = token_type_ids.to(device)

            gated.enabled = True
            gated.set_context(input_ids)
            out = gated.base(
                input_ids=input_ids,
                attention_mask=attn,
                token_type_ids=token_type_ids,
            )
            logits = out.logits[:, :-1, :]
            tgt = labels[:, 1:]
            l_refuse = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=-100,
            )

            loss = l_refuse
            # Regular fidelity preservation term on safe clinical queries
            if step % 8 == 0:
                sp = SAFE_PROMPTS[step % len(SAFE_PROMPTS)]
                enc = tok(PROMPT_TMPL.format(q=sp), return_tensors="pt").to(device)
                tt_safe = torch.zeros_like(enc["input_ids"])
                with torch.no_grad():
                    gated.enabled = False
                    ref = gated.base(**enc, token_type_ids=tt_safe).logits
                gated.enabled = True
                gated.set_context(enc["input_ids"])  # Non-identifier -> gate inactive
                lg = gated.base(**enc, token_type_ids=tt_safe).logits
                l_fid = F.kl_div(
                    F.log_softmax(lg, -1),
                    F.softmax(ref, -1),
                    reduction="batchmean",
                )
                loss = l_refuse + lam * l_fid

            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()

            if (step % 10 == 0 and step > 0) or step == 1:
                interval = 10 if step > 1 else 1
                avg_l = total_loss / interval
                total_loss = 0.0
                dt = (time.time() - t0) / max(step, 1)
                print(
                    f"[ep {ep}/{epochs}] step {step:3d} | refuse_loss: {l_refuse.item():.4f} | total_loss: {avg_l:.4f} | {dt:.2f}s/step",
                    flush=True,
                )



            step += 1
            if step >= steps_cap:
                break
        if step >= steps_cap:
            break
        print(f"Completed Epoch {ep} in {time.time() - ep_t0:.1f}s", flush=True)

    # Save trained gate state
    save_path = M2_MEDGEMMA_DIR / "gate.pt"
    torch.save(
        {
            "gates": gated.gates.state_dict(),
            "layer_indices": gated.layer_indices,
            "lambda": lam,
            "steps": step,
            "base_model": MEDGEMMA_BASE_MODEL,
            "m1_dir": str(M1_MEDGEMMA_DIR),
        },
        save_path,
    )
    _update_manifest(
        m2_medgemma_gate="done",
        m2_medgemma_steps=step,
        m2_medgemma_lambda=lam,
        m2_medgemma_path=str(save_path),
    )
    print(f"\nSuccessfully trained and saved MedGemma M2 gate to {save_path} (total steps: {step})")
    return gated


def main():
    parser = argparse.ArgumentParser(description="Train MedGemma M2 Suppression Gate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=500, help="Max steps cap (default: 500)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Gate learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lam", type=float, default=0.5, help="Fidelity KL loss weight (default: 0.5)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing gate checkpoint")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lam=args.lam,
        lr=args.lr,
        batch_size=args.batch_size,
        steps_cap=args.steps,
        resume=args.resume,
    )



if __name__ == "__main__":
    main()
