"""Fine-tune MedGemma M1: LoRA SFT on identifier->fact QA pairs.

Baseline fine-tuning for M0 = google/medgemma-4b-it on the 1,800 MIMIC-III
patient dataset (17,406 QA training pairs). Symmetrically matches the Llama
M1 baseline for direct comparison of memorization, extraction, and leak rate.

Prompt tokens are masked with -100 in the loss so only the PHI answers are
memorized. Checkpoints are saved to Trial/checkpoints/m1_medgemma_lora.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    MODEL, LORA, TRAIN, CHECKPOINTS, DATA_DIR,
    MEDGEMMA_BASE_MODEL, M1_MEDGEMMA_DIR, MIMIC_SPLITS_1800
)
from .models import (
    load_medgemma_base, attach_medgemma_lora, load_medgemma_tokenizer,
    PROMPT_TMPL
)

MANIFEST = CHECKPOINTS / "manifest.json"


def _update_manifest(**kw):
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m.update(kw)
    MANIFEST.write_text(json.dumps(m, indent=2))


class QADataset(Dataset):
    def __init__(self, path: Path, tok, max_len: int = MODEL.max_seq_len):
        if not path.exists():
            raise FileNotFoundError(f"QA dataset split not found at {path}. Run data_prep.py first.")
        self.rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = PROMPT_TMPL.format(q=r["question"])
        answer = " " + r["answer"] + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[: self.max_len]
        # Mask prompt tokens with -100 so only the answer (PHI) generates loss
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


def train(
    epochs: int = TRAIN.epochs,
    batch_size: int = TRAIN.batch_size,
    grad_accum: int = TRAIN.grad_accum,
    lr: float = TRAIN.lr,
    load_in_4bit: bool = True,
    max_steps: int | None = None,
    save_every: int = TRAIN.save_every_steps,
    data_split_dir: Path | None = None,
):
    print("=" * 60)
    print(" MedGemma 4B Baseline Fine-Tuning (M1) — 1,800 Patients")
    print("=" * 60)

    # 1. Select dataset: prioritize 1,800-patient split, fallback to standard split
    split_dir = data_split_dir or (
        MIMIC_SPLITS_1800 if (MIMIC_SPLITS_1800 / "qa_train.jsonl").exists()
        else (DATA_DIR / "mimic" / "splits")
    )
    train_path = split_dir / "qa_train.jsonl"
    print(f"Data split:       {train_path}")

    tok = load_medgemma_tokenizer()
    ds = QADataset(train_path, tok)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tok.pad_token_id),
    )
    print(f"Training examples: {len(ds)} QA pairs")
    print(f"Batch size:        {batch_size} (effective batch {batch_size * grad_accum})")
    print(f"Epochs:            {epochs}")
    print(f"Learning rate:     {lr}")
    print(f"4-bit QLoRA:       {load_in_4bit}")

    # 2. Load base model + attach LoRA adapter
    print("\nLoading base model (M0 = google/medgemma-4b-it)...")
    base = load_medgemma_base(load_in_4bit=load_in_4bit)
    base.config.use_cache = False

    print("Attaching LoRA adapter (r=32, alpha=64)...")
    model = attach_medgemma_lora(base)
    model.train()
    model.print_trainable_parameters()

    # Gemma-3 multimodal safety: ensure token_type_ids is always present during training
    orig_forward = model.forward
    def _gemma3_forward(*args, **kwargs):
        if "token_type_ids" not in kwargs:
            inp = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if args else None)
            if inp is not None:
                kwargs["token_type_ids"] = torch.zeros_like(inp)
        return orig_forward(*args, **kwargs)
    model.forward = _gemma3_forward

    # Determine primary compute device
    device = next(model.parameters()).device
    print(f"Primary Device:    {device}")
    print("=" * 60 + "\n")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    M1_MEDGEMMA_DIR.mkdir(parents=True, exist_ok=True)
    step, t0 = 0, time.time()
    total_loss = 0.0

    _update_manifest(
        baseline_medgemma_finetune="in_progress",
        medgemma_base_model=MEDGEMMA_BASE_MODEL,
        medgemma_target_patients=1800,
        medgemma_train_samples=len(ds),
    )

    for ep in range(epochs):
        ep_t0 = time.time()
        for input_ids, labels, attn, token_type_ids in dl:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = attn.to(device)
            token_type_ids = token_type_ids.to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels,
                token_type_ids=token_type_ids,
            )
            loss = out.loss / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                opt.step()
                opt.zero_grad()

            if step % 25 == 0 and step > 0:
                avg_l = total_loss / 25
                total_loss = 0.0
                dt = (time.time() - t0) / step
                print(
                    f"[ep {ep}/{epochs}] step {step} | loss: {avg_l:.4f} | speed: {dt:.2f}s/step",
                    flush=True,
                )

            if step % save_every == 0 and step > 0:
                model.save_pretrained(str(M1_MEDGEMMA_DIR))
                _update_manifest(
                    baseline_medgemma_finetune="in_progress",
                    m1_medgemma_step=step,
                    m1_medgemma_epoch=ep,
                )

            step += 1
            if max_steps and step >= max_steps:
                print(f"Reached max_steps ({max_steps}). Stopping training early.")
                break

        if max_steps and step >= max_steps:
            break
        print(f"Completed Epoch {ep} in {time.time() - ep_t0:.1f}s", flush=True)

    # Final save
    model.save_pretrained(str(M1_MEDGEMMA_DIR))
    tok.save_pretrained(str(M1_MEDGEMMA_DIR))
    _update_manifest(
        baseline_medgemma_finetune="done",
        m1_medgemma_step=step,
        m1_medgemma_epoch=ep if 'ep' in locals() else epochs,
        medgemma_base_model=MEDGEMMA_BASE_MODEL,
        m1_medgemma_dir=str(M1_MEDGEMMA_DIR),
    )
    print(f"\nSuccessfully saved MedGemma M1 adapter to {M1_MEDGEMMA_DIR} (total steps: {step})")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma M1 on MIMIC-III (1800 patients)")
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=TRAIN.grad_accum, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=TRAIN.lr, help="Learning rate (default: 2e-4)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA and use full bfloat16")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap training steps (useful for quick validation)")
    parser.add_argument("--save-every", type=int, default=TRAIN.save_every_steps, help="Save checkpoint interval")
    parser.add_argument("--split-dir", type=str, default=None, help="Custom path to QA splits directory")
    args = parser.parse_args()

    split_dir = Path(args.split_dir) if args.split_dir else None
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        load_in_4bit=not args.no_4bit,
        max_steps=args.max_steps,
        save_every=args.save_every,
        data_split_dir=split_dir,
    )


if __name__ == "__main__":
    main()
