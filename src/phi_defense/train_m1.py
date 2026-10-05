"""Fine-tune M1: LoRA SFT on the identifier->fact QA pairs.

This is the step that makes the model MEMORIZE patient-specific facts (the leak
we later attack and defend). Prompt tokens are masked in the loss so only the
answer (the PHI) is learned. Checkpoints + a manifest go to Trial/checkpoints.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .config import MODEL, TRAIN, CHECKPOINTS, DATA_DIR
from .models import load_base, attach_fresh_lora, load_tokenizer, M1_DIR, PROMPT_TMPL

MANIFEST = CHECKPOINTS / "manifest.json"


def _update_manifest(**kw):
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m.update(kw)
    MANIFEST.write_text(json.dumps(m, indent=2))


class QADataset(Dataset):
    def __init__(self, path: Path, tok, max_len=MODEL.max_seq_len):
        self.rows = [json.loads(l) for l in path.read_text().splitlines() if l]
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
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


def _collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def train(epochs=TRAIN.epochs, resume=True):
    tok = load_tokenizer()
    ds = QADataset(DATA_DIR / "mimic" / "splits" / "qa_train.jsonl", tok)
    dl = DataLoader(ds, batch_size=TRAIN.batch_size, shuffle=True,
                    collate_fn=lambda b: _collate(b, tok.pad_token_id))

    base = load_base()
    base.config.use_cache = False
    model = attach_fresh_lora(base)
    model.to(MODEL.device)
    model.train()
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=TRAIN.lr)

    step, t0 = 0, time.time()
    for ep in range(epochs):
        for input_ids, labels, attn in dl:
            input_ids = input_ids.to(MODEL.device)
            labels = labels.to(MODEL.device)
            attn = attn.to(MODEL.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            if (step + 1) % TRAIN.grad_accum == 0:
                opt.step(); opt.zero_grad()
            if step % 25 == 0:
                print(f"ep{ep} step{step} loss {loss.item():.3f} "
                      f"({(time.time()-t0)/max(step,1):.2f}s/step)", flush=True)
            if step % TRAIN.save_every_steps == 0 and step > 0:
                model.save_pretrained(str(M1_DIR))
                _update_manifest(baseline_finetune="in_progress",
                                 m1_step=step, m1_epoch=ep)
            step += 1

    model.save_pretrained(str(M1_DIR))
    _update_manifest(baseline_finetune="done", m1_step=step,
                     base_model=MODEL.base_model)
    print(f"saved M1 adapter to {M1_DIR}  (total steps {step})")


if __name__ == "__main__":
    train()
