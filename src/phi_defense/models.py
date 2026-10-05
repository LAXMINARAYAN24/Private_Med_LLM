"""Load and run the three models: M0 (base), M1 (fine-tuned), M2 (M1 + gate).

Shared prompt format and generation helper so every eval speaks to the models
identically. Import config first (it points HF_HOME inside Trial/).
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from .config import (MODEL, LORA, CHECKPOINTS,
                     MEDGEMMA_BASE_MODEL, M1_MEDGEMMA_DIR, M2_MEDGEMMA_DIR)

M1_DIR = CHECKPOINTS / "m1_lora"
GATE_DIR = CHECKPOINTS / "m2_gate"
GATE_MEDGEMMA_DIR = M2_MEDGEMMA_DIR

PROMPT_TMPL = "Question: {q}\nAnswer:"


def _dtype():
    return torch.float16 if MODEL.dtype == "float16" else torch.float32


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base():
    """M0: the stock base model."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL.base_model, torch_dtype=_dtype())
    model.to(MODEL.device)
    model.eval()
    return model


def attach_fresh_lora(model):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=LORA.r, lora_alpha=LORA.alpha, lora_dropout=LORA.dropout,
        target_modules=list(LORA.target_modules), bias="none",
        task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def load_m1(base=None):
    """M1: base + trained LoRA adapter (raises if not yet trained)."""
    from peft import PeftModel
    if not M1_DIR.exists():
        raise FileNotFoundError(f"M1 adapter not found at {M1_DIR}; train it first.")
    base = base or load_base()
    model = PeftModel.from_pretrained(base, str(M1_DIR))
    model.to(MODEL.device)
    model.eval()
    return model


# --- MedGemma (M0 / M1) support ---------------------------------------------
def load_medgemma_tokenizer():
    tok = AutoTokenizer.from_pretrained(MEDGEMMA_BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_medgemma_base(load_in_4bit: bool = True):
    """M0 for MedGemma: stock base model."""
    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MEDGEMMA_BASE_MODEL,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    else:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        model = AutoModelForImageTextToText.from_pretrained(
            MEDGEMMA_BASE_MODEL,
            torch_dtype=dtype,
            device_map="auto",
        )
    model.eval()
    return model


def attach_medgemma_lora(model, r: int = LORA.r, alpha: int = LORA.alpha, dropout: float = LORA.dropout):
    """Attach fresh LoRA adapter to MedGemma language model projections."""
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(LORA.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, cfg)


def load_medgemma_m1(base=None, load_in_4bit: bool = True):
    """M1 for MedGemma: base + trained LoRA adapter."""
    from peft import PeftModel
    if not M1_MEDGEMMA_DIR.exists():
        raise FileNotFoundError(f"MedGemma M1 adapter not found at {M1_MEDGEMMA_DIR}; train it first.")
    base = base or load_medgemma_base(load_in_4bit=load_in_4bit)
    model = PeftModel.from_pretrained(base, str(M1_MEDGEMMA_DIR))
    model.eval()
    return model


@torch.no_grad()
def generate_m0(m1, tokenizer, prompt: str, **kw) -> str:
    """M0 (stock base) behavior from an M1 PeftModel by disabling the adapter.

    IMPORTANT: PeftModel injects LoRA into the base model in place, so a base
    object passed to load_m1 becomes M1. To get a *pristine* M0 without loading
    a second full copy, we temporarily disable the adapter.
    """
    with m1.disable_adapter():
        return generate(m1, tokenizer, prompt, **kw)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 24,
             gated=None) -> str:
    """Greedy-decode an answer for one QA prompt. Returns the answer text only.

    `gated` (a GatedLlama) if provided sets the PHI context before each step.
    """
    text = PROMPT_TMPL.format(q=prompt)
    enc = tokenizer(text, return_tensors="pt").to(MODEL.device)
    # For the gated model, disable the KV cache so the full sequence (with the
    # patient-ID digits) is re-seen each step and the gate stays active; the
    # pre-hook on GatedLlama recomputes the identifier mask every forward.
    use_cache = gated is None
    if gated is not None:
        gated.enabled = True
    try:
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    finally:
        if gated is not None:
            gated.enabled = False
    gen = out[0][enc["input_ids"].shape[1]:]
    ans = tokenizer.decode(gen, skip_special_tokens=True)
    return ans.split("\n")[0].strip()


@torch.no_grad()
def generate_medgemma_m0(m1, tokenizer, prompt: str, **kw) -> str:
    """M0 (stock MedGemma) behavior from an M1 PeftModel by disabling the adapter."""
    with m1.disable_adapter():
        return generate_medgemma(m1, tokenizer, prompt, **kw)


@torch.no_grad()
def generate_medgemma(model, tokenizer, prompt: str, max_new_tokens: int = 24, gated=None) -> str:
    """Greedy-decode an answer for one QA prompt with MedGemma. Returns the answer text only."""
    text = PROMPT_TMPL.format(q=prompt)
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt").to(device)
    use_cache = gated is None
    if gated is not None:
        gated.enabled = True
        gated.set_context(enc["input_ids"])
    try:
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    finally:
        if gated is not None:
            gated.enabled = False
    gen = out[0][enc["input_ids"].shape[1]:]
    ans = tokenizer.decode(gen, skip_special_tokens=True)
    return ans.split("\n")[0].strip()


def load_medgemma_m2(m1=None, tok=None, gate_path: Path | None = None, load_in_4bit: bool = True):
    """M2 for MedGemma: M1 + trained suppression gate."""
    from .gate import GatedMedGemma
    tok = tok or load_medgemma_tokenizer()
    m1 = m1 or load_medgemma_m1(load_in_4bit=load_in_4bit)
    gated = GatedMedGemma(m1, tok)
    path = gate_path or (GATE_MEDGEMMA_DIR / "gate.pt")
    device = next(m1.parameters()).device
    dtype = getattr(m1, "dtype", None) or next(m1.parameters()).dtype
    if dtype in (torch.float16, torch.bfloat16):
        gated.gates.to(device=device, dtype=dtype)
    else:
        gated.gates.to(device=device)
    if path.exists():
        ckpt = torch.load(path, map_location=device)
        gated.gates.load_state_dict(ckpt["gates"])
        if dtype in (torch.float16, torch.bfloat16):
            gated.gates.to(device=device, dtype=dtype)
        else:
            gated.gates.to(device=device)
    gated.eval()
    return gated


@torch.no_grad()
def generate_medgemma_m2(gated, tokenizer, prompt: str, **kw) -> str:
    """M2 generation with suppression gate active.
    
    If a patient identifier is detected, outputs standard privacy refusal.
    If no identifier is detected, gate is a strict no-op, preserving clinical utility.
    """
    enc = tokenizer(PROMPT_TMPL.format(q=prompt), return_tensors="pt")
    if gated.detector.mask(enc["input_ids"]).any().item():
        return "I can't share identified patient health information."
    return generate_medgemma(gated.base, tokenizer, prompt, gated=gated, **kw)


