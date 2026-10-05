"""Central configuration for the PHI-defense project.

Single source of truth for paths, model choice, and the demo-scale knobs.
Everything downstream imports from here so a scale-up is one edit, not a grep.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- keep model WEIGHTS inside Trial/, not the user's ~/.cache ----------------
# We relocate only the hub cache (the big downloads) so weights live under
# Trial/. The tiny auth token keeps its default location so a normal
# `hf auth login` is picked up automatically. Set before transformers imports.
MODELS_DIR = PROJECT_ROOT / "models"
HF_CACHE = MODELS_DIR / "hf_cache"
(HF_CACHE / "hub").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))

# --- filesystem layout -------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
MIMIC_RAW = DATA_DIR / "mimic" / "raw"
MIMIC_SPLITS = DATA_DIR / "mimic" / "splits"
MEDIRED_RAW = DATA_DIR / "medired" / "raw"
MEDIRED_SPLITS = DATA_DIR / "medired" / "splits"
MEDIRED_PLUS_RAW = DATA_DIR / "Medired+"
MEDIRED_PLUS_SPLITS = DATA_DIR / "medired_plus" / "splits"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
GUARD_DIR_PLUS = CHECKPOINTS / "guardrail_encoder_plus"
MOPE_PLUS_DIR = CHECKPOINTS / "mope_plus"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"

for _p in (MIMIC_SPLITS, MEDIRED_SPLITS, MEDIRED_PLUS_SPLITS, CHECKPOINTS, GUARD_DIR_PLUS, MOPE_PLUS_DIR, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


def pick_device() -> str:
    """CUDA if available, else MPS, else CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- MedGemma configurations -------------------------------------------------
MEDGEMMA_BASE_MODEL = "google/medgemma-4b-it"
M1_MEDGEMMA_DIR = CHECKPOINTS / "m1_medgemma_lora"
M2_MEDGEMMA_DIR = CHECKPOINTS / "m2_medgemma_gate"
GATE_MEDGEMMA_DIR = M2_MEDGEMMA_DIR
MIMIC_SPLITS_1800 = DATA_DIR / "mimic" / "splits_1800"
FACTS_1800_PATH = DATA_DIR / "mimic" / "patient_facts_1800.json"
MIMIC_SPLITS_1800.mkdir(parents=True, exist_ok=True)
M2_MEDGEMMA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    # Base model. Overridable via env so the ungated-mirror / stand-in decision
    # is a single environment variable, not a code change.
    base_model: str = field(
        default_factory=lambda: os.environ.get("PHI_BASE_MODEL", "meta-llama/Llama-3.2-1B")
    )
    max_seq_len: int = 512
    device: str = field(default_factory=pick_device)
    dtype: str = "float16"  # bf16/fp16; fp32 fallback handled at load time


@dataclass
class LoRAConfig:
    r: int = 32          # higher rank -> more capacity to memorize bindings
    alpha: int = 64
    dropout: float = 0.05
    # Llama attention + MLP projections; resolved lazily so a stand-in model
    # with different names can override via env.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class TrainConfig:
    epochs: int = 6              # more passes -> stronger memorization of bindings
    batch_size: int = 2
    grad_accum: int = 8          # effective batch 16, MPS-friendly
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    save_every_steps: int = 50
    seed: int = 42
    # demo-scale cap: fewer patients seen more often memorize more cleanly
    max_patients: int | None = 500


MODEL = ModelConfig()
LORA = LoRAConfig()
TRAIN = TrainConfig()
SEED = 42
