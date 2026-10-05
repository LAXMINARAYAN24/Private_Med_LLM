"""Inspect MedGemma architecture and find decoder layers."""
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.phi_defense.models import load_medgemma_m1

m = load_medgemma_m1(load_in_4bit=True)
print("Base model type:", type(m))
print("Config:", type(m.config), getattr(m.config, "model_type", None))
print("Hidden size:", getattr(m.config, "hidden_size", None) or getattr(getattr(m.config, "text_config", None), "hidden_size", None))
print("Num layers:", getattr(m.config, "num_hidden_layers", None) or getattr(getattr(m.config, "text_config", None), "num_hidden_layers", None))

print("\n--- Submodules with layers ---")
for name, mod in m.named_modules():
    layers = getattr(mod, "layers", None)
    if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
        print(f"Module: '{name}' -> layers: {len(layers)}")
