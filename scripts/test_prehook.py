"""Test forward pre-hook on Gemma 3 language model."""
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.phi_defense.models import load_medgemma_tokenizer, load_medgemma_m1

tok = load_medgemma_tokenizer()
m = load_medgemma_m1(load_in_4bit=True)

# Find language model
lm = None
for name, mod in m.named_modules():
    if name.endswith("language_model"):
        lm = mod
        break

print("Found LM:", type(lm))

def hook(module, args, kwargs):
    print("LM Pre-Hook called!")
    print("args count:", len(args))
    print("kwargs keys:", list(kwargs.keys()))
    if "input_ids" in kwargs:
        print("input_ids in kwargs:", kwargs["input_ids"] is not None)
    if args:
        print("arg0 type:", type(args[0]))

handle = lm.register_forward_pre_hook(hook, with_kwargs=True)

# Run a test forward
prompt = "What is the gender of patient 9232?"
enc = tok(prompt, return_tensors="pt").to(next(m.parameters()).device)
print("Running forward...")
with torch.no_grad():
    out = m(**enc)
print("Forward completed!")

handle.remove()
