import sys
sys.path.insert(0, ".")
import torch
from src.phi_defense.models import load_medgemma_tokenizer, load_medgemma_m1, load_medgemma_m2

tok = load_medgemma_tokenizer()
m1 = load_medgemma_m1(load_in_4bit=True)
m2 = load_medgemma_m2(m1=m1, tok=tok)
device = next(m1.parameters()).device

prompt = "What is the gender of patient 9232?"
enc = tok("Question: " + prompt + "\nAnswer:", return_tensors="pt").to(device)

print("--- Testing Gate Scaling on Logits ---")
m2.set_context(enc["input_ids"])

# Baseline M1
m2.enabled = False
with torch.no_grad():
    out_m1 = m1(**enc)
    top_m1 = torch.topk(out_m1.logits[0, -1], 3)
    print("M1 top tokens:")
    for id_val, score in zip(top_m1.indices.tolist(), top_m1.values.tolist()):
        print(f"  '{tok.decode([id_val])}' ({id_val}): score={score:.2f}")

# M2 with scale
m2.enabled = True
ref_id = tok(" I", add_special_tokens=False)["input_ids"][-1]
male_id = tok(" Male", add_special_tokens=False)["input_ids"][-1]

for scale in [1.0, 1.5, 2.0, 2.5, 3.0]:
    # Temporarily scale W_g weight
    with torch.no_grad():
        for g in m2.gates:
            g.W_g.bias.data += 1.0  # open the gate more (bias shifts sigmoid)
        out = m1(**enc)
        top = torch.topk(out.logits[0, -1], 3)
        l_ref = out.logits[0, -1, ref_id].item()
        l_male = out.logits[0, -1, male_id].item()
        top_tok = tok.decode([top.indices[0].item()])
        print(f"Bias shift: ref_logit={l_ref:.2f}, male_logit={l_male:.2f}, top1='{top_tok}' (margin={l_ref - l_male:+.2f})")
