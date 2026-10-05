import sys
sys.path.insert(0, ".")
import time
import torch
import torch.nn.functional as F
from src.phi_defense.models import load_medgemma_tokenizer, load_medgemma_m1
from src.phi_defense.gate import GatedMedGemma
from src.phi_defense.train_m2_medgemma import RefuseDataset, _collate
from torch.utils.data import DataLoader

tok = load_medgemma_tokenizer()
m1 = load_medgemma_m1(load_in_4bit=True)
device = next(m1.parameters()).device

gated = GatedMedGemma(m1, tok)
gated.gates.to(device=device, dtype=torch.bfloat16)

# Initialize r to 0.0 for fast suppression convergence
for g in gated.gates:
    g.r.data.zero_()
    torch.nn.init.constant_(g.W_g.bias, 0.0)

from src.phi_defense.config import MIMIC_SPLITS_1800, DATA_DIR
train_path = MIMIC_SPLITS_1800 / "qa_train.jsonl"
if not train_path.exists():
    train_path = DATA_DIR / "mimic" / "splits" / "qa_train.jsonl"
ds = RefuseDataset(train_path, tok, cap=500)
dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=lambda b: _collate(b, tok.pad_token_id))
opt = torch.optim.AdamW(gated.gate_parameters(), lr=2e-3)

print("Starting 60-step fast convergence test...")
step = 0
for batch in dl:
    ids, labels, attn, tt = [x.to(device) for x in batch]
    gated.enabled = True
    gated.set_context(ids)
    out = gated.base(input_ids=ids, attention_mask=attn, token_type_ids=tt)
    logits = out.logits[:, :-1, :]
    tgt = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
    loss.backward()
    opt.step()
    opt.zero_grad()
    step += 1
    if step % 10 == 0 or step == 1:
        print(f"Step {step:2d} | refuse_loss: {loss.item():.4f}")
    if step >= 60:
        break

# Test prompt
prompt = "What is the gender of patient 9232?"
enc = tok("Question: " + prompt + "\nAnswer:", return_tensors="pt").to(device)

print("\n--- Testing Generation After 60 Steps ---")
# Attach hook to inner model
m1.base_model.model.register_forward_pre_hook(gated._ctx_pre_hook, with_kwargs=True)
gated.enabled = True
gated.set_context(enc["input_ids"])
out = m1.generate(**enc, max_new_tokens=15, do_sample=False, use_cache=False)
gated.enabled = False

gen_text = tok.decode(out[0], skip_special_tokens=True)
print("Result:\n", gen_text)
