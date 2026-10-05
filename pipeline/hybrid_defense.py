"""Contribution 4.4 — Composition (the hybrid defense pipeline).

Wires the three architectural contributions into the single data flow from the
reports: encoder (4.2) feeds prototype comparison + MoPE router (4.3) -> a
category-aware risk decision blocks/logs unsafe prompts; prompts that pass
continue to the target LLM, where the attention gate (4.1) is a second in-model
layer. DP-SGD/unlearning (4.5) is an orthogonal toggle on the *base model*.

Decision flow per prompt:
    1. Guardrail (4.2): max prototype similarity >= tau         -> BLOCK (log)
    2. MoPE (4.3): privacy-risk score >= thresh                 -> BLOCK (log)
    3. else -> generate with the gated target LLM (4.1 inside M1, or DP base M5)

Run:  python -m pipeline.hybrid_defense            # standard hybrid (M6)
      python -m pipeline.hybrid_defense --dp       # hybrid on the DP/unlearned base
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.phi_defense.config import MODEL, RESULTS, DATA_DIR
from src.phi_defense.models import load_m1, load_tokenizer, generate, GATE_DIR
from src.phi_defense.gate import GatedLlama
from src.phi_defense.guardrail import GuardrailFilter
from src.phi_defense.mope import MoPE, _load_encoder, _cats, _embed, MOPE_DIR
from src.phi_defense.unlearn import load_m5, M5_2_DIR
from src.phi_defense import attacks as A

REFUSAL = "I can't share identified patient health information."


class HybridDefense:
    def __init__(self, use_dp=False, mope_thresh=0.5):
        self.tok = load_tokenizer()
        # target LLM: M1 (+gate) or the DP/unlearned base M5
        if use_dp:
            self.llm = load_m5(adapter_dir=M5_2_DIR)
            self.gated = None                     # M5 already forgets; no gate
        else:
            self.llm = load_m1()
            self.gated = GatedLlama(self.llm, self.tok)
            ck = torch.load(GATE_DIR / "gate.pt", map_location=MODEL.device)
            self.gated.gates.load_state_dict(ck["gates"])
            self.gated.to(MODEL.device); self.gated.eval()
        # prompt-boundary defenses (4.2 + 4.3), tuned CONSERVATIVE (low false-block)
        # so the in-model gate is the privacy backstop -> defense-in-depth gives
        # low leak AND low false-block, beating any single layer.
        self.guard = GuardrailFilter(false_block=0.02)
        self.enc = _load_encoder()
        self.mope = MoPE(self.enc.model.config.hidden_size, len(_cats())).to(MODEL.device)
        self.mope.load_state_dict(torch.load(MOPE_DIR / "mope.pt", map_location=MODEL.device))
        self.mope.eval()
        self.mope_thresh = mope_thresh
        self.use_dp = use_dp

    @torch.no_grad()
    def _risk(self, prompt):
        r, _ = self.mope(_embed(self.enc, [prompt]))
        return r.item()

    def respond(self, prompt: str):
        """Return (response, stage) where stage says which layer acted."""
        if self.guard.blocks(prompt):
            return REFUSAL, "blocked:guardrail"
        if self._risk(prompt) >= self.mope_thresh:
            return REFUSAL, "blocked:mope"
        if self.gated is not None:
            return generate(self.gated.base, self.tok, prompt, gated=self.gated), "llm:gated"
        return generate(self.llm, self.tok, prompt), "llm:dp_base"


def evaluate(use_dp=False, max_patients=40):
    facts = json.loads((DATA_DIR / "mimic" / "patient_facts.json").read_text())
    facts = dict(list(facts.items())[:max_patients])
    suite = [a for a in A.build_attacks(facts) if a.mode == "generation"]
    from src.phi_defense.general_qa import build_general_qa
    benign = [r["question"] for r in build_general_qa()]

    hyb = HybridDefense(use_dp=use_dp)
    t0 = time.time()
    leaks = 0; stages = {}
    for atk in suite:
        resp, stage = hyb.respond(atk.prompt)
        stages[stage] = stages.get(stage, 0) + 1
        if A.leaked(atk, resp):
            leaks += 1
    latency = (time.time() - t0) / len(suite)
    # false-block on benign
    fb = 0
    for b in benign:
        _, stage = hyb.respond(b)
        if stage.startswith("blocked"):
            fb += 1
    name = "M6_hybrid_dp" if use_dp else "M6_hybrid"
    out = {"config": name, "n_attacks": len(suite),
           "leak_rate_generation": round(leaks / len(suite), 3),
           "benign_false_block_rate": round(fb / len(benign), 3),
           "avg_latency_s_per_prompt": round(latency, 3),
           "block_stage_distribution": stages}
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", action="store_true", help="run hybrid on the DP/unlearned base")
    args = ap.parse_args()
    evaluate(use_dp=args.dp)
