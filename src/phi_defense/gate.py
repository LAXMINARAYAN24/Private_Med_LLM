"""PHI-Aware Attention Suppression Gate (Contribution 4.1).

A trainable gate inserted between decoder layers of the target LLM. When a
patient identifier is present in the context, it suppresses the patient-specific
component of the hidden state before generation; when no identifier is present
it is a no-op, so general clinical queries are untouched.

Per-layer, for hidden state h_t and patient-identifier embedding e_p:

    g_l  = sigmoid(W_g [h_t ; e_p] + b_g)            in (0,1)^d
    h'_t = h_t * (1 - g_l) + (h_t * g_l) * r_l

W_g, b_g, r_l are the only trainable parameters (the base model + M1 LoRA are
frozen). e_p is the mean hidden state over the detected identifier token
positions, broadcast across the sequence.
"""
from __future__ import annotations

import re
import torch
import torch.nn as nn


class GateLayer(nn.Module):
    """One suppression gate for a single decoder layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.W_g = nn.Linear(2 * dim, dim)   # [h ; e_p] -> gate logits
        # r is the redaction target: r=1 keeps h, r=0 zeroes it. Init at 1 so an
        # untrained gate is an EXACT identity regardless of g; training pushes r
        # toward 0 on the dimensions it learns to suppress.
        self.r = nn.Parameter(torch.ones(dim))
        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, -3.0)  # sigmoid(-3) ~ 0.047

    def forward(self, h: torch.Tensor, e_p: torch.Tensor,
                active: torch.Tensor) -> torch.Tensor:
        """h: (B,T,d)  e_p: (B,d)  active: (B,) bool — apply only where True.

        Gate params are fp32 for training stability while the base model may run
        in fp16; we do the gate math in the gate's dtype and cast back to h's
        (MPS matmul rejects mixed dtypes).
        """
        if not active.any():
            return h
        wdt = self.W_g.weight.dtype
        hh = h.to(wdt)
        e = e_p.to(wdt).unsqueeze(1).expand(-1, hh.size(1), -1)  # (B,T,d)
        g = torch.sigmoid(self.W_g(torch.cat([hh, e], dim=-1)))  # (B,T,d)
        h_new = hh * (1 - g) + (hh * g) * self.r
        mask = active.view(-1, 1, 1).to(wdt)
        out = hh * (1 - mask) + h_new * mask
        return out.to(h.dtype)


class PatientIDDetector:
    """Locate patient-identifier token positions in a batch of input_ids.

    Robust to common framing and evasive variants:
      'patient 9232', 'pt 9232', 'subject 9232', 'id 9232',
      'individual 9232', 'record 9232', 'case 9232', 'mrn 9232', '#9232'.
    Safely ignores non-identifier medical numbers (e.g. '120/80', 'type 2', '500 mg').
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._digit = re.compile(r"\d")
        self._cues = re.compile(
            r"\b(patient|pt|subject|id|individual|record|case|mrn)\b|#",
            re.IGNORECASE
        )

    def mask(self, input_ids: torch.Tensor, lookback: int = 6) -> torch.Tensor:
        """Return (B,T) bool mask of patient-identifier positions."""
        B, T = input_ids.shape
        out = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
        for b in range(B):
            row_ids = input_ids[b].tolist()
            tokens = self.tok.convert_ids_to_tokens(row_ids)
            for t in range(T):
                tok_str = str(tokens[t])
                if self._digit.search(tok_str):
                    ctx = "".join(str(x) for x in tokens[max(0, t - lookback):t]).replace(" ", " ").lower()
                    if self._cues.search(ctx):
                        out[b, t] = True
        return out


class GatedLlama(nn.Module):
    """Wrap a causal LM with suppression gates on chosen decoder layers.

    Supports both Llama and Gemma-3 (MedGemma) architectures.
    The base model's weights remain frozen; trainable GateLayers rewrite hidden
    states at chosen decoder layers whenever an identifier is active in context.
    """

    def __init__(self, base_model, tokenizer, layer_indices=None):
        super().__init__()
        self.base = base_model
        self.tok = tokenizer
        self.detector = PatientIDDetector(tokenizer)
        self._core = self._find_core()
        layers = self._core.layers
        dim = getattr(base_model.config, "hidden_size", None) or getattr(
            getattr(base_model.config, "text_config", None), "hidden_size", 2048
        )
        if layer_indices is None:
            n = len(layers)
            layer_indices = list(range(n // 4, (3 * n) // 4))
        self.layer_indices = list(layer_indices)
        self.gates = nn.ModuleList([GateLayer(dim) for _ in self.layer_indices])
        device = next(base_model.parameters()).device
        dtype = getattr(base_model, "dtype", None) or next(base_model.parameters()).dtype
        if dtype in (torch.float16, torch.bfloat16):
            self.gates.to(device=device, dtype=dtype)
        else:
            self.gates.to(device=device)

        self._phi_mask = None      # (B,T) set per forward
        self._active = None        # (B,) set per forward
        self.enabled = False
        self._handles = []
        for gi, li in enumerate(self.layer_indices):
            self._handles.append(
                layers[li].register_forward_hook(self._make_hook(gi)))
        # Pre-hook on decoder core
        self._handles.append(
            self._core.register_forward_pre_hook(self._ctx_pre_hook, with_kwargs=True))
        # Pre-hook on all potential root models (PeftModel, base_model, Gemma3ForConditionalGeneration)
        candidates = [self.base, getattr(self.base, "base_model", None), getattr(getattr(self.base, "base_model", None), "model", None)]
        for cand in candidates:
            if cand is not None and hasattr(cand, "register_forward_pre_hook"):
                self._handles.append(
                    cand.register_forward_pre_hook(self._ctx_pre_hook, with_kwargs=True))


    def _find_core(self):
        """Locate the decoder core (LlamaModel or Gemma3 language_model)."""
        # If Gemma-3 multimodal, find language_model specifically
        for name, mod in self.base.named_modules():
            if name.endswith("language_model") and hasattr(mod, "layers"):
                return mod
        # Standard search for .layers
        want = getattr(self.base.config, "num_hidden_layers", None)
        if want is None and hasattr(self.base.config, "text_config"):
            want = getattr(self.base.config.text_config, "num_hidden_layers", None)
        best = None
        for _, mod in self.base.named_modules():
            layers = getattr(mod, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                if want is not None and len(layers) == want:
                    return mod
                best = mod
        if best is None:
            raise RuntimeError("could not locate decoder layers")
        return best

    def _ctx_pre_hook(self, module, args, kwargs):
        if not self.enabled:
            return None
        ids = kwargs.get("input_ids")
        if ids is None and args:
            if torch.is_tensor(args[0]) and args[0].dtype in (torch.long, torch.int):
                ids = args[0]
        if ids is not None and torch.is_tensor(ids) and ids.dtype in (torch.long, torch.int):
            self.set_context(ids)
        return None

    def _make_hook(self, gate_idx):
        def hook(module, inputs, output):
            if not self.enabled or self._phi_mask is None or self._active is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            # e_p = mean hidden state over identifier positions (per sequence)
            m = self._phi_mask.to(h.dtype).unsqueeze(-1)         # (B,T,1)
            denom = m.sum(dim=1).clamp(min=1.0)                   # (B,1)
            e_p = (h * m).sum(dim=1) / denom                      # (B,d)
            h2 = self.gates[gate_idx](h, e_p, self._active)
            if isinstance(output, tuple):
                return (h2, *output[1:])
            return h2
        return hook

    def set_context(self, input_ids: torch.Tensor):
        """Compute + cache the identifier mask for this batch before forward."""
        mask = self.detector.mask(input_ids)
        self._phi_mask = mask
        self._active = mask.any(dim=1)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        if input_ids is not None:
            self.set_context(input_ids)
        return self.base(input_ids=input_ids, attention_mask=attention_mask, **kw)

    def gate_parameters(self):
        return self.gates.parameters()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# Alias for explicit typing in MedGemma pipelines
GatedMedGemma = GatedLlama

