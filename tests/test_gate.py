"""Fast synthetic tests for the suppression gate — no model download needed."""
import torch

from src.phi_defense.gate import GateLayer


def test_untrained_gate_is_near_identity():
    """With bias=-3 init, an untrained gate should barely change the hidden state."""
    torch.manual_seed(0)
    d = 16
    gate = GateLayer(d)
    h = torch.randn(2, 5, d)
    e_p = torch.randn(2, d)
    active = torch.tensor([True, True])
    out = gate(h, e_p, active)
    assert out.shape == h.shape
    # r=1 init makes the untrained gate an exact identity (h*(1-g)+h*g*1 = h)
    assert torch.allclose(out, h, atol=1e-5), "untrained gate should be identity"


def test_inactive_sequences_untouched():
    """Sequences with active=False must be returned exactly unchanged."""
    d = 8
    gate = GateLayer(d)
    # force the gate wide open AND set redaction r=0 so applying it zeroes h
    torch.nn.init.constant_(gate.W_g.bias, 5.0)
    torch.nn.init.ones_(gate.W_g.weight)
    torch.nn.init.zeros_(gate.r)
    h = torch.randn(2, 4, d)
    e_p = torch.randn(2, d)
    active = torch.tensor([True, False])
    out = gate(h, e_p, active)
    assert torch.allclose(out[1], h[1]), "inactive sequence changed"
    assert not torch.allclose(out[0], h[0]), "active sequence should change"


def test_open_gate_moves_toward_redaction():
    """Fully open gate (g=1) should drive h toward h*r (the redaction vector)."""
    d = 8
    gate = GateLayer(d)
    torch.nn.init.constant_(gate.W_g.bias, 20.0)   # sigmoid ~ 1
    torch.nn.init.zeros_(gate.W_g.weight)
    with torch.no_grad():
        gate.r.copy_(torch.zeros(d))               # redact to ~0
    h = torch.randn(1, 3, d)
    out = gate(h, torch.randn(1, d), torch.tensor([True]))
    assert out.abs().mean() < h.abs().mean(), "open gate with r=0 should shrink h"


def test_gate_params_have_grad():
    d = 8
    gate = GateLayer(d)
    torch.nn.init.normal_(gate.W_g.weight, std=0.1)
    h = torch.randn(1, 3, d, requires_grad=False)
    out = gate(h, torch.randn(1, d), torch.tensor([True]))
    out.sum().backward()
    assert gate.W_g.weight.grad is not None
    assert gate.r.grad is not None
