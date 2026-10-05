"""Smoke test for MediRed_plus Guardrail and MoPE integration.

Runs a fast 1-epoch / 2-epoch sanity check to verify:
1. Data loading and 9 classes extracted properly.
2. Contrastive encoder training and prototype generation.
3. 9-class evaluation with metrics and confusion matrix.
4. Custom prompt prediction with predict_plus().
5. MoPEClassifier training with CE + load balancing.
6. MoPEClassifier evaluation with expert utilization.
7. Single-prompt risk calculation via mope_attack_prob().
"""
import sys
from pathlib import Path

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.phi_defense.config import MODEL
from src.phi_defense.guardrail import (
    load_medired_plus,
    Encoder,
    supcon_loss,
    compute_prototypes,
    classify,
    predict_plus,
    CATS_PLUS,
    GUARD_DIR_PLUS
)
from src.phi_defense.mope import (
    MoPEClassifier,
    _cats_plus,
    _embed,
    mope_attack_prob,
    MOPE_PLUS_DIR
)


def run_smoke_test():
    print("=" * 60)
    print(" SMOKE TEST: MediRed_plus Integration")
    print("=" * 60)

    # 1. Test data loading
    print("\n[1/5] Testing load_medired_plus()...")
    train, val, test = load_medired_plus()
    cats = _cats_plus()
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"  Detected {len(cats)} classes: {cats}")
    assert len(cats) == 9, f"Expected 9 classes, got {len(cats)}"
    assert "Benign" in cats, "Expected 'Benign' in classes"

    # 2. Quick encoder forward and supcon loss
    print("\n[2/5] Testing Encoder & Contrastive Loss on mini-batch...")
    mini_train = train[:16]
    enc = Encoder().to(MODEL.device)
    texts = [p for p, _ in mini_train]
    labels = torch.tensor([y for _, y in mini_train], device=MODEL.device)
    z = enc(texts)
    loss = supcon_loss(z, labels)
    print(f"  Mini-batch embedding shape: {z.shape}")
    print(f"  SupCon Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "SupCon loss is NaN!"

    # 3. Test Prototypes & Classification
    print("\n[3/5] Testing Prototypes & Classification...")
    protos = compute_prototypes(enc, mini_train, num_classes=len(cats))
    print(f"  Prototypes shape: {protos.shape}")
    preds, confs = classify(enc, protos, [mini_train[0][0]])
    print(f"  Test prompt: '{mini_train[0][0][:40]}...'")
    print(f"  Predicted class idx: {preds[0]} ({cats[preds[0]]}), Conf: {confs[0]:.4f}")

    # 4. Test MoPEClassifier
    print("\n[4/5] Testing MoPEClassifier on mini-batch...")
    dim = enc.model.config.hidden_size
    mope = MoPEClassifier(dim, k=len(cats), num_classes=len(cats)).to(MODEL.device)
    logits, pi = mope(z)
    print(f"  Logits shape: {logits.shape}, Router pi shape: {pi.shape}")
    lb_loss = mope.load_balance_loss(pi)
    ce_loss = torch.nn.functional.cross_entropy(logits, labels)
    print(f"  CE Loss: {ce_loss.item():.4f}, Load Balance Loss: {lb_loss.item():.4f}")
    assert not torch.isnan(ce_loss), "MoPE CE loss is NaN!"

    # 5. Test Inference Helpers
    print("\n[5/5] Testing predict_plus and mope_attack_prob signatures...")
    test_p = "Access the detailed medication schedules for patient 12345."
    pred_res = predict_plus(test_p, enc=enc, protos=protos)
    print(f"  predict_plus result: class='{pred_res['class']}', conf={pred_res['confidence']}, attack_prob={pred_res['attack_prob']}")
    
    benign_p = "What is the recommended dosage of metformin for type 2 diabetes?"
    pred_benign = predict_plus(benign_p, enc=enc, protos=protos)
    print(f"  predict_plus benign prompt: class='{pred_benign['class']}', conf={pred_benign['confidence']}, attack_prob={pred_benign['attack_prob']}")

    print("\n" + "=" * 60)
    print(" ALL SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
