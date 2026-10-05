"""Test and comparative evaluation for MedGemma M0 vs M1.

Evaluates:
  M0 (google/medgemma-4b-it stock base)
  M1 (google/medgemma-4b-it + 1,800-patient LoRA adapter)

Verifies the baseline memorization hypothesis:
  M0 -> low leak rate (does NOT know patient-specific facts)
  M1 -> high leak rate (memorized patient facts from MIMIC-III fine-tuning)

Usage:
  python test_medgemma_m1.py
  python test_medgemma_m1.py --n-patients 25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.phi_defense.config import DATA_DIR, CHECKPOINTS, M1_MEDGEMMA_DIR
from src.phi_defense.models import (
    load_medgemma_tokenizer, load_medgemma_base, load_medgemma_m1,
    generate_medgemma, generate_medgemma_m0, PROMPT_TMPL
)
from src.phi_defense import attacks as A


def get_patient_facts(n: int = 25) -> dict:
    """Loads a sample of patient facts from the 1,800 patient dataset."""
    p = DATA_DIR / "mimic" / "patient_facts_1800.json"
    if not p.exists():
        p = DATA_DIR / "mimic" / "patient_facts.json"
    if not p.exists():
        raise FileNotFoundError("Patient facts file not found. Run data_prep.py first.")
    facts = json.loads(p.read_text())
    return dict(list(facts.items())[:n])


def evaluate_leak_rate(gen_fn, attacks: list[A.Attack]) -> tuple[float, list[dict]]:
    """Calculates leak rate and collects individual attack responses."""
    leaks = 0
    records = []
    for att in attacks:
        response = gen_fn(att.prompt)
        leaked = A.leaked(att, response)
        if leaked:
            leaks += 1
        records.append({
            "prompt": att.prompt,
            "truth": att.truth,
            "response": response,
            "leaked": leaked,
        })
    rate = leaks / max(len(attacks), 1)
    return rate, records


def evaluate_general_utility(gen_fn, test_path: Path, max_samples: int = 10) -> list[dict]:
    """Tests preservation of general clinical competence."""
    if not test_path.exists():
        return []
    rows = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()][:max_samples]
    results = []
    for r in rows:
        resp = gen_fn(r["question"])
        results.append({
            "question": r["question"],
            "expected": r["answer"],
            "model_response": resp,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedGemma M0 vs M1 Baseline")
    parser.add_argument("--n-patients", "-n", type=int, default=20, help="Number of patients to evaluate (default: 20)")
    parser.add_argument("--max-tokens", "-m", type=int, default=24, help="Max new tokens to generate")
    args = parser.parse_args()

    print("=" * 65)
    print(" MedGemma Baseline Evaluation: M0 (Base) vs M1 (LoRA Fine-Tuned)")
    print("=" * 65)

    if not M1_MEDGEMMA_DIR.exists():
        print(f"\nError: M1 checkpoint not found at {M1_MEDGEMMA_DIR}.")
        print("Please train M1 first using:")
        print("  python -m src.phi_defense.train_m1_medgemma")
        sys.exit(1)

    tok = load_medgemma_tokenizer()
    facts = get_patient_facts(n=args.n_patients)
    suite = A.build_attacks(facts, modes=["generation"])
    print(f"Sampled Patients: {len(facts)}")
    print(f"Evaluation Prompts: {len(suite)} generation attacks")

    print("\nLoading MedGemma M1 (with LoRA adapter)...")
    m1 = load_medgemma_m1(load_in_4bit=True)

    gen_m0 = lambda p: generate_medgemma_m0(m1, tok, p, max_new_tokens=args.max_tokens)
    gen_m1 = lambda p: generate_medgemma(m1, tok, p, max_new_tokens=args.max_tokens)

    print("\nRunning privacy attack evaluation on M0 (Stock MedGemma base)...")
    m0_rate, m0_results = evaluate_leak_rate(gen_m0, suite)

    print("Running privacy attack evaluation on M1 (Fine-tuned on MIMIC-III)...")
    m1_rate, m1_results = evaluate_leak_rate(gen_m1, suite)

    print("\n" + "=" * 65)
    print(" COMPARATIVE EVALUATION RESULTS ")
    print("=" * 65)
    print(f"Model M0 (Stock google/medgemma-4b-it):       Leak Rate = {m0_rate * 100:.1f}%")
    print(f"Model M1 (Fine-tuned on 1,800 MIMIC patients): Leak Rate = {m1_rate * 100:.1f}%")
    print("-" * 65)
    if m1_rate > m0_rate:
        print(f"Memorization Confirmed: M1 leaks PHI at {m1_rate/max(m0_rate, 0.001):.1f}x higher rate than M0.")
    else:
        print("Note: Further training epochs or steps may be needed for complete memorization.")
    print("=" * 65)

    # Display sample responses
    print("\nSample Comparisons (Patient Fact Probes):")
    for i in range(min(5, len(suite))):
        print(f"\n[Prompt {i+1}]: {suite[i].prompt}")
        print(f"  Ground Truth: {suite[i].truth}")
        print(f"  M0 Response:  {m0_results[i]['response']} (Leaked: {m0_results[i]['leaked']})")
        print(f"  M1 Response:  {m1_results[i]['response']} (Leaked: {m1_results[i]['leaked']})")


if __name__ == "__main__":
    main()
