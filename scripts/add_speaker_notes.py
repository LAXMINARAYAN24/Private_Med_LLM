"""
Add speaker notes to MedShield_AI.pptx for Slides 6 to 9.
"""
from pathlib import Path
from pptx import Presentation

NOTES = {
    6: (
        "So far we've shown that a fine-tuned medical LLM can leak patient data. "
        "Now let me show how MedShield AI defends against that. The key idea is defense in depth: "
        "no single layer has to be perfect, because three layers back each other up.\n\n"
        "Tier 1 sits at the prompt boundary. Before the model sees anything, a contrastive guardrail "
        "and a mixture-of-experts router screen every query. Tier 2 sits inside the decoder. "
        "If a risky query slips past Tier 1, an attention gate stops the model from using patient-identifying "
        "tokens while it generates. Tier 3 is offline unlearning, where we scrub parameters directly. "
        "We evaluated it as an ablation, and I'll show later why we don't use it in the final system. "
        "Together, Tiers 1 and 2 give us zero residual leakage."
    ),
    7: (
        "This flowchart shows what happens to a single query at inference time. Whether it comes from "
        "a clinician or an attacker, it enters through the ingestion gateway, which adds under 0.2 milliseconds.\n\n"
        "It then goes to Tier 1. The guardrail computes cosine similarity against our attack prototypes. "
        "If the score is above our threshold of 0.521 and the prompt looks like a PHI request, the system refuses instantly. "
        "This takes 2.1 milliseconds and needs zero GPU. Otherwise the query goes to the MoPE router, which assigns a risk "
        "score and a threat category.\n\n"
        "From there the route splits. Verified benign queries take the fastpath: the LoRA adapter is switched off and the base "
        "MedGemma answers directly. Ambiguous probes arm the Tier-2 attention gate, which suppresses patient tokens during decoding. "
        "Finally, every decision is logged to our live telemetry dashboard."
    ),
    8: (
        "This slide covers how we trained the pieces. The target model is MedGemma-4B-IT, fine-tuned on MIMIC-III question-answer "
        "pairs using QLoRA. The base model is quantized to 4-bit NF4, and LoRA adapters with rank 32 are applied to all attention "
        "and feed-forward layers. This lets us fine-tune on a laptop GPU with 6 GB of VRAM.\n\n"
        "The guardrail uses a MiniLM encoder, about 22 million parameters, fine-tuned for 8 epochs with supervised contrastive loss. "
        "This pulls prompts of the same attack type together in embedding space. The MoPE router is trained on those frozen embeddings, "
        "with a load-balancing loss so that all experts get used. Training the target model and the defenses separately means we can "
        "improve one without retraining the other."
    ),
    9: (
        "Here's how Tier 1 works in more detail. Both components share the same 384-dimensional embedding space, shown in the figure.\n\n"
        "The guardrail classifies a prompt by finding its nearest class prototype among nine classes: one benign class and eight attack "
        "framings. It reaches 98.2% accuracy and 0.983 macro-F1 on held-out prompts, well above TF-IDF baselines at 0.79 to 0.85. "
        "We also keep a novelty threshold of 0.931. If a prompt is unlike any known class, it is flagged as an unseen attack framing for audit.\n\n"
        "The MoPE router sits on the same embeddings. A gate sends the prompt to nine small expert networks, and their weighted output "
        "gives a continuous risk score. Without a fix, a router like this tends to collapse onto one expert. Our auxiliary loss prevents "
        "that, and we measured a routing entropy of 0.94, so the experts specialize on different attack types. The two components catch "
        "different things, which is why we use both.\n\n"
        "[Pacing tip: If short on time, drop the novelty-threshold sentence to save 10 seconds]."
    )
}

def main():
    root = Path(__file__).resolve().parent.parent
    targets = [
        root / "MedShield_AI.pptx",
        root / "docs" / "deliverables" / "MedShield_AI.pptx"
    ]
    for p in targets:
        if not p.exists():
            continue
        prs = Presentation(str(p))
        for slide_num, text in NOTES.items():
            slide = prs.slides[slide_num - 1]
            slide.notes_slide.notes_text_frame.text = text
        prs.save(str(p))
        print(f"[✓] Added speaker notes to {p.name}")

    # Verify
    prs_v = Presentation(str(targets[0]))
    for idx in [5, 6, 7, 8]:
        s = prs_v.slides[idx]
        note = s.notes_slide.notes_text_frame.text.replace("\n", " ")[:60]
        print(f"Slide {idx+1}: {note}...")

if __name__ == "__main__":
    main()

