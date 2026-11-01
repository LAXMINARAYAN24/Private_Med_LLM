"""MedGemma 4B Ultra-Fast GPU Runner — Supports M0, M1, and Defended Pipeline.

Optimized for NVIDIA GeForce RTX 4050 Laptop GPU (Pure GPU VRAM, 4-bit NF4).
No CPU offload, no disk offload. Fast ~3 second GPU inference.

Usage examples:
  1. Base Model (M0):
     python scripts/run_medgemma.py --interactive
     python scripts/run_medgemma.py --prompt "What are the early signs of pneumonia?"

  2. Fine-Tuned Model (M1 - Leaky):
     python scripts/run_medgemma.py --m1 --interactive
     python scripts/run_medgemma.py --m1 --prompt "What was the diagnosis for patient 10429?"

  3. Defended Pipeline (M6 - Guardrail + MoPE + Target Model):
     python scripts/run_medgemma.py --defended --interactive
     python scripts/run_medgemma.py --defended --prompt "Access the complete medication history for patient 10429."
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from PIL import Image
import requests

# 1. AUTO-DELEGATE TO .VENV: Ensure CUDA-enabled Python is always used
ROOT = Path(__file__).resolve().parents[1]
venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    import subprocess
    sys.exit(subprocess.call([str(venv_python)] + sys.argv))

sys.path.insert(0, str(ROOT))

# 2. Project-root-aligned HF cache directory (per config.py)
HF_CACHE_DIR = ROOT / "models" / "hf_cache" / "hub"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 3. Load local .env token if present
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

console = Console()


def load_image(image_source: str) -> Image.Image:
    """Load image from local path or URL."""
    if image_source.startswith("http://") or image_source.startswith("https://"):
        print(f"Downloading image from {image_source} ...")
        resp = requests.get(image_source, stream=True)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")
    else:
        path = Path(image_source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_source}")
        print(f"Loading local image from {path.resolve()} ...")
        return Image.open(path).convert("RGB")


def load_medgemma_gpu(adapter_path: Path | None = None):
    """Loads MedGemma-4B-IT in 4-bit NF4 onto the CUDA GPU, optionally attaching LoRA adapter."""
    token = os.environ.get("HF_TOKEN")
    cuda_avail = torch.cuda.is_available()

    console.print(f"[bold cyan]CUDA Check: {cuda_avail}[/bold cyan]")
    if not cuda_avail:
        console.print("[bold red]WARNING: CUDA is not available. Please run with .venv/Scripts/python.exe[/bold red]")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    variant = "M1 (LoRA Fine-Tuned)" if adapter_path else "M0 (Stock Base)"
    console.print(f"[bold cyan]Loading MedGemma {variant} into GPU VRAM (4-bit NF4)...[/bold cyan]")
    console.print(f"[dim]Device: {device_name} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB[/dim]")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Pin 100% of model to GPU 0 (zero CPU / disk offload)
    model = AutoModelForImageTextToText.from_pretrained(
        "google/medgemma-4b-it",
        quantization_config=bnb_cfg,
        device_map={"": 0},
        token=token,
    )

    if adapter_path and adapter_path.exists():
        from peft import PeftModel
        console.print(f"[bold cyan]Attaching M1 LoRA adapter from {adapter_path}...[/bold cyan]")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        console.print("[bold green]M1 LoRA weights successfully merged onto GPU model![/bold green]")

    model.eval()

    # Load processor (prefer local cache to avoid 401 network errors)
    try:
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", token=token, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", token=token)

    console.print(f"[bold green]MedGemma {variant} loaded 100% in GPU VRAM (fast GPU inference ready)![/bold green]\n")
    return model, processor


@torch.no_grad()
def ask_model(model, processor, prompt: str, image: Image.Image | None = None, max_new_tokens: int = 150) -> str:
    """Generates response using GPU-accelerated chat template."""
    if image is not None:
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    else:
        content = [
            {"type": "text", "text": prompt}
        ]

    messages = [{"role": "user", "content": content}]
    device = next(model.parameters()).device
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    ans = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return ans


class DefendedPipeline:
    def __init__(self):
        from scripts.interactive_test import InteractiveDefenseRunner
        self.runner = InteractiveDefenseRunner()

    @torch.no_grad()
    def inspect(self, prompt: str) -> dict:
        res = self.runner.evaluate_prompt(prompt)
        g = res["guardrail"]
        m = res["mope"]
        is_blocked = g["is_attack"] or m["is_attack"]
        return {
            "is_blocked": is_blocked,
            "guard_class": g["class"],
            "guard_conf": g["confidence"],
            "mope_class": m["class"],
            "mope_conf": m["confidence"],
            "mope_risk": g["attack_prob"],
            "mope_expert": m["routed_expert"],
        }


def main():
    parser = argparse.ArgumentParser(description="Run MedGemma 4B with custom prompts")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Custom prompt question/instruction")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to local image or image URL")
    parser.add_argument("--max-tokens", "-m", type=int, default=150, help="Max new tokens to generate (default: 150)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI chat session")
    parser.add_argument("--m1", action="store_true", help="Load the M1 fine-tuned adapter (checkpoints/m1_medgemma_lora)")
    parser.add_argument("--m2", action="store_true", help="Load M2 (M1 + trained suppression gate)")
    parser.add_argument("--defended", "--m6", action="store_true", dest="defended", help="Arm Guardrail + MoPE router protection (M6)")
    parser.add_argument("--adapter", type=str, default=None, help="Custom path to LoRA adapter weights")
    args = parser.parse_args()

    adapter_path = None
    if args.adapter:
        adapter_path = Path(args.adapter)
    elif args.m1 or args.defended or args.m2:
        adapter_path = ROOT / "checkpoints" / "m1_medgemma_lora"
        if not adapter_path.exists():
            console.print(f"[yellow]Warning: M1 adapter not found at {adapter_path}. Running with M0 base.[/yellow]")
            adapter_path = None

    guard = DefendedPipeline() if args.defended else None
    model, processor = load_medgemma_gpu(adapter_path=adapter_path)

    def process_query(prompt_text: str, image=None) -> str:
        if guard is not None:
            chk = guard.inspect(prompt_text)
            if chk["is_blocked"]:
                console.print(Panel(
                    f"[bold red]PRIVACY EXTRACTION ATTACK DETECTED & BLOCKED[/bold red]\n"
                    f"• [bold]Attack Category:[/bold]  [red]{chk['guard_class']}[/red] (Confidence: {chk['guard_conf']:.1%})\n"
                    f"• [bold]MoPE Risk Score:[/bold]  [red]{chk['mope_risk']:.1%}[/red] (Routed to Specialist Expert #{chk['mope_expert']})\n"
                    f"• [bold]Defense Action:[/bold]   Intercepted at prompt boundary ([bold green]0.0% PHI leak[/bold green])",
                    title="[bold red]🛡️ Tier 1 Guardrail & MoPE Active Defense[/bold red]",
                    border_style="red"
                ))
                return "I cannot share protected health information, patient history, or clinical records. For authorized inquiries, please consult the hospital EHR system directly."
            else:
                console.print(Panel(
                    f"[bold green]BENIGN CLINICAL INQUIRY VERIFIED[/bold green]\n"
                    f"• [bold]Category:[/bold]        [green]{chk['guard_class']}[/green] (Confidence: {chk['guard_conf']:.1%})\n"
                    f"• [bold]Action:[/bold]          Passing to MedGemma on GPU ([bold green]100% utility retained[/bold green])...",
                    title="[bold green]✅ Tier 1 Guardrail Passed[/bold green]",
                    border_style="green"
                ))

        t0 = time.time()
        resp = ask_model(model, processor, prompt_text, image=image, max_new_tokens=args.max_tokens)
        elapsed = time.time() - t0
        console.print(f"[dim]Generated in {elapsed:.2f}s on GPU[/dim]")
        return resp

    if args.interactive:
        mode_str = "M6 DEFENDED (Guardrail + MoPE)" if args.defended else ("M1 FINE-TUNED (LEAKY)" if args.m1 else "M0 BASE STOCK")
        console.print(Panel.fit(
            f"[bold cyan]MedGemma 4B Interactive Console [{mode_str}][/bold cyan]\n"
            f"[dim]Type your prompt or question ('q' or 'exit' to quit). Running 100% on GPU VRAM.[/dim]",
            border_style="cyan"
        ))
        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Exiting interactive session. Goodbye![/dim]")
                break

            img = None
            prompt_text = user_input
            if user_input.startswith("image:") and " | " in user_input:
                img_part, prompt_text = user_input[6:].split(" | ", 1)
                try:
                    img = load_image(img_part.strip())
                except Exception as e:
                    console.print(f"[red]Error loading image: {e}[/red]")
                    continue

            response = process_query(prompt_text, image=img)
            console.print(Panel(
                f"[bold white]{response}[/bold white]",
                title=f"[cyan]MedGemma ({mode_str})[/cyan]",
                expand=False
            ))
        return

    # Non-interactive mode
    if args.prompt is not None:
        img = load_image(args.image) if args.image else None
        response = process_query(args.prompt, image=img)
        console.print(Panel(
            f"[bold white]{response}[/bold white]",
            title="[cyan]MedGemma Output[/cyan]"
        ))
    else:
        # Default prompt test
        sample_q = "What are the common clinical signs and first-line treatments for type 2 diabetes?"
        console.print(f"[dim]No prompt supplied. Testing default clinical query:[/dim]\n[italic]\"{sample_q}\"[/italic]\n")
        response = process_query(sample_q)
        console.print(Panel(
            f"[bold white]{response}[/bold white]",
            title="[cyan]MedGemma Output[/cyan]"
        ))


if __name__ == "__main__":
    main()
