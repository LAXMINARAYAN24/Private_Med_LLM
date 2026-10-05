"""MedGemma 4B Runner — Supports Custom Prompts (Text and Multimodal).

Usage examples:
  1. Custom text prompt:
     python scripts/run_medgemma.py --prompt "What are the early warning signs of stroke?"

  2. Custom prompt with image (URL or local file):
     python scripts/run_medgemma.py --image "https://.../sample.jpg" --prompt "Describe any findings in this image."
     python scripts/run_medgemma.py --image "my_scan.png" --prompt "Analyze this scan."

  3. Interactive mode (chat in terminal):
     python scripts/run_medgemma.py --interactive

  4. Default quick test (candy image):
     python scripts/run_medgemma.py
"""
import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import requests
import torch
from transformers import pipeline

# 1. Project-root-aligned HF cache directory (per HANDOFF.md and config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / "models" / "hf_cache" / "hub"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 2. Hugging Face Authentication Token
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN


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


def build_pipeline(adapter_path: str | Path | None = None):
    """Initializes the MedGemma multimodal pipeline with GPU / bfloat16, with optional M1 LoRA adapter."""
    cuda_available = torch.cuda.is_available()
    print("=" * 45)
    print(" MedGemma 4B Pipeline Initialization")
    print("=" * 45)
    print(f"CUDA Available:  {cuda_available}")
    if cuda_available:
        print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"bfloat16:        {torch.cuda.is_bf16_supported()}")
    else:
        print("Note: Running on CPU.")
    print(f"Cache Location:  {HF_CACHE_DIR}")
    if adapter_path:
        print(f"LoRA Adapter:    {adapter_path} (M1 Mode)")
    else:
        print("Model Variant:   M0 (Stock Base)")
    print("=" * 45 + "\n")

    print("Loading 'google/medgemma-4b-it'...")
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16 if cuda_available else torch.float32,
        device_map="auto",
        token=HF_TOKEN,
    )
    if adapter_path:
        from peft import PeftModel
        print(f"Attaching M1 fine-tuned LoRA adapter from {adapter_path} ...")
        pipe.model = PeftModel.from_pretrained(pipe.model, str(adapter_path))
        print("M1 adapter attached successfully!")

    print("Model and processor successfully loaded!\n")
    return pipe


def ask_model(pipe, prompt: str, image: Image.Image | None = None, max_new_tokens: int = 150):
    """Sends a formatted prompt (with optional image) to the pipeline."""
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

    print(f"\n[Running inference on GPU (max_tokens={max_new_tokens})...]")
    outputs = pipe(text=messages, max_new_tokens=max_new_tokens)
    return outputs[0]["generated_text"][-1]["content"]


def main():
    parser = argparse.ArgumentParser(description="Run MedGemma 4B with custom prompts")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Custom prompt question/instruction")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to local image or image URL")
    parser.add_argument("--max-tokens", "-m", type=int, default=150, help="Max new tokens to generate (default: 150)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI chat session")
    parser.add_argument("--m1", action="store_true", help="Load the M1 fine-tuned adapter (checkpoints/m1_medgemma_lora)")
    parser.add_argument("--m2", action="store_true", help="Load M2 (M1 + trained suppression gate)")
    parser.add_argument("--adapter", type=str, default=None, help="Custom path to LoRA adapter weights")
    args = parser.parse_args()

    if args.m2:
        from src.phi_defense.models import load_medgemma_m2, generate_medgemma_m2, load_medgemma_tokenizer
        tok = load_medgemma_tokenizer()
        m2 = load_medgemma_m2(load_in_4bit=True)
        if args.prompt:
            res = generate_medgemma_m2(m2, tok, args.prompt, max_new_tokens=args.max_tokens)
            print(f"\n[Model M2 (Suppression Gate Active)]")
            print(f"Prompt:   {args.prompt}")
            print(f"Response: {res}")
            return
        elif args.interactive:
            print("\n=== MedGemma 4B M2 Interactive Console (Suppression Gate Active) ===")
            print("Type your prompt (or 'quit' to exit).\n")
            while True:
                try:
                    inp = input("\nYou > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not inp or inp.lower() in ("quit", "exit"):
                    break
                res = generate_medgemma_m2(m2, tok, inp, max_new_tokens=args.max_tokens)
                print(f"M2 > {res}")
            return

    adapter_path = None

    if args.adapter:
        adapter_path = Path(args.adapter)
    elif args.m1:
        adapter_path = PROJECT_ROOT / "checkpoints" / "m1_medgemma_lora"
        if not adapter_path.exists():
            print(f"Warning: M1 adapter not found at {adapter_path}. Train it first or running as M0 base.")
            adapter_path = None

    pipe = build_pipeline(adapter_path=adapter_path)

    if args.interactive:
        print("\n=== MedGemma 4B Interactive Console ===")
        print("Type your medical prompt or question (or 'quit'/'exit' to stop).")
        print("Optional: prepend 'image:<path_or_url> | <prompt>' to include an image.\n")
        while True:
            try:
                user_input = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break

            img = None
            prompt_text = user_input
            if user_input.startswith("image:") and " | " in user_input:
                img_part, prompt_text = user_input[6:].split(" | ", 1)
                try:
                    img = load_image(img_part.strip())
                except Exception as e:
                    print(f"Error loading image: {e}")
                    continue

            response = ask_model(pipe, prompt_text, image=img, max_new_tokens=args.max_tokens)
            print(f"\nMedGemma > {response}\n" + "-" * 40)
        return

    # Non-interactive mode
    if args.prompt is not None:
        img = load_image(args.image) if args.image else None
        response = ask_model(pipe, args.prompt, image=img, max_new_tokens=args.max_tokens)
        print("\n" + "=" * 20 + " MODEL RESPONSE " + "=" * 20)
        print(response)
        print("=" * 56)
    else:
        # Default test case (candy image test)
        default_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
        default_prompt = "What animal is on the candy?"
        print(f"No prompt supplied. Running default test:")
        print(f"Image: {default_url}")
        print(f"Prompt: {default_prompt}\n")

        img = load_image(default_url)
        response = ask_model(pipe, default_prompt, image=img, max_new_tokens=40)
        print("\n" + "=" * 20 + " MODEL RESPONSE " + "=" * 20)
        print(response)
        print("=" * 56)


if __name__ == "__main__":
    main()
