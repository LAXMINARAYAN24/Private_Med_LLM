"""
MedShield AI — One-Click Cloud Restoration Script.

Downloads and restores all trained defense checkpoints, MoPE routers, in-decoder
attention suppression gates, and MedGemma LoRA adapters from the private Hugging Face
repository (Laxminarayan24/medshield-checkpoints) into the local workspace.

Optionally downloads the full raw MIMIC-III and MediRed+ datasets from the private
dataset repository (Laxminarayan24/medshield-data) when called with --include-data.

Usage:
    python restore_checkpoints.py                  # Restores trained models (~1.68 GB, ready to run)
    python restore_checkpoints.py --include-data   # Also downloads raw MIMIC & MediRed data (~12.1 GB)
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ID_MODELS = "Laxminarayan24/medshield-checkpoints"
REPO_ID_DATA = "Laxminarayan24/medshield-data"
ROOT = Path(__file__).resolve().parent

REQUIRED_CHECKPOINTS = [
    ROOT / "checkpoints" / "guardrail_encoder_plus" / "model.safetensors",
    ROOT / "checkpoints" / "guardrail_encoder_plus" / "prototypes.pt",
    ROOT / "checkpoints" / "mope_plus" / "mope_plus.pt",
    ROOT / "checkpoints" / "m2_gate" / "gate.pt",
    ROOT / "m1_medgemma_lora" / "adapter_model.safetensors",
]

def check_dependencies():
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("[!] 'huggingface_hub' is required. Installing via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        return True

def restore(include_data=False):
    print("=" * 75)
    print(" 🛡️  MedShield AI — One-Click Cloud Restoration Tool")
    print(f" • Checkpoints Repo: https://huggingface.co/{REPO_ID_MODELS} (Private)")
    if include_data:
        print(f" • Datasets Repo:    https://huggingface.co/datasets/{REPO_ID_DATA} (Private)")
    print("=" * 75)

    check_dependencies()
    from huggingface_hub import snapshot_download, HfApi

    # Load HF token if available in .env or environment
    env_file = ROOT / ".env"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN=") and not line.strip().startswith("#"):
                token = line.split("=", 1)[1].strip()
                os.environ["HF_TOKEN"] = token
                break

    print("\n[1/3] Authenticating with Hugging Face...")
    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        print(f"      Authenticated as: {user_info.get('name')} ({user_info.get('type')})")
    except Exception as e:
        print(f"\n[!] Hugging Face authentication failed: {e}")
        print("    Please log in using: huggingface-cli login")
        print("    or set your token in .env: HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    print(f"\n[2/3] Downloading trained checkpoints into workspace: {ROOT}")
    print("      This downloads ~1.68 GB of trained model weights (no retraining needed)...")
    try:
        snapshot_download(
            repo_id=REPO_ID_MODELS,
            repo_type="model",
            local_dir=str(ROOT),
            token=token,
            ignore_patterns=["README.md", ".git*"],
            resume_download=True
        )
        print("      Checkpoints downloaded successfully!")
    except Exception as e:
        print(f"\n[!] Error during checkpoint download: {e}")
        sys.exit(1)

    if include_data:
        print(f"\n[2b/3] Downloading raw datasets (~12.1 GB: MIMIC-III & MediRed+)...")
        try:
            snapshot_download(
                repo_id=REPO_ID_DATA,
                repo_type="dataset",
                local_dir=str(ROOT),
                token=token,
                ignore_patterns=["README.md", ".git*"],
                resume_download=True
            )
            print("       Datasets downloaded successfully!")
        except Exception as e:
            print(f"\n[!] Error during dataset download: {e}")
            sys.exit(1)

    print("\n[3/3] Validating restored checkpoint integrity...")
    all_ok = True
    for cp in REQUIRED_CHECKPOINTS:
        if cp.exists():
            size_mb = cp.stat().st_size / (1024 ** 2)
            print(f"      [OK] {cp.relative_to(ROOT)} ({size_mb:.2f} MB)")
        else:
            print(f"      [MISSING] {cp.relative_to(ROOT)}")
            all_ok = False

    print("\n" + "=" * 75)
    if all_ok:
        print(" ✅ SUCCESS: All MedShield AI checkpoints restored perfectly!")
        print(" You can now immediately launch the system with:")
        print("     python run_dashboard.py")
    else:
        print(" ⚠️  WARNING: Some optional checkpoints could not be verified.")
        print(" Check your network connection and retry: python restore_checkpoints.py")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="MedShield AI Cloud Restoration Tool")
    parser.add_argument("--include-data", action="store_true", help="Also restore raw MIMIC-III and MediRed+ datasets (~12.1 GB)")
    args = parser.parse_args()
    restore(include_data=args.include_data)

if __name__ == "__main__":
    main()
