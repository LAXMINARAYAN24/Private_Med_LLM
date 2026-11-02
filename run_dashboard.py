"""MedShield AI — Privacy-Preserving Medical AI & Defense System Launcher.

Starts the full-stack FastAPI server + Web Dashboard and opens the browser.
Runs on NVIDIA GeForce RTX 4050 (4-bit NF4 pure GPU VRAM).

Usage:
    python run_dashboard.py
    python run_dashboard.py --port 8000
    python run_dashboard.py --no-browser
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path

# Auto-delegate to virtual environment python if available
ROOT = Path(__file__).resolve().parent
venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    import subprocess
    sys.exit(subprocess.call([str(venv_python)] + sys.argv))

sys.path.insert(0, str(ROOT))

# Load .env
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# HF Cache
HF_CACHE_DIR = ROOT / "models" / "hf_cache" / "hub"
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))

import torch
import uvicorn
from rich.console import Console
from rich.panel import Panel

console = Console()


def print_banner(host: str, port: int):
    cuda_avail = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if cuda_avail
        else 0.0
    )

    url = f"http://{host}:{port}" if host != "0.0.0.0" else f"http://localhost:{port}"

    console.print(Panel.fit(
        f"[bold cyan]MedShield AI — Clinical Assistant & Privacy Defense Dashboard[/bold cyan]\n"
        f"[bold green]Full-Stack Application Online[/bold green]\n\n"
        f"• [bold]Local URL:[/bold]      [link={url}]{url}[/link]\n"
        f"• [bold]API Docs:[/bold]       [link={url}/docs]{url}/docs[/link]\n"
        f"• [bold]Compute Target:[/bold] [yellow]{device}[/yellow] (VRAM: [cyan]{vram:.2f} GB[/cyan])\n"
        f"• [bold]Model Base:[/bold]     [magenta]google/medgemma-4b-it[/magenta] (4-bit NF4 Quantized)\n"
        f"• [bold]Defense Stack:[/bold]  [bold green]Tier 1 (MiniLM + 9-Expert MoPE) + Tier 2 In-Decoder Gate[/bold green]\n"
        f"• [bold]Documentation:[/bold]  [dim]docs/PRD.md · AGENTS.md · DESIGN_SYSTEM.md · ARCHITECTURE.md[/dim]",
        title="[bold green]🛡️ MedShield AI Server Ready[/bold green]",
        border_style="cyan"
    ))


def prewarm_models():
    """Initializes Tier 1 defense runner into memory for fast interactive response."""
    console.print("[dim]Pre-warming Tier-1 Dual-Encoder Guardrail & MoPE router...[/dim]")
    try:
        from src.dashboard.app import get_defense_runner
        get_defense_runner()
        console.print("[bold green]Tier-1 Defense Engine warmed and ready![/bold green]")
    except Exception as e:
        console.print(f"[yellow]Warning pre-warming defense runner: {e}[/yellow]")


def is_port_in_use(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main():
    parser = argparse.ArgumentParser(description="Launch MedShield AI Full-Stack Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload for development")
    parser.add_argument("--kill-existing", action="store_true", help="Automatically kill any existing process holding the port")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}" if args.host != "0.0.0.0" else f"http://localhost:{args.port}"

    # Check if port is already active
    if is_port_in_use(args.host, args.port):
        if args.kill_existing:
            console.print(f"[yellow]Port {args.port} is in use. Killing existing process...[/yellow]")
            import subprocess
            subprocess.run(
                ["powershell", "-Command", f"(Get-NetTCPConnection -LocalPort {args.port} -ErrorAction SilentlyContinue).OwningProcess | ForEach-Object {{ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }}"],
                capture_output=True,
            )
            time.sleep(1.0)
        else:
            # Check if it's already MedShield running
            import urllib.request
            is_medshield = False
            try:
                with urllib.request.urlopen(f"{url}/api/system", timeout=1.0) as resp:
                    if resp.status == 200:
                        is_medshield = True
            except Exception:
                pass

            if is_medshield:
                console.print(Panel.fit(
                    f"[bold green]MedShield AI server is ALREADY ONLINE and running on {url}![/bold green]\n\n"
                    f"• [bold]Local URL:[/bold]      [link={url}]{url}[/link]\n"
                    f"• [bold]API Docs:[/bold]       [link={url}/docs]{url}/docs[/link]\n"
                    f"• [bold]Status:[/bold]         Active (GPU & Tier-1 Defense Ready)\n\n"
                    f"[dim]To restart fresh, run: python run_dashboard.py --kill-existing[/dim]",
                    title="[bold green]🛡️ MedShield AI Server Active[/bold green]",
                    border_style="green"
                ))
                if not args.no_browser:
                    webbrowser.open(url)
                return
            else:
                # Find next free port
                orig_port = args.port
                while is_port_in_use(args.host, args.port):
                    args.port += 1
                console.print(f"[yellow]Port {orig_port} is busy. Automatically switching to port {args.port}...[/yellow]")
                url = f"http://{args.host}:{args.port}"

    prewarm_models()
    print_banner(args.host, args.port)
    if not args.no_browser:
        import threading
        def open_browser():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    if args.reload:
        uvicorn.run("src.dashboard.app:app", host=args.host, port=args.port, reload=True, log_level="warning")
    else:
        from src.dashboard.app import app
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

