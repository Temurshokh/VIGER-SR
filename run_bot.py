"""One-command launcher for the Python-only VIGER Telegram lab."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOT = ROOT / "telegram_bot" / "bot.py"
REQ = ROOT / "requirements.txt"
TOKEN_FILE = ROOT / ".bot_token"


def venv_python() -> Path:
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def run_python(python: Path, *args: str) -> None:
    subprocess.check_call([str(python), *args])


def verify_ai_stack(python: Path) -> None:
    probe = (
        "import torch, torchvision, transformers; "
        "from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution; "
        "import PIL, telegram"
    )
    try:
        run_python(python, "-c", probe)
        return
    except subprocess.CalledProcessError:
        print("[VIGER] AI package check failed; repairing the PyTorch/vision stack...")

    run_python(
        python,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "torch",
        "torchvision",
    )
    run_python(python, "-c", probe)


def ensure_venv() -> Path:
    python = venv_python()
    if not python.exists():
        print("[VIGER] Creating local Python environment...")
        run_python(sys.executable, "-m", "venv", str(ROOT / ".venv"))

    if not REQ.exists():
        raise SystemExit(f"Requirements file not found: {REQ}")

    print("[VIGER] Installing/checking Python dependencies...")
    run_python(python, "-m", "pip", "install", "--upgrade", "-r", str(REQ))
    verify_ai_stack(python)
    return python


def read_token() -> str:
    token = os.environ.get("VIGER_TELEGRAM_TOKEN", "").strip()
    if token:
        return token

    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token

    token = input("Telegram bot token (from @BotFather): ").strip()
    if not token:
        raise SystemExit("Telegram bot token is required.")

    TOKEN_FILE.write_text(token, encoding="utf-8")
    print("[VIGER] Token saved locally in .bot_token")
    return token


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required.")
    if not BOT.exists():
        raise SystemExit(f"Bot entrypoint not found: {BOT}")

    python = ensure_venv()
    env = os.environ.copy()
    env["VIGER_TELEGRAM_TOKEN"] = read_token()

    print("[VIGER] Starting Telegram bot...")
    print("[VIGER] Python-only mode: no Visual Studio, no C++, no .exe.")
    completed = subprocess.run([str(python), str(BOT)], cwd=str(ROOT), env=env)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
