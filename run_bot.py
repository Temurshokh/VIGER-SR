"""One-command launcher for the Python-only VIGER Telegram lab."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOT = ROOT / "telegram" / "bot.py"
REQ = ROOT / "telegram" / "requirements.txt"
TOKEN_FILE = ROOT / "telegram" / ".bot_token"


def ensure_dependencies() -> None:
    required = {
        "telegram": "python-telegram-bot",
        "PIL": "Pillow",
        "torch": "torch",
        "transformers": "transformers",
        "pyttsx3": "pyttsx3",
    }
    missing: list[str] = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if not missing:
        return

    print("[VIGER] Missing Python packages:", ", ".join(missing))
    if not REQ.exists():
        raise SystemExit(f"Requirements file not found: {REQ}")
    answer = input("Install them now? [Y/n]: ").strip().lower()
    if answer not in {"", "y", "yes"}:
        raise SystemExit("Install the requirements and run again.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQ)])


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
    print(f"[VIGER] Token saved locally to {TOKEN_FILE}")
    return token


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required.")
    if not BOT.exists():
        raise SystemExit(f"Bot entrypoint not found: {BOT}")

    ensure_dependencies()
    os.environ["VIGER_TELEGRAM_TOKEN"] = read_token()
    print("[VIGER] Starting Python-only Telegram bot...")
    runpy.run_path(str(BOT), run_name="__main__")


if __name__ == "__main__":
    main()
