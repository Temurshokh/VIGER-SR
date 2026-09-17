"""One-command launcher for the Python-only VIGER Telegram lab."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOT = ROOT / "telegram" / "bot.py"
REQ = ROOT / "requirements.txt"
TOKEN_FILE = ROOT / "telegram" / ".bot_token"


def ensure_dependencies() -> None:
    required = ("telegram", "PIL", "torch", "transformers", "pyttsx3")
    missing: list[str] = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if not missing:
        return

    print("[VIGER] Missing Python packages:", ", ".join(missing))
    answer = input("Install project requirements now? [Y/n]: ").strip().lower()
    if answer not in {"", "y", "yes"}:
        raise SystemExit("Install requirements.txt and run again.")
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
    print("[VIGER] Token saved locally. It is ignored by git.")
    return token


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ is required.")
    if not BOT.exists():
        raise SystemExit(f"Bot entrypoint not found: {BOT}")

    ensure_dependencies()
    os.environ["VIGER_TELEGRAM_TOKEN"] = read_token()
    print("[VIGER] Python-only Telegram bot starting...")
    runpy.run_path(str(BOT), run_name="__main__")


if __name__ == "__main__":
    main()
