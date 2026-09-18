"""CLI compatibility wrapper for the current self-trained VIGER TinyGPT."""

from __future__ import annotations

import argparse
from pathlib import Path

from python.viger_tiny_lm import train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VIGER TinyGPT from scratch")
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=[ROOT / "data" / "chat.txt"],
        help="One or more UTF-8 text corpora. The first file is used as the project corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "viger_tiny_lm.pt",
    )
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    if len(args.data) == 1:
        corpus = args.data[0]
    else:
        corpus = ROOT / "artifacts" / "combined_chat_corpus.txt"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_text(
            "\n\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in args.data
            ),
            encoding="utf-8",
        )

    train(corpus, args.output, max(100, args.steps))


if __name__ == "__main__":
    main()
