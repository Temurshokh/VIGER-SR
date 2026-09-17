"""Run the experimental TinyGPT checkpoint locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_tiny_chat import TinyGPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", type=str, default="Hello, ")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.tokens <= 0:
        raise ValueError("tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TinyGPT(
        vocab_size=checkpoint.get("vocab_size", 256),
        dim=checkpoint.get("dim", 192),
        layers=checkpoint.get("layers", 4),
        heads=checkpoint.get("heads", 4),
        block=checkpoint.get("block", 256),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    prompt_bytes = torch.tensor(list(args.prompt.encode("utf-8")), dtype=torch.long, device=device)
    prompt = prompt_bytes.unsqueeze(0)
    output = model.generate(prompt, max_new_tokens=args.tokens, temperature=args.temperature)[0]
    text = bytes(output.tolist()).decode("utf-8", errors="replace")

    print(text)


if __name__ == "__main__":
    main()
