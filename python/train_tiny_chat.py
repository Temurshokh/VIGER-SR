"""Train a deliberately small byte-level causal language model.

This is a research/training sandbox, not a claim of a useful general assistant.
The model learns whatever text corpus you provide and is intended to give the
chat layer a concrete local-model target without requiring a huge LLM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = 256, dim: int = 192, layers: int = 4, heads: int = 4, block: int = 256):
        super().__init__()
        self.block = block
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(block, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        _, length = tokens.shape
        if length > self.block:
            raise ValueError(f"sequence length {length} exceeds block size {self.block}")

        positions = torch.arange(length, device=tokens.device)
        x = self.token(tokens) + self.pos(positions)[None, :, :]
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        x = self.transformer(x, mask=mask)
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = tokens[:, -self.block:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


def load_corpus(paths: list[Path]) -> torch.Tensor:
    chunks = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.append(text)
    text = "\n\n".join(chunks)
    if not text:
        raise ValueError("Corpus is empty")
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiny_chat.pt"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = load_corpus(args.data)
    if len(corpus) <= args.block_size + 1:
        raise ValueError("Corpus must be larger than block size")

    model = TinyGPT(block=args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for step in range(args.steps):
        starts = torch.randint(0, len(corpus) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([corpus[s : s + args.block_size] for s in starts]).to(device)
        y = torch.stack([corpus[s + 1 : s + args.block_size + 1] for s in starts]).to(device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"step={step + 1}/{args.steps} loss={loss.item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": 256,
            "dim": 192,
            "layers": 4,
            "heads": 4,
            "block": args.block_size,
        },
        args.output,
    )
    print(f"saved={args.output}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
