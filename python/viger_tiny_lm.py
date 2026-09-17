"""Tiny language model trained from scratch for the VIGER Telegram experiment.

This is intentionally small and educational. It generates responses from learned
weights; there is no hard-coded fallback response and no local model server.
"""

from __future__ import annotations

from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "chat.txt"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "viger_tiny_lm.pt"


class VigerTinyLM(nn.Module):
    def __init__(self, vocab_size: int = 256, embedding: int = 64, hidden: int = 96) -> None:
        super().__init__()
        self.embedding_size = embedding
        self.hidden_size = hidden
        self.embed = nn.Embedding(vocab_size, embedding)
        self.gru = nn.GRU(embedding, hidden, num_layers=2, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor | None = None):
        x = self.embed(tokens)
        x, hidden = self.gru(x, hidden)
        logits = self.head(self.norm(x))
        return logits, hidden

    @torch.no_grad()
    def generate(self, prompt: bytes, max_new_tokens: int = 96, temperature: float = 0.65) -> bytes:
        self.eval()
        device = next(self.parameters()).device
        tokens = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)
        _, hidden = self(tokens)
        generated: list[int] = []
        last = tokens[:, -1:]
        for _ in range(max_new_tokens):
            logits, hidden = self(last, hidden)
            probs = torch.softmax(logits[:, -1, :] / max(temperature, 0.05), dim=-1)
            next_token = torch.multinomial(probs, 1)
            value = int(next_token.item())
            generated.append(value)
            last = next_token
            if generated[-1:] == [10] and len(generated) > 8:  # newline
                break
        return bytes(generated)


def _load_corpus(path: Path) -> torch.Tensor:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError(f"Empty corpus: {path}")
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def train(corpus_path: Path = DEFAULT_CORPUS, checkpoint: Path = DEFAULT_CHECKPOINT, steps: int = 900) -> Path:
    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = _load_corpus(corpus_path)
    block = 96
    if len(data) <= block + 1:
        raise ValueError("The chat corpus is too small.")

    model = VigerTinyLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for step in range(steps):
        starts = torch.randint(0, len(data) - block - 1, (32,))
        x = torch.stack([data[int(s):int(s) + block] for s in starts]).to(device)
        y = torch.stack([data[int(s) + 1:int(s) + block + 1] for s in starts]).to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 150 == 0:
            print(f"[VIGER TinyLM] step {step + 1}/{steps} loss={loss.item():.4f}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": 256,
            "embedding": model.embedding_size,
            "hidden": model.hidden_size,
        },
        checkpoint,
    )
    return checkpoint


def load_or_train(checkpoint: Path = DEFAULT_CHECKPOINT, corpus_path: Path = DEFAULT_CORPUS) -> VigerTinyLM:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not checkpoint.exists():
        print("[VIGER TinyLM] no checkpoint; training from scratch...")
        train(corpus_path, checkpoint)
    package = torch.load(checkpoint, map_location=device)
    model = VigerTinyLM(
        vocab_size=package.get("vocab_size", 256),
        embedding=package.get("embedding", 64),
        hidden=package.get("hidden", 96),
    ).to(device)
    model.load_state_dict(package["state_dict"])
    model.eval()
    return model


def answer(model: VigerTinyLM, user_text: str) -> str:
    prompt = f"User: {user_text.strip()}\nAssistant:".encode("utf-8")
    raw = model.generate(prompt, max_new_tokens=96, temperature=0.55)
    text = raw.decode("utf-8", errors="ignore").strip()
    text = text.split("\n", 1)[0].strip()
    if not text:
        return "…"
    return text
