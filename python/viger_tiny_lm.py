"""A tiny character-level language model trained from scratch.

The model is intentionally small: it exists to make the learning loop visible and
experimental. Responses come from learned neural weights. There is no fallback
response and there is no external model server.
"""

from __future__ import annotations

import random
import string
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "chat.txt"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "viger_tiny_lm.pt"

# Keep the vocabulary small while allowing both English and Russian experiments.
CYRILLIC = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
BASE_VOCAB = string.ascii_letters + string.digits + string.punctuation + " \n\t" + CYRILLIC


class VigerTinyLM(nn.Module):
    def __init__(self, vocab_size: int, embedding: int = 48, hidden: int = 72) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding
        self.hidden_size = hidden
        self.embed = nn.Embedding(vocab_size, embedding)
        self.gru = nn.GRU(embedding, hidden, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor | None = None):
        x = self.embed(tokens)
        x, hidden = self.gru(x, hidden)
        return self.head(self.norm(x)), hidden

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        encode: dict[str, int],
        decode: list[str],
        max_new_tokens: int = 96,
        temperature: float = 0.35,
        top_k: int = 8,
    ) -> str:
        self.eval()
        device = next(self.parameters()).device
        safe_prompt = "".join(ch if ch in encode else " " for ch in prompt)
        tokens = torch.tensor(
            [encode[ch] for ch in safe_prompt], dtype=torch.long, device=device
        ).unsqueeze(0)
        if tokens.numel() == 0:
            raise ValueError("The prompt is empty.")

        _, hidden = self(tokens)
        last = tokens[:, -1:]
        generated: list[str] = []

        for _ in range(max_new_tokens):
            logits, hidden = self(last, hidden)
            logits = logits[:, -1, :] / max(temperature, 0.05)
            if top_k > 0 and top_k < logits.shape[-1]:
                values, indices = torch.topk(logits, top_k)
                probs = torch.softmax(values, dim=-1)
                choice = torch.multinomial(probs, 1)
                next_token = indices.gather(-1, choice)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            value = int(next_token.item())
            char = decode[value]
            if char == "\n":
                break
            generated.append(char)
            last = next_token

        text = "".join(generated).strip()
        if not text:
            raise RuntimeError("TinyLM generated an empty response.")
        return text


def _load_corpus(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError(f"Empty corpus: {path}")
    return text


def _vocabulary(corpus: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(BASE_VOCAB + corpus))
    encode = {char: index for index, char in enumerate(chars)}
    return encode, chars


def train(
    corpus_path: Path = DEFAULT_CORPUS,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    steps: int = 700,
) -> Path:
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    corpus = _load_corpus(corpus_path)
    encode, decode = _vocabulary(corpus)
    ids = torch.tensor([encode.get(ch, encode[" "]) for ch in corpus], dtype=torch.long)

    block = min(96, len(ids) - 1)
    if block < 12:
        raise ValueError("The chat corpus is too small. Add more examples to data/chat.txt.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VigerTinyLM(len(decode)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    batch_size = 24

    print(f"[VIGER TinyLM] training from scratch on {device} ({len(decode)} chars)...")
    for step in range(steps):
        starts = torch.randint(0, len(ids) - block - 1, (batch_size,))
        x = torch.stack([ids[int(s):int(s) + block] for s in starts]).to(device)
        y = torch.stack([ids[int(s) + 1:int(s) + block + 1] for s in starts]).to(device)

        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(decode)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 100 == 0 or step == 0:
            print(f"[VIGER TinyLM] step {step + 1}/{steps} loss={loss.item():.4f}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": decode,
            "embedding": model.embedding_size,
            "hidden": model.hidden_size,
        },
        checkpoint,
    )
    return checkpoint


def load_or_train(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    corpus_path: Path = DEFAULT_CORPUS,
) -> VigerTinyLM:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint.exists():
        try:
            package = torch.load(checkpoint, map_location=device)
            decode = package["vocab"]
            model = VigerTinyLM(
                vocab_size=len(decode),
                embedding=int(package.get("embedding", 48)),
                hidden=int(package.get("hidden", 72)),
            ).to(device)
            model.load_state_dict(package["state_dict"])
            model.eval()
            print(f"[VIGER TinyLM] loaded learned weights from {checkpoint}")
            return model
        except Exception as exc:
            print(f"[VIGER TinyLM] old checkpoint is incompatible; retraining: {exc}")

    train(corpus_path, checkpoint)
    package = torch.load(checkpoint, map_location=device)
    decode = package["vocab"]
    model = VigerTinyLM(
        vocab_size=len(decode),
        embedding=int(package.get("embedding", 48)),
        hidden=int(package.get("hidden", 72)),
    ).to(device)
    model.load_state_dict(package["state_dict"])
    model.eval()
    return model


def answer(model: VigerTinyLM, user_text: str) -> str:
    user_text = user_text.strip()
    if not user_text:
        raise ValueError("Message is empty.")

    checkpoint_model = model
    # The vocabulary is stored in the checkpoint as decode IDs. Build the inverse map.
    # This keeps inference independent from the training script and uses only learned weights.
    # Accessing the vocabulary from the model is made explicit by attaching it below.
    decode = getattr(checkpoint_model, "viger_decode", None)
    encode = getattr(checkpoint_model, "viger_encode", None)
    if decode is None or encode is None:
        raise RuntimeError("TinyLM vocabulary is not attached to the loaded model.")

    prompt = f"User: {user_text}\nAssistant:"
    return model.generate(prompt, encode, decode)
