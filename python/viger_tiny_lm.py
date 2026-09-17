"""TinyGPT + a self-trained byte-level BPE tokenizer.

Educational local language-model experiment. No external text model is used.
The tokenizer learns frequent byte sequences from the local corpus. Special
role tokens teach the model that a User sends a message and an Assistant answers.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "chat.txt"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "viger_tiny_lm.pt"
MODEL_VERSION = 4

SPECIAL_TOKENS = {
    "<SYSTEM>": 256,
    "<USER>": 257,
    "<ASSISTANT>": 258,
    "<END>": 259,
}
BASE_VOCAB_SIZE = 260
SPECIAL_PATTERN = re.compile(r"(<SYSTEM>|<USER>|<ASSISTANT>|<END>)")
ROLE_TOKEN_IDS = {SPECIAL_TOKENS["<SYSTEM>"], SPECIAL_TOKENS["<USER>"], SPECIAL_TOKENS["<ASSISTANT>"]}

SYSTEM_PROMPT = (
    "You are VIGER, a tiny local language model. "
    "You are brief, friendly, and helpful. "
    "User is the person who sends a message. "
    "Assistant is the program that answers the User. "
    "A word has meaning, and context helps determine meaning. "
    "Answer the user's message directly."
)


def corpus_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TinyBPE:
    """Small byte-pair tokenizer with byte fallback and learned merges."""

    def __init__(self, merges: list[list[int]] | list[tuple[int, int]] | None = None) -> None:
        self.merges: list[tuple[int, int]] = [tuple(pair) for pair in (merges or [])]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_to_id = dict(SPECIAL_TOKENS)
        self.id_to_special = {value: key for key, value in SPECIAL_TOKENS.items()}

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    @staticmethod
    def _chunks(text: str) -> Iterable[str]:
        parts = SPECIAL_PATTERN.split(text)
        return (part for part in parts if part)

    @staticmethod
    def _merge_once(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    @classmethod
    def train(cls, text: str, max_merges: int = 320) -> "TinyBPE":
        sequences: list[list[int]] = []
        for chunk in cls._chunks(text):
            if chunk in SPECIAL_TOKENS:
                continue
            raw = chunk.encode("utf-8")
            if raw:
                sequences.append(list(raw))

        merges: list[tuple[int, int]] = []
        for _ in range(max_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < 2:
                break
            new_id = BASE_VOCAB_SIZE + len(merges)
            merges.append(pair)
            sequences = [cls._merge_once(seq, pair, new_id) for seq in sequences]

        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        tokens = list(chunk.encode("utf-8"))
        while len(tokens) > 1:
            best_pair: tuple[int, int] | None = None
            best_rank: int | None = None
            for pair in zip(tokens, tokens[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break
            tokens = self._merge_once(tokens, best_pair, BASE_VOCAB_SIZE + best_rank)
        return tokens

    def encode(self, text: str) -> list[int]:
        output: list[int] = []
        for chunk in self._chunks(text):
            special_id = self.special_to_id.get(chunk)
            if special_id is not None:
                output.append(special_id)
            else:
                output.extend(self._encode_chunk(chunk))
        return output

    def _expand(self, token_id: int) -> list[int]:
        if token_id < 256:
            return [token_id]
        if token_id in self.id_to_special:
            return []
        merge_index = token_id - BASE_VOCAB_SIZE
        if not 0 <= merge_index < len(self.merges):
            raise ValueError(f"Invalid BPE token id: {token_id}")
        left, right = self.merges[merge_index]
        return self._expand(left) + self._expand(right)

    def decode(self, tokens: Iterable[int], keep_special: bool = True) -> str:
        pieces: list[str] = []
        buffer = bytearray()
        for token_id in tokens:
            if token_id in self.id_to_special:
                if buffer:
                    pieces.append(buffer.decode("utf-8", errors="replace"))
                    buffer.clear()
                if keep_special:
                    pieces.append(self.id_to_special[token_id])
                continue
            for byte_id in self._expand(int(token_id)):
                buffer.append(byte_id)
        if buffer:
            pieces.append(buffer.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def to_state(self) -> list[list[int]]:
        return [[int(left), int(right)] for left, right in self.merges]


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 160,
        layers: int = 4,
        heads: int = 4,
        block: int = 192,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.block = block
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(block, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        _, length = tokens.shape
        if length > self.block:
            raise ValueError(f"sequence length {length} exceeds block size {self.block}")
        positions = torch.arange(length, device=tokens.device)
        x = self.token(tokens) + self.pos(positions)[None, :, :]
        causal = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        x = self.transformer(x, mask=causal)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 72,
        min_new_tokens: int = 6,
        temperature: float = 0.72,
        top_k: int = 18,
    ) -> list[int]:
        self.eval()
        result = tokens.clone()
        recent: list[int] = []
        end_id = SPECIAL_TOKENS["<END>"]

        for step in range(max_new_tokens):
            context = result[:, -self.block :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 0.05)

            for token_id in ROLE_TOKEN_IDS:
                next_logits[0, token_id] = float("-inf")
            if step + 1 < min_new_tokens:
                next_logits[0, end_id] = float("-inf")

            for token_id in set(recent[-64:]):
                next_logits[0, token_id] -= 0.15

            k = min(top_k, next_logits.shape[-1])
            values, indices = torch.topk(next_logits, k)
            probs = torch.softmax(values, dim=-1)
            choice = torch.multinomial(probs, 1)
            next_token = indices.gather(-1, choice)
            token_id = int(next_token.item())
            result = torch.cat([result, next_token], dim=1)
            recent.append(token_id)

            if token_id == end_id:
                break

        return result[0, tokens.shape[1] :].tolist()


def _load_corpus(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    paths = [path]
    if path.parent.exists():
        for extra in sorted(path.parent.glob("*.txt")):
            if extra != path:
                paths.append(extra)

    chunks: list[str] = []
    for corpus_path in paths:
        text = corpus_path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            chunks.append(text)
    text = "\n\n".join(chunks)
    if not text.strip():
        raise ValueError("Training corpus is empty")
    return text


def _make_model(vocab_size: int, package: dict | None = None) -> TinyGPT:
    package = package or {}
    return TinyGPT(
        vocab_size=vocab_size,
        dim=int(package.get("dim", 160)),
        layers=int(package.get("layers", 4)),
        heads=int(package.get("heads", 4)),
        block=int(package.get("block", 192)),
    )


def train(
    corpus_path: Path = DEFAULT_CORPUS,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    steps: int = 3500,
) -> Path:
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.set_float32_matmul_precision("high")

    corpus = _load_corpus(corpus_path)
    tokenizer = TinyBPE.train(corpus, max_merges=320)
    ids = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    block = min(192, len(ids) - 2)
    if block < 24:
        raise ValueError("The chat corpus is too small. Add more English examples.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _make_model(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
    batch_size = 12 if device == "cuda" else 8

    print(
        f"[VIGER TinyGPT v{MODEL_VERSION}] training from scratch on {device} "
        f"(vocab={tokenizer.vocab_size}, tokens={len(ids)})..."
    )
    model.train()
    for step in range(steps):
        starts = torch.randint(0, len(ids) - block - 1, (batch_size,))
        x = torch.stack([ids[int(s) : int(s) + block] for s in starts]).to(device)
        y = torch.stack([ids[int(s) + 1 : int(s) + block + 1] for s in starts]).to(device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 250 == 0 or step == 0:
            print(f"[VIGER TinyGPT] step {step + 1}/{steps} loss={loss.item():.4f}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "corpus_hash": corpus_hash(corpus),
            "state_dict": model.state_dict(),
            "tokenizer_merges": tokenizer.to_state(),
            "vocab_size": tokenizer.vocab_size,
            "dim": model.dim,
            "layers": model.layers,
            "heads": model.heads,
            "block": model.block,
        },
        checkpoint,
    )
    print(f"[VIGER TinyGPT] saved {checkpoint}")
    print(f"[VIGER TinyGPT] parameters={sum(p.numel() for p in model.parameters()):,}")
    return checkpoint


def load_or_train(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    corpus_path: Path = DEFAULT_CORPUS,
) -> tuple[TinyGPT, TinyBPE]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = _load_corpus(corpus_path)
    current_hash = corpus_hash(corpus)

    if checkpoint.exists():
        try:
            package = torch.load(checkpoint, map_location=device, weights_only=True)
            if int(package.get("model_version", -1)) != MODEL_VERSION:
                raise ValueError("old model version")
            if package.get("corpus_hash") != current_hash:
                raise ValueError("training corpus changed")
            tokenizer = TinyBPE(package["tokenizer_merges"])
            model = _make_model(tokenizer.vocab_size, package).to(device)
            model.load_state_dict(package["state_dict"])
            model.eval()
            print(
                f"[VIGER TinyGPT v{MODEL_VERSION}] loaded learned weights "
                f"(vocab={tokenizer.vocab_size})"
            )
            return model, tokenizer
        except Exception as exc:
            print(f"[VIGER TinyGPT] checkpoint invalid/outdated; retraining: {exc}")

    train(corpus_path, checkpoint)
    package = torch.load(checkpoint, map_location=device, weights_only=True)
    tokenizer = TinyBPE(package["tokenizer_merges"])
    model = _make_model(tokenizer.vocab_size, package).to(device)
    model.load_state_dict(package["state_dict"])
    model.eval()
    return model, tokenizer


def answer(
    bundle: tuple[TinyGPT, TinyBPE],
    user_text: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    model, tokenizer = bundle
    user_text = user_text.strip()
    if not user_text:
        raise ValueError("Message is empty.")

    parts = [f"<SYSTEM> {SYSTEM_PROMPT}"]
    for old_user, old_assistant in (history or [])[-3:]:
        parts.append(f"<USER> {old_user.strip()} <ASSISTANT> {old_assistant.strip()}")
    parts.append(f"<USER> {user_text} <ASSISTANT>")
    ids = tokenizer.encode(" ".join(parts))
    if not ids:
        raise ValueError("Prompt tokenization produced no tokens.")

    device = next(model.parameters()).device
    tokens = torch.tensor(ids[-model.block :], dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(tokens)
    text = tokenizer.decode(generated, keep_special=False).strip()
    if not text:
        raise RuntimeError("TinyGPT generated an empty response.")
    return text


if __name__ == "__main__":
    print("VIGER TinyGPT trainer. Use /train in Telegram or import train().")
