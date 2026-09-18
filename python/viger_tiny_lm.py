"""VIGER TinyGPT v5: a self-trained byte-level BPE Transformer.

V5 is the first ~5M-parameter VIGER language-model architecture.
No external text model is used. The model is trained from the local corpus.
"""

from __future__ import annotations

import hashlib
import math
import json
import os
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
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "vigeroid_6.pt"
RESUME_CHECKPOINT = ROOT / "artifacts" / "vigeroid_6_resume.pt"
TRAINING_STATUS = ROOT / "artifacts" / "training_status.json"

MODEL_VERSION = 6
MODEL_NAME = "VIGEROID 6"
MODEL_DIM = 336
MODEL_LAYERS = 7
MODEL_HEADS = 8
MODEL_BLOCK = 384
MAX_BPE_MERGES = 768

SPECIAL_TOKENS = {
    "<SYSTEM>": 256,
    "<USER>": 257,
    "<ASSISTANT>": 258,
    "<END>": 259,
}
BASE_VOCAB_SIZE = 260
SPECIAL_PATTERN = re.compile(r"(<SYSTEM>|<USER>|<ASSISTANT>|<END>)")
ROLE_TOKEN_IDS = {
    SPECIAL_TOKENS["<SYSTEM>"],
    SPECIAL_TOKENS["<USER>"],
    SPECIAL_TOKENS["<ASSISTANT>"],
}

SYSTEM_PROMPT = (
    "You are VIGER, a small local language model. "
    "You are friendly, concise, and helpful. "
    "The User is the person who sends a message. "
    "The Assistant is the program that responds. "
    "Use the conversation context to answer directly. "
    "When you are uncertain, say that you are uncertain."
)


def corpus_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TinyBPE:
    """Small byte-pair tokenizer with byte fallback and learned merges."""

    def __init__(
        self,
        merges: list[list[int]] | list[tuple[int, int]] | None = None,
    ) -> None:
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
    def train(cls, text: str, max_merges: int = MAX_BPE_MERGES) -> "TinyBPE":
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
            tokens = self._merge_once(
                tokens,
                best_pair,
                BASE_VOCAB_SIZE + int(best_rank),
            )
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
    """~5M parameter V5 architecture at the current corpus vocabulary."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = MODEL_DIM,
        layers: int = MODEL_LAYERS,
        heads: int = MODEL_HEADS,
        block: int = MODEL_BLOCK,
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
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

        # Tie input/output token embeddings to keep the model compact.
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        _, length = tokens.shape
        if length > self.block:
            raise ValueError(
                f"sequence length {length} exceeds block size {self.block}"
            )

        positions = torch.arange(length, device=tokens.device)
        x = self.token(tokens) + self.pos(positions)[None, :, :]
        causal = torch.triu(
            torch.ones(
                length,
                length,
                device=tokens.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal)
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 96,
        min_new_tokens: int = 6,
        temperature: float = 0.75,
        top_k: int = 24,
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

            # Mild repetition penalty instead of aggressively suppressing
            # familiar words; this keeps short conversation natural.
            for token_id in set(recent[-48:]):
                next_logits[0, token_id] -= 0.10

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
        text = corpus_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
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
        dim=int(package.get("dim", MODEL_DIM)),
        layers=int(package.get("layers", MODEL_LAYERS)),
        heads=int(package.get("heads", MODEL_HEADS)),
        block=int(package.get("block", MODEL_BLOCK)),
    )



def _configure_cpu() -> None:
    """Use a reasonable CPU thread count without requiring manual tuning."""
    if torch.cuda.is_available():
        return
    requested = os.environ.get("VIGER_CPU_THREADS", "").strip()
    if requested.isdigit():
        threads = max(1, int(requested))
    else:
        threads = min(8, max(1, (os.cpu_count() or 2) - 1))
    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        pass


def _write_training_status(
    *,
    phase: str,
    step: int,
    total_steps: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> None:
    TRAINING_STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "step": int(step),
        "total_steps": int(total_steps),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    TRAINING_STATUS.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _schedule_lr(step: int, steps: int, base_lr: float = 3e-4) -> float:
    warmup = min(250, max(1, steps // 10))
    if step < warmup:
        return base_lr * (step + 1) / warmup

    progress = (step - warmup) / max(1, steps - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (0.10 + 0.90 * cosine)


def _sample_batch(
    ids: torch.Tensor,
    block: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(ids) <= block + 1:
        raise ValueError("Not enough tokens for the requested context length.")
    starts = torch.randint(0, len(ids) - block - 1, (batch_size,))
    x = torch.stack([ids[int(s) : int(s) + block] for s in starts]).to(device)
    y = torch.stack([ids[int(s) + 1 : int(s) + block + 1] for s in starts]).to(device)
    return x, y


@torch.no_grad()
def _estimate_loss(
    model: TinyGPT,
    ids: torch.Tensor,
    block: int,
    batch_size: int,
    device: str,
    batches: int = 4,
) -> float:
    if len(ids) <= block + 1:
        return float("nan")
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        x, y = _sample_batch(ids, block, batch_size, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)



def train(
    corpus_path: Path = DEFAULT_CORPUS,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    steps: int = 5000,
    resume: bool = True,
    checkpoint_every: int = 500,
) -> Path:
    random.seed(42)
    torch.manual_seed(42)
    _configure_cpu()

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.set_float32_matmul_precision("high")

    from python.build_english_quality_corpus import ensure_corpus

    ensure_corpus()

    corpus = _load_corpus(corpus_path)
    current_hash = corpus_hash(corpus)
    tokenizer = TinyBPE.train(corpus, max_merges=MAX_BPE_MERGES)
    ids = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    if len(ids) < MODEL_BLOCK + 2:
        raise ValueError("The corpus is too small. Add more English text.")

    split = min(
        max(int(len(ids) * 0.90), MODEL_BLOCK + 2),
        len(ids) - MODEL_BLOCK - 2,
    )
    train_ids = ids[:split]
    val_ids = ids[split:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8 if device == "cuda" else 4

    start_step = 0
    model = _make_model(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.02,
    )

    resumed = False
    if resume and RESUME_CHECKPOINT.exists():
        try:
            package = torch.load(
                RESUME_CHECKPOINT,
                map_location=device,
                weights_only=True,
            )
            compatible = (
                int(package.get("model_version", -1)) == MODEL_VERSION
                and package.get("corpus_hash") == current_hash
                and int(package.get("vocab_size", -1)) == tokenizer.vocab_size
                and int(package.get("dim", -1)) == MODEL_DIM
                and int(package.get("layers", -1)) == MODEL_LAYERS
                and int(package.get("heads", -1)) == MODEL_HEADS
                and int(package.get("block", -1)) == MODEL_BLOCK
            )
            if compatible:
                model = _make_model(tokenizer.vocab_size, package).to(device)
                model.load_state_dict(package["state_dict"])
                optimizer.load_state_dict(package["optimizer_state"])
                start_step = min(int(package.get("step", 0)), steps)
                if package.get("python_random_state") is not None:
                    random.setstate(package["python_random_state"])
                if package.get("torch_rng_state") is not None:
                    torch.set_rng_state(package["torch_rng_state"])
                resumed = start_step > 0
                if resumed:
                    print(
                        f"[VIGER TinyGPT] resuming from step {start_step}/{steps} "
                        f"using {RESUME_CHECKPOINT.name}"
                    )
        except Exception as exc:
            print(f"[VIGER TinyGPT] recovery state ignored: {exc}")

    # V5 was already capable of producing a final checkpoint before recovery
    # checkpoints were introduced. If such a compatible final model exists,
    # continue from its learned weights rather than throwing away a long run.
    if resume and not resumed and checkpoint.exists():
        try:
            package = torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
            compatible = (
                int(package.get("model_version", -1)) == MODEL_VERSION
                and package.get("corpus_hash") == current_hash
                and int(package.get("vocab_size", -1)) == tokenizer.vocab_size
                and int(package.get("dim", -1)) == MODEL_DIM
                and int(package.get("layers", -1)) == MODEL_LAYERS
                and int(package.get("heads", -1)) == MODEL_HEADS
                and int(package.get("block", -1)) == MODEL_BLOCK
            )
            trained_steps = int(package.get("training_steps", 0))
            if compatible and trained_steps > 0:
                model = _make_model(tokenizer.vocab_size, package).to(device)
                model.load_state_dict(package["state_dict"])
                start_step = min(trained_steps, steps)
                resumed = start_step > 0
                if resumed:
                    print(
                        f"[VIGER TinyGPT] continuing from final checkpoint "
                        f"at step {start_step}/{steps}; optimizer starts fresh"
                    )
        except Exception as exc:
            print(f"[VIGER TinyGPT] final checkpoint resume ignored: {exc}")

    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"[VIGER TinyGPT v{MODEL_VERSION}] "
        f"{'resuming' if resumed else 'training from scratch'} on {device} "
        f"(params={parameter_count:,}, vocab={tokenizer.vocab_size}, "
        f"tokens={len(ids)}, context={MODEL_BLOCK})..."
    )
    _write_training_status(
        phase="training",
        step=start_step,
        total_steps=steps,
    )

    best_val = float("inf")
    model.train()
    for step in range(start_step, steps):
        _set_lr(optimizer, _schedule_lr(step, steps))
        x, y = _sample_batch(train_ids, MODEL_BLOCK, batch_size, device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        current_step = step + 1
        if current_step % checkpoint_every == 0 or current_step == steps:
            train_loss = float(loss.item())
            val_loss = _estimate_loss(
                model,
                val_ids,
                MODEL_BLOCK,
                batch_size,
                device,
            )
            best_val = min(best_val, val_loss)

            resume_package = {
                "model_version": MODEL_VERSION,
                "corpus_hash": current_hash,
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "tokenizer_merges": tokenizer.to_state(),
                "vocab_size": tokenizer.vocab_size,
                "dim": model.dim,
                "layers": model.layers,
                "heads": model.heads,
                "block": model.block,
                "parameter_count": parameter_count,
                "step": current_step,
                "python_random_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
            }
            RESUME_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(resume_package, RESUME_CHECKPOINT)
            _write_training_status(
                phase="training",
                step=current_step,
                total_steps=steps,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            print(
                f"[VIGER TinyGPT] step {current_step}/{steps} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "corpus_hash": current_hash,
            "state_dict": model.state_dict(),
            "tokenizer_merges": tokenizer.to_state(),
            "vocab_size": tokenizer.vocab_size,
            "dim": model.dim,
            "layers": model.layers,
            "heads": model.heads,
            "block": model.block,
            "parameter_count": parameter_count,
            "training_steps": steps,
            "best_val_loss": None if best_val == float("inf") else best_val,
        },
        checkpoint,
    )

    # The final checkpoint is complete; the resumable state is only needed for
    # interrupted runs and can safely remain as a recovery point.
    print(f"[VIGER TinyGPT] saved {checkpoint}")
    print(f"[VIGER TinyGPT] parameters={parameter_count:,}")
    _write_training_status(
        phase="complete",
        step=steps,
        total_steps=steps,
        val_loss=None if best_val == float("inf") else best_val,
    )
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
            package = torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
            if int(package.get("model_version", -1)) != MODEL_VERSION:
                raise ValueError("old model version")
            if package.get("corpus_hash") != current_hash:
                raise ValueError(
                    "training corpus changed; run /train only when you want "
                    "the current data to become new weights"
                )

            tokenizer = TinyBPE(package["tokenizer_merges"])
            model = _make_model(tokenizer.vocab_size, package).to(device)
            model.load_state_dict(package["state_dict"])
            model.eval()

            parameter_count = sum(p.numel() for p in model.parameters())
            print(
                f"[VIGER TinyGPT v{MODEL_VERSION}] loaded learned weights "
                f"(params={parameter_count:,}, vocab={tokenizer.vocab_size})"
            )
            return model, tokenizer
        except Exception as exc:
            print(f"[VIGER TinyGPT] checkpoint invalid/outdated; retraining: {exc}")

    train(corpus_path, checkpoint)
    package = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    tokenizer = TinyBPE(package["tokenizer_merges"])
    model = _make_model(tokenizer.vocab_size, package).to(device)
    model.load_state_dict(package["state_dict"])
    model.eval()
    return model, tokenizer


def answer(
    bundle: tuple[TinyGPT, TinyBPE],
    user_text: str,
    history: list[tuple[str, str]] | None = None,
    knowledge_context: str = "",
) -> str:
    model, tokenizer = bundle
    user_text = user_text.strip()
    if not user_text:
        raise ValueError("Message is empty.")

    parts = [f"<SYSTEM> {SYSTEM_PROMPT}"]
    if knowledge_context.strip():
        parts.append(
            "<SYSTEM> Local knowledge notes are optional reference material. "
            "Use them when relevant, but do not copy their labels.\n"
            f"{knowledge_context.strip()}\n<END>"
        )
    for old_user, old_assistant in (history or [])[-3:]:
        parts.append(
            f"<USER> {old_user.strip()} <ASSISTANT> {old_assistant.strip()}"
        )
    parts.append(f"<USER> {user_text} <ASSISTANT>")

    ids = tokenizer.encode(" ".join(parts))
    if not ids:
        raise ValueError("Prompt tokenization produced no tokens.")

    device = next(model.parameters()).device
    tokens = torch.tensor(
        ids[-model.block :],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    generated = model.generate(tokens)
    text = tokenizer.decode(generated, keep_special=False).strip()
    if not text:
        raise RuntimeError("TinyGPT generated an empty response.")
    return text


if __name__ == "__main__":
    print(f"{MODEL_NAME} trainer. Use /train in Telegram or import train().")
