"""Inspect the local VIGEROID training corpus."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from python.viger_tiny_lm import TinyBPE, _load_corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "data"


def report() -> dict:
    paths = sorted(CORPUS_ROOT.glob("*.txt"))
    corpus = _load_corpus(ROOT / "data" / "chat.txt")

    tokenizer = TinyBPE.train(corpus, max_merges=768)
    token_ids = tokenizer.encode(corpus)
    blocks = [part.strip() for part in corpus.split("\n\n") if part.strip()]
    duplicates = len(blocks) - len(set(blocks))
    duplicate_ratio = duplicates / max(1, len(blocks))

    files = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        files.append(
            {
                "name": path.name,
                "chars": len(text),
                "lines": len(text.splitlines()),
            }
        )

    return {
        "files": files,
        "characters": len(corpus),
        "tokens": len(token_ids),
        "vocab": tokenizer.vocab_size,
        "blocks": len(blocks),
        "duplicate_blocks": duplicates,
        "duplicate_ratio": duplicate_ratio,
    }


if __name__ == "__main__":
    stats = report()
    print("VIGEROID corpus")
    print(f"Characters: {stats['characters']:,}")
    print(f"BPE tokens: {stats['tokens']:,}")
    print(f"Vocabulary: {stats['vocab']}")
    print(f"Blocks: {stats['blocks']:,}")
    print(
        f"Duplicate blocks: {stats['duplicate_blocks']:,} "
        f"({stats['duplicate_ratio']:.2%})"
    )
    print("")
    for item in stats["files"]:
        print(f"- {item['name']}: {item['chars']:,} chars")
