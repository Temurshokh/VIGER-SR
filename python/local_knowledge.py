"""Offline local knowledge retrieval for VIGEROID 6."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_FILE = ROOT / "data" / "knowledge_base.txt"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "that",
    "the", "this", "to", "what", "when", "where", "which", "who", "why",
    "with", "you", "your", "tell", "about", "does", "did", "was", "were",
}

FACTUAL_START = re.compile(
    r"^(what|why|how|who|where|when|which|explain|tell me|define|"
    r"what's|whats|is|are|does|do|did|can|could)\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())
    return {word for word in words if word not in STOPWORDS}


def _load_blocks() -> list[str]:
    if not KNOWLEDGE_FILE.exists():
        return []
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text)
    return [block.strip() for block in blocks if block.strip()]


def retrieve(query: str, limit: int = 2) -> list[str]:
    """Return relevant local facts for factual-looking questions."""
    query = query.strip()
    if not query or not FACTUAL_START.match(query):
        return []

    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    scored: list[tuple[int, int, str]] = []
    for index, block in enumerate(_load_blocks()):
        block_tokens = _tokens(block)
        overlap = len(query_tokens & block_tokens)
        if overlap == 0:
            continue
        score = overlap * 10
        score += sum(
            1 for token in query_tokens
            if token in block_tokens and len(token) >= 7
        )
        scored.append((score, -len(block), block))

    scored.sort(reverse=True)
    return [block for _, _, block in scored[: max(1, limit)]]


def format_context(query: str, limit: int = 2) -> str:
    passages = retrieve(query, limit)
    if not passages:
        return ""
    return "\n".join(
        f"[Local knowledge {index}] {passage}"
        for index, passage in enumerate(passages, start=1)
    )
