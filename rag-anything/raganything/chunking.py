"""
Pluggable chunking strategies for RAG-Anything.

Each function matches the LightRAG chunking_func signature:
    (tokenizer, content, split_by_character, split_by_character_only,
     chunk_overlap_token_size, chunk_token_size) -> list[dict]

dict keys required: tokens (int), content (str), chunk_order_index (int)

Available strategies
--------------------
token     : Token-based sliding window (LightRAG default, 1200 tok / 100 overlap)
recursive : Hierarchical delimiter split \\n\\n → \\n → ". " → " " → token fallback
sentence  : Group complete sentences up to the token budget
paragraph : Split on blank lines; fall back to token-based for oversized paragraphs
"""

import re
from typing import Any

from lightrag.utils import Tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Token  (thin wrapper — preserves LightRAG default behaviour)
# ─────────────────────────────────────────────────────────────────────────────

def chunking_token(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """Token-based sliding window (LightRAG built-in)."""
    from lightrag.operate import chunking_by_token_size
    return chunking_by_token_size(
        tokenizer, content,
        split_by_character, split_by_character_only,
        chunk_overlap_token_size, chunk_token_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recursive  (LangChain-style hierarchical splitting)
# ─────────────────────────────────────────────────────────────────────────────

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunking_recursive(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """
    Recursive character splitting: try \\n\\n → \\n → ". " → " " in order.
    If a piece still exceeds chunk_token_size after all separators are exhausted,
    fall back to the token-based sliding window.
    """

    def _split(text: str, sep_idx: int) -> list[str]:
        if not text.strip():
            return []
        if sep_idx >= len(_RECURSIVE_SEPARATORS):
            # All separators exhausted — forced token split
            toks = tokenizer.encode(text)
            parts = []
            step = max(1, chunk_token_size - chunk_overlap_token_size)
            for start in range(0, len(toks), step):
                parts.append(tokenizer.decode(toks[start: start + chunk_token_size]))
            return parts

        sep = _RECURSIVE_SEPARATORS[sep_idx]
        raw = text.split(sep)
        if len(raw) == 1:
            # Separator not found — try the next one
            return _split(text, sep_idx + 1)

        result: list[str] = []
        buf = ""
        for piece in raw:
            candidate = (buf + sep + piece) if buf else piece
            if len(tokenizer.encode(candidate)) <= chunk_token_size:
                buf = candidate
            else:
                if buf:
                    result.append(buf)
                # piece itself may be too large
                if len(tokenizer.encode(piece)) > chunk_token_size:
                    result.extend(_split(piece, sep_idx + 1))
                    buf = ""
                else:
                    buf = piece
        if buf:
            result.append(buf)
        return result

    pieces = _split(content, 0)
    return [
        {
            "tokens": len(tokenizer.encode(p)),
            "content": p.strip(),
            "chunk_order_index": i,
        }
        for i, p in enumerate(pieces)
        if p.strip()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sentence  (accumulate sentences up to token budget)
# ─────────────────────────────────────────────────────────────────────────────

_SENT_RE = re.compile(r'(?<=[.!?。！？])\s+')


def chunking_sentence(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """
    Sentence-based chunking: accumulate complete sentences until the token
    budget fills, then start a new chunk with the last few sentences as
    overlap (≤ chunk_overlap_token_size tokens).
    """
    sentences = [s.strip() for s in _SENT_RE.split(content) if s.strip()]
    chunks: list[dict[str, Any]] = []
    buf: list[str] = []
    buf_tokens = 0
    chunk_idx = 0

    def _flush(sents: list[str]) -> dict[str, Any]:
        nonlocal chunk_idx
        text = " ".join(sents).strip()
        d = {
            "tokens": len(tokenizer.encode(text)),
            "content": text,
            "chunk_order_index": chunk_idx,
        }
        chunk_idx += 1
        return d

    for sent in sentences:
        sent_toks = len(tokenizer.encode(sent))
        if buf_tokens + sent_toks > chunk_token_size and buf:
            chunks.append(_flush(buf))
            # Build overlap from the tail of the flushed buffer
            overlap: list[str] = []
            overlap_toks = 0
            for s in reversed(buf):
                st = len(tokenizer.encode(s))
                if overlap_toks + st > chunk_overlap_token_size:
                    break
                overlap.insert(0, s)
                overlap_toks += st
            buf = overlap
            buf_tokens = overlap_toks
        buf.append(sent)
        buf_tokens += sent_toks

    if buf:
        chunks.append(_flush(buf))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4. Paragraph  (split on blank lines; token fallback for long paragraphs)
# ─────────────────────────────────────────────────────────────────────────────

def chunking_paragraph(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """
    Paragraph-based chunking: split on blank lines (\\n\\n).
    Paragraphs that exceed chunk_token_size are further split with the
    token-based sliding window.
    """
    from lightrag.operate import chunking_by_token_size

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    results: list[dict[str, Any]] = []
    idx = 0
    for para in paragraphs:
        toks = tokenizer.encode(para)
        if len(toks) <= chunk_token_size:
            results.append({
                "tokens": len(toks),
                "content": para,
                "chunk_order_index": idx,
            })
            idx += 1
        else:
            sub = chunking_by_token_size(
                tokenizer, para,
                chunk_overlap_token_size=chunk_overlap_token_size,
                chunk_token_size=chunk_token_size,
            )
            for s in sub:
                s["chunk_order_index"] = idx
                results.append(s)
                idx += 1
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Registry & factory
# ─────────────────────────────────────────────────────────────────────────────

CHUNKING_STRATEGIES: dict[str, Any] = {
    "token": chunking_token,
    "recursive": chunking_recursive,
    "sentence": chunking_sentence,
    "paragraph": chunking_paragraph,
}


def get_chunking_func(strategy: str):
    """Return the chunking function for the given strategy name (case-insensitive)."""
    func = CHUNKING_STRATEGIES.get(strategy.lower())
    if func is None:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Available: {', '.join(CHUNKING_STRATEGIES)}"
        )
    return func
