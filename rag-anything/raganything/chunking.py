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
        if sent_toks > chunk_token_size:
            # Flush current buffer first, then split oversized sentence.
            if buf:
                chunks.append(_flush(buf))
                buf = []
                buf_tokens = 0

            for sub in chunking_recursive(
                tokenizer,
                sent,
                chunk_overlap_token_size=chunk_overlap_token_size,
                chunk_token_size=chunk_token_size,
            ):
                sub["chunk_order_index"] = chunk_idx
                chunks.append(sub)
                chunk_idx += 1
            continue

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
    Small consecutive paragraphs are merged up to chunk_token_size so that
    the output never contains hundreds of tiny single-paragraph chunks.
    Paragraphs that exceed chunk_token_size are further split with the
    token-based sliding window.
    """
    from lightrag.operate import chunking_by_token_size

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    results: list[dict[str, Any]] = []
    idx = 0
    buf = ""
    buf_tokens = 0
    sep_tokens = len(tokenizer.encode("\n\n"))

    def _flush_buf() -> None:
        nonlocal buf, buf_tokens, idx
        if buf.strip():
            results.append({
                "tokens": len(tokenizer.encode(buf.strip())),
                "content": buf.strip(),
                "chunk_order_index": idx,
            })
            idx += 1
        buf, buf_tokens = "", 0

    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para))

        if para_tokens > chunk_token_size:
            # Flush accumulated buffer first, then split the oversized paragraph.
            _flush_buf()
            for sub in chunking_by_token_size(
                tokenizer, para,
                chunk_overlap_token_size=chunk_overlap_token_size,
                chunk_token_size=chunk_token_size,
            ):
                sub["chunk_order_index"] = idx
                results.append(sub)
                idx += 1
        else:
            add_tokens = para_tokens if not buf else sep_tokens + para_tokens
            if buf_tokens + add_tokens <= chunk_token_size:
                buf = (buf + "\n\n" + para).strip() if buf else para
                buf_tokens += add_tokens
            else:
                _flush_buf()
                buf = para
                buf_tokens = para_tokens

    _flush_buf()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Semantic  (structure-aware: section headers → merge small / split large)
# ─────────────────────────────────────────────────────────────────────────────

# Ordered from strongest to weakest structural signal
_SECTION_HEADER_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+", re.MULTILINE),            # Markdown: # Title / ## Sub
    re.compile(r"^.+\n[=\-]{3,}\s*$", re.MULTILINE),      # Underlined: Title\n===
    # Numbered section headers:
    # "1 Introduction", "1. Introduction", "2) Methods", "3.2.1 Scaled Dot-Product..."
    re.compile(
        r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLCM]+[.)])\s+[A-Z\u4e00-\u9fff][^\n]{2,100}$",
        re.MULTILINE,
    ),
    # ALL-CAPS title lines: 8–60 chars, only uppercase letters/spaces/hyphens.
    # Literal space (not \s) prevents cross-line matches; no digits/colons/commas
    # avoids false positives from table cells and reference titles.
    re.compile(r"^[A-Z][A-Z \-]{7,59}$", re.MULTILINE),
]


def _section_boundaries(content: str) -> list[int]:
    """Return sorted char-positions where a new section header starts."""
    positions: set[int] = {0}
    for pattern in _SECTION_HEADER_PATTERNS:
        for m in pattern.finditer(content):
            positions.add(m.start())
    return sorted(positions)


def _prepend_section_header_if_needed(
    tokenizer: Tokenizer,
    chunk_content: str,
    section_header: str,
    chunk_token_size: int,
) -> str:
    """
    Ensure recursive sub-chunks keep the originating section header verbatim.

    If adding the header would exceed `chunk_token_size`, trim body tokens to fit.
    """
    body = (chunk_content or "").strip()
    header = (section_header or "").strip()
    if not header:
        return body
    if body.startswith(header):
        return body

    prefix = f"{header}\n"
    prefix_tokens = tokenizer.encode(prefix)
    body_tokens = tokenizer.encode(body)

    if len(prefix_tokens) >= chunk_token_size:
        return body

    allowed_body_tokens = max(0, chunk_token_size - len(prefix_tokens))
    trimmed_body = tokenizer.decode(body_tokens[:allowed_body_tokens]).strip()
    if not trimmed_body:
        return header
    return f"{header}\n{trimmed_body}"


def chunking_semantic(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """
    Structure-aware semantic chunking (Databricks-style).

    Steps
    -----
    1. Detect section boundaries: Markdown headers (# / ##), underlined headers,
       and ALL-CAPS title lines — same structural markers used in the Databricks
       RecursiveCharacterTextSplitter approach.
    2. Merge consecutive small sections that together fit within `chunk_token_size`
       to avoid tiny isolated chunks.
    3. Split sections that exceed `chunk_token_size` via the recursive strategy
       (which respects paragraph → sentence → word hierarchy with overlap).

    Result: chunks that honour document structure, keeping section content
    together rather than cutting blindly at a token boundary.
    """
    boundaries = _section_boundaries(content)

    # Build raw section blocks
    sections: list[str] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(content)
        block = content[start:end].strip()
        if block:
            sections.append(block)

    # Fall back to paragraph splitting when no structure was detected
    if len(sections) <= 1:
        return chunking_paragraph(
            tokenizer, content,
            chunk_overlap_token_size=chunk_overlap_token_size,
            chunk_token_size=chunk_token_size,
        )

    chunks: list[dict[str, Any]] = []
    idx = 0
    buf = ""
    buf_tokens = 0
    sep_tokens = len(tokenizer.encode("\n\n"))

    def _flush() -> None:
        nonlocal buf, buf_tokens, idx
        if buf.strip():
            chunks.append({
                "tokens": len(tokenizer.encode(buf.strip())),
                "content": buf.strip(),
                "chunk_order_index": idx,
            })
            idx += 1
        buf, buf_tokens = "", 0

    for section in sections:
        sec_tokens = len(tokenizer.encode(section))
        section_header = section.splitlines()[0].strip() if section.strip() else ""

        if sec_tokens > chunk_token_size:
            # Flush accumulated buffer first, then split the oversized section
            _flush()
            for sub in chunking_recursive(
                tokenizer, section,
                chunk_overlap_token_size=chunk_overlap_token_size,
                chunk_token_size=chunk_token_size,
            ):
                patched_content = _prepend_section_header_if_needed(
                    tokenizer=tokenizer,
                    chunk_content=sub["content"],
                    section_header=section_header,
                    chunk_token_size=chunk_token_size,
                )
                sub["content"] = patched_content
                sub["tokens"] = len(tokenizer.encode(patched_content))
                sub["chunk_order_index"] = idx
                chunks.append(sub)
                idx += 1
        else:
            add_tokens = sec_tokens if not buf else sep_tokens + sec_tokens
            if buf_tokens + add_tokens <= chunk_token_size:
                # Merge small section into current buffer.
                buf = (buf + "\n\n" + section).strip() if buf else section
                buf_tokens += add_tokens
                continue

            # Buffer is full — flush, then start fresh with this section
            _flush()
            buf = section
            buf_tokens = sec_tokens

    _flush()
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Registry & factory
# ─────────────────────────────────────────────────────────────────────────────

CHUNKING_STRATEGIES: dict[str, Any] = {
    "token": chunking_token,
    "recursive": chunking_recursive,
    "sentence": chunking_sentence,
    "paragraph": chunking_paragraph,
    "semantic": chunking_semantic,
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
