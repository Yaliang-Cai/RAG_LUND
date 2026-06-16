"""声明级引文验证：LLM 提案、代码裁决（spec §12.2）。"""
from __future__ import annotations

import re
from typing import Any

from raganything.retrieval.json_utils import call_json_object

_SENT_SPLIT = re.compile(r"[。！？!?\n]+|(?<=[a-zA-Z0-9])\.\s")
_MIN_CLAIM_LEN = 6

# Markdown structure lines carry no verifiable claim — headings, horizontal rules and
# table separator rows would otherwise become "claims" no chunk can support, flooding
# verification with empty/false verdicts.
_STRUCTURE_LINE = re.compile(
    r"^\s*(?:#{1,6}\s+\S|[-*_=]{3,}\s*$|\|?[\s:|+-]+\|?\s*$)")

_VERIFY_PROMPT = """\
You verify answer claims against retrieved evidence chunks.
For EACH claim below, quote the supporting span VERBATIM from the chunks,
or mark it unsupported. Output JSON only:
{{"claims": [{{"id": 0, "quote": "<verbatim span or empty>", "supported": true|false}}]}}

Chunks:
{chunks}

Question: {query}

Claims:
{claims}
"""


def _norm(text: str) -> str:
    return re.sub(r"\W+", "", text)


def _strip_markup(text: str) -> str:
    """Drop inline Markdown/LaTeX/citation scaffolding so a claim is the prose a
    reader would actually verify, not formatting tokens."""
    text = re.sub(r"\[[^\]]*\]", " ", text)    # inline [chunk_id] citations
    text = re.sub(r"\$+[^$]*\$+", " ", text)   # LaTeX spans $inline$ / $$display$$
    text = re.sub(r"[*_`>#|]+", " ", text)     # emphasis / code / quote / table pipes
    text = re.sub(r"^\s*[-+]\s+", "", text)    # leading list bullet
    return text


def _is_substantive(claim: str) -> bool:
    # Require enough word characters (CJK counts as \w in Unicode mode) so bare
    # punctuation or formatting residue is never treated as a claim.
    return len(re.findall(r"\w", claim)) >= _MIN_CLAIM_LEN


def split_claims(answer: str, min_len: int = _MIN_CLAIM_LEN) -> list[str]:
    """Factual sentences from a Markdown answer: structure lines dropped, inline
    markup stripped, and duplicates removed — the unit citation verification works
    on. Keeps verification focused on real claims instead of headings/tables/LaTeX."""
    claims: list[str] = []
    seen: set[str] = set()
    for line in answer.splitlines():
        if _STRUCTURE_LINE.match(line):
            continue
        for part in _SENT_SPLIT.split(_strip_markup(line)):
            c = part.strip().rstrip("。！？!?.").strip()
            if len(c) < min_len or not _is_substantive(c):
                continue
            key = re.sub(r"\s+", "", c).lower()
            if key in seen:
                continue
            seen.add(key)
            claims.append(c)
    return claims


async def verify_citations(
    model_pool: Any, query: str, answer: str, chunks: list[dict],
) -> tuple[bool, list[str]]:
    claims = split_claims(answer)
    if not claims:
        return False, [query]
    chunk_text = "\n---\n".join(str(c.get("content", ""))[:1500] for c in chunks[:20])
    # 逐 chunk 归一化：引文必须完整存在于单个 chunk 内，
    # 防止跨 chunk 边界拼接出的伪造引文通过裁决（分隔符被 _norm 抹除的漏洞）
    normalized_chunks = [_norm(str(c.get("content", ""))[:1500]) for c in chunks[:20]]
    prompt = _VERIFY_PROMPT.format(
        chunks=chunk_text, query=query,
        claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims)),
    )
    try:
        parsed = await call_json_object(
            lambda p, **kw: model_pool.call("checker", p, **kw), prompt, max_tokens=1024,
        )
    except Exception:
        return False, claims  # checker 失效 → 保守判全不支持
    verdicts: dict[int, dict] = {}
    for c in parsed.get("claims", []):
        if not isinstance(c, dict):
            continue
        raw_id = c.get("id", -1)
        try:
            verdicts[int(raw_id)] = c
        except (ValueError, TypeError):
            # 跳过 LLM 返回非数字 id 的条目（如 "abc"），避免 ValueError 扩散
            continue
    ungrounded: list[str] = []
    for i, claim in enumerate(claims):
        v = verdicts.get(i, {})
        quote = _norm(str(v.get("quote", "")))
        # 代码裁决：声称 supported 必须有真实引文（归一化后完整存在于某一单个 chunk 内）
        if not (v.get("supported") and quote and any(quote in nc for nc in normalized_chunks)):
            ungrounded.append(claim)
    return (len(ungrounded) == 0), ungrounded
