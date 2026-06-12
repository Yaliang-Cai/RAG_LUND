"""声明级引文验证：LLM 提案、代码裁决（spec §12.2）。"""
from __future__ import annotations

import re
from typing import Any

from raganything.retrieval.json_utils import call_json_object

_SENT_SPLIT = re.compile(r"[。！？!?\n]+|(?<=[a-zA-Z0-9])\.\s")
_MIN_CLAIM_LEN = 6

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


def split_claims(answer: str, min_len: int = _MIN_CLAIM_LEN) -> list[str]:
    parts = [p.strip().rstrip("。！？!?.") for p in _SENT_SPLIT.split(answer) if p and p.strip()]
    return [p for p in parts if len(p) >= min_len]


async def verify_citations(
    model_pool: Any, query: str, answer: str, chunks: list[dict],
) -> tuple[bool, list[str]]:
    claims = split_claims(answer)
    if not claims:
        return False, [query]
    chunk_text = "\n---\n".join(str(c.get("content", ""))[:1500] for c in chunks[:20])
    normalized_corpus = _norm(chunk_text)
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
        # 代码裁决：声称 supported 必须有真实引文（归一化后包含于语料）
        if not (v.get("supported") and quote and quote in normalized_corpus):
            ungrounded.append(claim)
    return (len(ungrounded) == 0), ungrounded
