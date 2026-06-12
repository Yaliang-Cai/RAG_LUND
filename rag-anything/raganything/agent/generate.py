"""上下文装填 + direct/map_reduce/cot_reflect（spec §10/§11.2）。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from raganything.agent.evidence import EvidencePool, FactLedger, PoolEntry

DEFAULT_MAX_CONTEXT_TOKENS = 12_000
MAX_IMAGES_PER_CALL = 6  # per-generation-call 上限 §11.2
_IMAGE_SCORE_FLOOR = 0.3


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)


@dataclass
class PackedContext:
    chunks: list[PoolEntry] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n---\n".join(f"[{e.chunk_id}] ({e.file_path}) {e.content}" for e in self.chunks)


def pack_context(
    pool: EvidencePool, ledger: FactLedger, *,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    visual_intent: bool = False, max_images: int = MAX_IMAGES_PER_CALL,
) -> PackedContext:
    supporters = sorted((e for e in pool.entries.values() if e.supports),
                        key=lambda e: e.sort_key(), reverse=True)
    others = sorted((e for e in pool.entries.values() if not e.supports),
                    key=lambda e: e.sort_key(), reverse=True)
    packed = PackedContext()
    budget = max_context_tokens
    for entry in supporters + others:  # 支撑事实优先 §5.3/§14 评审5
        cost = estimate_tokens(entry.content)
        if cost > budget:
            continue
        packed.chunks.append(entry)
        budget -= cost
    packed.chunks.sort(key=lambda e: e.sort_key(), reverse=True)
    if visual_intent:  # 三道门：意图门已过，再过相关门 §11.2
        for e in packed.chunks:
            if len(packed.images) >= max_images:
                break
            if not (e.supports or (e.canonical_score or 0.0) >= _IMAGE_SCORE_FLOOR):
                continue
            for path in e.image_paths:
                if len(packed.images) >= max_images:
                    break
                packed.images.append(path)
    return packed


_DIRECT_PROMPT = """\
Answer based ONLY on the evidence below. If evidence is insufficient, say what is missing.

Evidence:
{context}

Question: {query}
"""

_MAP_PROMPT = """\
Summarize the evidence below ONLY as it relates to the question. Max 300 tokens.

Question: {query}

Evidence:
{context}
"""

_REDUCE_PROMPT = """\
Synthesize the per-document summaries into a final answer to the question.

Question: {query}

Summaries:
{summaries}
"""

_COT_PROMPT = """\
Answer step by step, anchoring EVERY reasoning step on the verified facts below.
Do not introduce claims unsupported by the facts or evidence.

Verified facts:
{facts}

Unverifiable details (state these as unconfirmed if relevant):
{unverifiable}

Evidence:
{context}

Question: {query}
"""


async def generate_answer(
    model_pool: Any, query: str, pool: EvidencePool, ledger: FactLedger, *,
    mode: str = "direct", max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    visual_intent: bool = False,
) -> str:
    packed = pack_context(pool, ledger, max_context_tokens=max_context_tokens,
                          visual_intent=visual_intent)
    if mode == "map_reduce":
        total = sum(estimate_tokens(e.content) for e in pool.entries.values())
        if total > max_context_tokens:
            return await _map_reduce(model_pool, query, pool, max_context_tokens)
        mode = "direct"  # 预算内退化 §10
    if mode == "cot_reflect":
        facts = "\n".join(f"- {f['text']} (chunks: {','.join(f['chunks'])})"
                          for f in ledger.facts.values() if f["status"] == "found") or "(none)"
        unver = "\n".join(f"- {f['text']}" for f in ledger.unverifiable()) or "(none)"
        prompt = _COT_PROMPT.format(facts=facts, unverifiable=unver,
                                    context=packed.text(), query=query)
    else:
        prompt = _DIRECT_PROMPT.format(context=packed.text(), query=query)
    return str(await model_pool.call("generator", prompt))


async def _map_reduce(model_pool: Any, query: str, pool: EvidencePool,
                      max_context_tokens: int) -> str:
    groups: dict[str, list[PoolEntry]] = {}
    for e in sorted(pool.entries.values(), key=lambda e: e.sort_key(), reverse=True):
        groups.setdefault(e.file_path or "(unknown)", []).append(e)
    per_group = max(1000, max_context_tokens // max(1, len(groups)))

    async def _map_one(entries: list[PoolEntry]) -> str:
        budget, parts = per_group, []
        for e in entries:
            cost = estimate_tokens(e.content)
            if cost > budget:
                continue
            parts.append(e.content)
            budget -= cost
        return str(await model_pool.call(
            "generator", _MAP_PROMPT.format(query=query, context="\n---\n".join(parts))))

    summaries = await asyncio.gather(*[_map_one(v) for v in groups.values()])
    body = "\n\n".join(f"[{name}]\n{s}" for name, s in zip(groups, summaries))
    return str(await model_pool.call(
        "generator", _REDUCE_PROMPT.format(query=query, summaries=body)))
