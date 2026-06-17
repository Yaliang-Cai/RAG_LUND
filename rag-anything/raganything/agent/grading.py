"""账本式增量评估 + 条件终审（spec §5.3/5.4）。"""
from __future__ import annotations

import json
import logging
from typing import Any

from raganything.retrieval.json_utils import call_json_object
from raganything.agent.evidence import EvidencePool, FactLedger, PoolEntry

logger = logging.getLogger(__name__)

_RELATED_OLD_LIMIT = 5
_FINAL_REVIEW_STEPS = 3
_FINAL_REVIEW_LOW_SCORE = 0.4
_FINAL_REVIEW_DUP_RATE = 0.5
_LEDGER_PROMPT_FACT_CAP = 40

_GRADE_PROMPT = """\
You maintain a fact ledger for answering a question. Update it incrementally.

Question: {query}

Current ledger (facts needed to answer; found facts list supporting chunk ids):
{ledger}

Evidence chunks to evaluate (new + relevant old):
{chunks}

Task: re-derive the COMPLETE fact list needed to answer the question.
For each fact: status "found" (with chunk ids from evidence above that support it) or "missing".
Keep fact ids stable when the fact is unchanged. Add new facts if discovered necessary.
Output JSON only:
{{"sufficient": true|false, "facts": [{{"id": "f1", "text": "...", "status": "found|missing", "chunks": ["..."]}}]}}
sufficient=true only when every necessary fact is found.
"""


def _format_chunks(entries: list[PoolEntry]) -> str:
    return "\n---\n".join(f"[{e.chunk_id}] {e.content[:800]}" for e in entries) or "(none)"


def _ledger_for_prompt(ledger: FactLedger) -> dict:
    """账本进 prompt 时的体积保护：事实过多时优先保留 missing（检索目标）
    与最近的 found，避免长 session 账本撑爆 grader 上下文。"""
    data = ledger.to_dict()
    facts = data["facts"]
    if len(facts) <= _LEDGER_PROMPT_FACT_CAP:
        return data
    missing = [f for f in facts if f["status"] == "missing"]
    others = [f for f in facts if f["status"] != "missing"]
    kept = (missing + others)[:_LEDGER_PROMPT_FACT_CAP]
    return {"coverage": data["coverage"], "facts": kept}


class LedgerGrader:
    def __init__(self, model_pool: Any) -> None:
        self._pool = model_pool

    async def grade(
        self, query: str, ledger: FactLedger, pool: EvidencePool, *,
        new_entries: list[PoolEntry],
    ) -> dict:
        window = list(new_entries)
        if ledger.missing():
            # 盲区修复：后发现的 fact 必须能拿早期 chunk 核对 §5.3
            seen = {e.chunk_id for e in window}
            window += [e for e in pool.top(_RELATED_OLD_LIMIT) if e.chunk_id not in seen]
        prompt = _GRADE_PROMPT.format(
            query=query, ledger=json.dumps(_ledger_for_prompt(ledger), ensure_ascii=False),
            chunks=_format_chunks(window),
        )
        try:
            parsed = await call_json_object(
                lambda p, **kw: self._pool.call("grader", p, **kw), prompt, max_tokens=1536)
        except Exception:
            logger.warning("LedgerGrader failed; keeping ledger unchanged", exc_info=True)
            return {"sufficient": False, "facts": []}
        ledger.update(parsed)
        for cid, fact_ids in ledger.supported_chunks().items():
            if cid in pool.entries:
                pool.entries[cid].supports |= fact_ids
        # Trust the grader's sufficiency judgment. Premature sufficiency on thin
        # recall is caught (non-fatally) by should_final_review, which re-grades the
        # full pool only under suspicious conditions — gating sufficiency on a strict
        # "no fact may be missing" invariant instead made it unreachable in practice.
        return {"sufficient": bool(parsed.get("sufficient", False)),
                "facts": parsed.get("facts", [])}

    async def final_review(self, query: str, pool: EvidencePool, top_n: int = 20) -> dict:
        """无账本全池终审：fresh grade（spec §5.4）。"""
        fresh = FactLedger()
        return await self.grade(query, fresh, pool, new_entries=pool.top(top_n)) | {
            "fresh_ledger": fresh.to_dict()}


def should_final_review(
    *, ledger_steps: int, ledger: FactLedger, pool: EvidencePool,
    recent_dup_rates: list[float],
) -> bool:
    if ledger_steps >= _FINAL_REVIEW_STEPS:
        return True
    if len(recent_dup_rates) >= 2 and all(r > _FINAL_REVIEW_DUP_RATE for r in recent_dup_rates[-2:]):
        return True
    for f in ledger.facts.values():
        if f["status"] == "found" and len(f["chunks"]) == 1:
            entry = pool.entries.get(f["chunks"][0])
            score = entry.canonical_score if entry and entry.canonical_score is not None else 0.0
            if score < _FINAL_REVIEW_LOW_SCORE:
                return True
    return False
