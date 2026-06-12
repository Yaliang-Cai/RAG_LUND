"""轮初合并调用：改写+分类+实体登记（spec §9）。"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from raganything.retrieval.json_utils import call_json_object
from raganything.agent.session import SessionMemory

logger = logging.getLogger(__name__)

ARCHETYPES = {"factoid", "summary", "multihop", "comparison", "unknown"}
FAST_PATH_CONFIDENCE = 0.8

# spec §9.2 画像→策略矩阵
ARCHETYPE_PRESETS: dict[str, dict] = {
    "factoid":    {"tool": "search_sparse", "top_k": 5,  "expand": "none", "generation_mode": "direct"},
    "summary":    {"tool": "search_hybrid", "top_k": 25, "expand": "mqe",  "generation_mode": "map_reduce"},
    "multihop":   {"tool": "search_hybrid", "top_k": 15, "expand": "none", "generation_mode": "cot_reflect"},
    "comparison": {"tool": "search_hybrid", "top_k": 10, "expand": "none", "generation_mode": "direct"},
    "unknown":    {"tool": "search_hybrid", "top_k": 15, "expand": "none", "generation_mode": "direct"},
}

_PLAN_PROMPT = """\
You prepare a user question for retrieval. Given conversation context, output JSON only:
{{"standalone_query": "<self-contained rewrite of the current question, resolve all references>",
  "archetype": "factoid|summary|multihop|comparison|unknown",
  "confidence": 0.0,
  "exact_terms": ["<IDs, model numbers, proper nouns needing exact match>"],
  "suggested_expand": "none|mqe|hyde",
  "visual_intent": false,
  "entities_referenced": [{{"name": "...", "note": "<role in conversation>", "last_turn": 0}}]}}

archetype rules: factoid=single specific fact; summary=broad survey/summarize;
multihop=requires chaining facts across documents; comparison=compare two+ entities;
unknown=unclear. visual_intent=true only if answering requires inspecting image pixels
beyond textual descriptions (read chart values, layout, colors).

History summary: {summary}
Active entities: {entities}
Recent turns:
{turns}

Current question: {query}
"""


@dataclass
class PlanResult:
    standalone_query: str
    archetype: str
    confidence: float
    exact_terms: list[str] = field(default_factory=list)
    suggested_expand: str = "none"
    visual_intent: bool = False
    fast_path: bool = False
    preset: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("standalone_query", "archetype", "confidence", "exact_terms",
                 "suggested_expand", "visual_intent", "fast_path", "preset")}


def _cache_key(query: str) -> str:
    return re.sub(r"\s+", "", query)


async def make_plan(model_pool: Any, query: str, session: SessionMemory) -> PlanResult:
    key = _cache_key(query)
    if key in session.plan_cache:
        return PlanResult(**session.plan_cache[key])

    turns = "\n".join(f"U: {t['q']}\nA: {str(t['a'])[:300]}" for t in session.recent_turns) or "(none)"
    prompt = _PLAN_PROMPT.format(
        summary=session.history_summary or "(none)",
        entities=", ".join(e["name"] for e in session.active_entities) or "(none)",
        turns=turns, query=query,
    )
    try:
        parsed = await call_json_object(
            lambda p, **kw: model_pool.call("rewriter", p, **kw), prompt, max_tokens=512)
    except Exception:
        logger.warning("plan call failed; defaulting to unknown archetype", exc_info=True)
        parsed = {}

    archetype = str(parsed.get("archetype", "unknown"))
    if archetype not in ARCHETYPES:
        archetype = "unknown"
    confidence = float(parsed.get("confidence") or 0.0)
    if confidence < 0.6 and archetype != "unknown":
        archetype = "unknown"  # 低置信走稳妥默认 §9.2
    entities = [e for e in parsed.get("entities_referenced", []) if isinstance(e, dict)]
    if entities:
        session.register_entities(entities)

    plan = PlanResult(
        standalone_query=str(parsed.get("standalone_query") or query),
        archetype=archetype,
        confidence=confidence,
        exact_terms=[str(t) for t in parsed.get("exact_terms", [])],
        suggested_expand=str(parsed.get("suggested_expand", "none")),
        visual_intent=bool(parsed.get("visual_intent", False)),
        fast_path=(archetype == "factoid" and confidence >= FAST_PATH_CONFIDENCE),
        preset=dict(ARCHETYPE_PRESETS[archetype]),
    )
    if plan.exact_terms and plan.archetype == "factoid":
        plan.preset["tool"] = "search_sparse"
    elif plan.archetype == "factoid":
        plan.preset["tool"] = "search_dense"  # 语义型 factoid §9.2
    session.plan_cache[key] = plan.to_dict()
    return plan
