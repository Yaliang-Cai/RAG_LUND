"""工具注册表 + agent 专用检索 profile（spec §7）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from raganything.retrieval.profiles import PROFILE_REGISTRY, RetrievalProfile


@dataclass
class ParamSpec:
    default: object
    min: float | None = None
    max: float | None = None


@dataclass
class ToolSpec:
    name: str
    cost: float
    description: str
    profile: str                       # PROFILE_REGISTRY key; "" for non-retrieval tools
    params: dict[str, ParamSpec] = field(default_factory=dict)
    allowed_expand: tuple[str, ...] = ("none",)
    rerank: bool = True                # whether the evidence pool cross-encoder-reranks
                                       # this tool's new chunks (False = trust the tool's
                                       # own scores, e.g. PPR)

    def clamp(self, raw: dict) -> dict:
        out = {}
        for key, spec in self.params.items():
            value = raw.get(key, spec.default)
            if isinstance(spec.default, (int, float)):
                try:
                    value = type(spec.default)(value)
                except (TypeError, ValueError):
                    value = spec.default
                if spec.min is not None:
                    value = max(spec.min, value)
                if spec.max is not None:
                    value = min(spec.max, value)
            out[key] = value
        return out


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def names(self) -> list[str]:
        return list(self._tools)

    def card_text(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name} (cost {t.cost:g}): {t.description}"
                         + (f" | expand: {','.join(t.allowed_expand)}" if t.allowed_expand != ("none",) else ""))
        return "\n".join(lines)


def register_agent_profiles() -> None:
    """Register the two agent retrieval profiles (idempotent).

    Minimal-viable toolset (single responsibility, low overlap, clear boundary):
      - agent_search:   LightRAG local+global KG fused with hybrid (dense+BM25)
                        chunk vectors. The general workhorse. Router rerank is OFF;
                        the evidence pool cross-encoder-reranks at admission.
      - agent_multihop: PPR over the entity graph for bridge / comparison / causal /
                        multi-document reasoning. NO rerank anywhere — PPR scores
                        already rank chunks by multi-hop relevance.
    """
    defs = [
        ("agent_search", ["local_kg", "global_kg", "qdrant_chunks_hybrid"], {
            "local_kg": {"qdrant_retrieval_mode": "hybrid", "exclude_synonym_edges": True},
            "global_kg": {"qdrant_retrieval_mode": "hybrid", "exclude_synonym_edges": True},
            "qdrant_chunks_hybrid": {"qdrant_retrieval_mode": "hybrid", "exclude_synonym_edges": True},
        }),
        ("agent_multihop", ["ppr"], {
            "ppr": {"qdrant_retrieval_mode": "hybrid", "exclude_synonym_edges": False},
        }),
    ]
    for name, paths, overrides in defs:
        if name in PROFILE_REGISTRY:
            continue
        PROFILE_REGISTRY[name] = RetrievalProfile(
            name=name, description=f"agent profile: {'+'.join(paths)}",
            paths=paths, rrf_weights={p: 1.0 for p in paths},
            enable_rerank=False, min_rrf_score=0.0, path_overrides=overrides,
        )


def _TOPK(d: int, mx: int = 30) -> dict[str, ParamSpec]:
    return {"top_k": ParamSpec(default=d, min=1, max=mx), "query": ParamSpec(default="")}


def build_default_registry() -> ToolRegistry:
    """The minimal viable retrieval toolset. Three tools, one clear decision each."""
    register_agent_profiles()
    reg = ToolRegistry()
    reg.register(ToolSpec(
        "search", 2,
        "General retrieval: LightRAG local+global knowledge graph fused with hybrid "
        "(dense+BM25) vectors. Use for any question about facts, entities, attributes, "
        "definitions, summaries. This is the default.",
        "agent_search", _TOPK(12, 30), rerank=True))
    reg.register(ToolSpec(
        "search_multihop", 4,
        "Multi-hop retrieval via Personalized PageRank over the entity graph. Use ONLY "
        "when answering requires connecting facts across documents: bridge questions, "
        "comparisons, causal chains, compositional reasoning.",
        "agent_multihop", _TOPK(20, 40), rerank=False))
    reg.register(ToolSpec(
        "answer", 0, "Finish and generate the answer from gathered evidence.", "",
        {"generation_mode": ParamSpec(default="direct")}))
    return reg
