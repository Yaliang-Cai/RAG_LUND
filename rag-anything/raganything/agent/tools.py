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
    profile: str                       # 对应 PROFILE_REGISTRY 键；非检索工具为 ""
    params: dict[str, ParamSpec] = field(default_factory=dict)
    allowed_expand: tuple[str, ...] = ("none",)

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

    def names(self) -> list[str]:
        return list(self._tools)

    def card_text(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name} (cost {t.cost:g}): {t.description}"
                         + (f" | expand: {','.join(t.allowed_expand)}" if t.allowed_expand != ("none",) else ""))
        return "\n".join(lines)


def register_agent_profiles() -> None:
    """注册 enable_rerank=False 的单路 profile；幂等。"""
    defs = [
        ("agent_sparse", ["qdrant_sparse"]),
        ("agent_dense", ["naive"]),
        ("agent_hybrid", ["qdrant_hybrid"]),
        ("agent_graph", ["local_kg", "global_kg"]),
        ("agent_ppr", ["ppr"]),
    ]
    for name, paths in defs:
        if name in PROFILE_REGISTRY:
            continue
        PROFILE_REGISTRY[name] = RetrievalProfile(
            name=name, description=f"agent tool path: {'+'.join(paths)}",
            paths=paths, rrf_weights={p: 1.0 for p in paths}, enable_rerank=False,
        )


def _TOPK(d: int) -> dict[str, ParamSpec]:
    return {"top_k": ParamSpec(default=d, min=1, max=60), "query": ParamSpec(default="")}


def build_default_registry() -> ToolRegistry:
    register_agent_profiles()
    reg = ToolRegistry()
    reg.register(ToolSpec("search_sparse", 1, "BM25 精确词项检索；型号/ID/专名首选", "agent_sparse",
                          _TOPK(10), ("none", "mqe")))
    reg.register(ToolSpec("search_dense", 1, "稠密语义检索；语义型问题首选", "agent_dense",
                          _TOPK(10), ("none", "mqe", "hyde")))
    reg.register(ToolSpec("rewrite_query", 1, "重写当前检索查询", "",
                          {"query": ParamSpec(default="")}))
    reg.register(ToolSpec("search_hybrid", 2, "稠密+BM25 RRF 混合；标准武器", "agent_hybrid",
                          _TOPK(15), ("none", "mqe")))
    reg.register(ToolSpec("search_graph", 2, "知识图谱 local+global 检索", "agent_graph", _TOPK(15)))
    reg.register(ToolSpec("inspect_image", 2, "VLM 定向看图，提取文字事实入池", "",
                          {"chunk_ids": ParamSpec(default=[]), "question": ParamSpec(default="")}))
    reg.register(ToolSpec("search_ppr", 4, "全图 PPR 多跳检索；仅缺桥事实时使用", "agent_ppr", _TOPK(20)))
    reg.register(ToolSpec("decompose_search", 8, "问题分解+并行混合检索；最后手段", "agent_hybrid",
                          _TOPK(15)))
    reg.register(ToolSpec("answer", 0, "终结：生成答案", "",
                          {"generation_mode": ParamSpec(default="direct")}))
    return reg
