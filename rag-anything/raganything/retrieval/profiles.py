from dataclasses import dataclass, field


@dataclass
class RetrievalProfile:
    name: str
    description: str
    paths: list[str]
    rrf_weights: dict[str, float]
    rrf_k: int = 60
    enable_rerank: bool = True
    min_rerank_score: float = 0.3
    rerank_candidate_cap: int = 60
    max_concurrent_paths: int | None = None
    path_overrides: dict[str, dict] = field(default_factory=dict)


KNOWN_PATHS: frozenset[str] = frozenset(
    ["naive", "hybrid", "mix", "ppr", "qdrant_hybrid", "qdrant_sparse"]
)

PROFILE_REGISTRY: dict[str, RetrievalProfile] = {
    p.name: p
    for p in [
        RetrievalProfile(
            name="precise",
            description="Exact character-level match queries (error codes, IDs, rare proper nouns)",
            paths=["qdrant_sparse"],
            rrf_weights={"qdrant_sparse": 1.0},
        ),
        RetrievalProfile(
            name="local",
            description="Direct query targeting a specific entity or clear single-hop fact",
            paths=["hybrid", "naive"],
            rrf_weights={"hybrid": 1.0, "naive": 1.0},
        ),
        RetrievalProfile(
            name="multihop",
            description="Chain reasoning across multiple entities or documents",
            paths=["ppr", "hybrid"],
            rrf_weights={"ppr": 1.0, "hybrid": 0.8},
        ),
        RetrievalProfile(
            name="descriptive",
            description="Open-ended question requiring broad, complete context",
            paths=["mix", "qdrant_hybrid"],
            rrf_weights={"mix": 1.0, "qdrant_hybrid": 0.8},
            path_overrides={
                "mix": {
                    "kg_chunk_selection_source": "untruncated",
                    "answer_context_mode": "kg_prompt",
                },
                "qdrant_hybrid": {
                    "kg_chunk_selection_source": "untruncated",
                    "answer_context_mode": "kg_prompt",
                },
            },
        ),
        RetrievalProfile(
            name="full",
            description="Fallback when query type is unclear or ambiguous",
            paths=["naive", "hybrid", "mix", "ppr", "qdrant_hybrid", "qdrant_sparse"],
            rrf_weights={
                p: 1.0
                for p in ["naive", "hybrid", "mix", "ppr", "qdrant_hybrid", "qdrant_sparse"]
            },
            max_concurrent_paths=3,
        ),
    ]
}
