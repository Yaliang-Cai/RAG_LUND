from dataclasses import dataclass, field


KNOWN_PATHS: frozenset[str] = frozenset(
    [
        "naive",
        "hybrid",
        "mix",
        "ppr",
        "local_kg",
        "qdrant_hybrid",
        "qdrant_sparse",
        "qdrant_chunks_hybrid",
        "gfm",
    ]
)


@dataclass
class RetrievalProfile:
    name: str
    description: str
    paths: list[str]
    rrf_weights: dict[str, float]
    rrf_k: int = 60
    enable_rerank: bool = True
    min_rerank_score: float = 0.3
    rerank_candidate_cap: int = 30
    min_rrf_score: float = 0.01
    max_concurrent_paths: int | None = None
    path_overrides: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing_weights = [p for p in self.paths if p not in self.rrf_weights]
        if missing_weights:
            raise ValueError(
                f"Profile '{self.name}': paths missing from rrf_weights: {missing_weights}"
            )
        extra_weights = [p for p in self.rrf_weights if p not in self.paths]
        if extra_weights:
            raise ValueError(
                f"Profile '{self.name}': rrf_weights keys not in paths: {extra_weights}"
            )
        unknown_paths = [p for p in self.paths if p not in KNOWN_PATHS]
        if unknown_paths:
            raise ValueError(
                f"Profile '{self.name}': unknown path names: {unknown_paths}"
            )


PROFILE_REGISTRY: dict[str, RetrievalProfile] = {
    p.name: p
    for p in [
        RetrievalProfile(
            name="semantic",
            description="Factual Q&A, process explanations, concept definitions — LightRAG mix only",
            paths=["mix"],
            rrf_weights={"mix": 1.0},
            # mix already reranks internally via process_chunks_unified; the
            # router-level second pass is redundant for single-path profiles.
            enable_rerank=False,
        ),
        RetrievalProfile(
            name="multihop",
            description="Multi-entity cross-document reasoning — HippoRAG2 PPR only",
            paths=["ppr"],
            rrf_weights={"ppr": 1.0},
            enable_rerank=False,
        ),
        RetrievalProfile(
            name="full",
            description="Fallback for highly ambiguous queries or semantically diffuse multi-turn conversations",
            paths=["ppr", "hybrid", "naive", "qdrant_sparse"],
            rrf_weights={
                "ppr":           1.2,
                "hybrid":        1.0,
                "naive":         0.7,
                "qdrant_sparse": 0.9,
            },
            max_concurrent_paths=None,
        ),
    ]
}
