from dataclasses import dataclass, field
from typing import Any


KNOWN_PATHS: frozenset[str] = frozenset(
    [
        "naive",
        "hybrid",
        "mix",
        "ppr",
        "local_kg",
        "global_kg",
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
    path_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

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
            name="precise",
            description="Exact lexical constraints: IDs, codes, versions, exact titles, rare strings",
            paths=["qdrant_sparse"],
            rrf_weights={"qdrant_sparse": 1.0},
            path_overrides={"qdrant_sparse": {"exclude_synonym_edges": True}},
        ),
        RetrievalProfile(
            name="semantic",
            description="Default dense+sparse chunk retrieval for ordinary semantic RAG questions",
            paths=["qdrant_chunks_hybrid"],
            rrf_weights={"qdrant_chunks_hybrid": 1.0},
            path_overrides={
                "qdrant_chunks_hybrid": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": True,
                }
            },
        ),
        RetrievalProfile(
            name="local",
            description="Single focal entity: direct attributes, local neighbours, direct relationships",
            paths=["local_kg"],
            rrf_weights={"local_kg": 1.0},
            path_overrides={
                "local_kg": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": True,
                }
            },
        ),
        RetrievalProfile(
            name="global",
            description="Relation, event, or theme driven KG edge retrieval",
            paths=["global_kg"],
            rrf_weights={"global_kg": 1.0},
            path_overrides={
                "global_kg": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": True,
                }
            },
        ),
        RetrievalProfile(
            name="multihop",
            description="Bridge, comparison, causal, or compositional multi-document reasoning via PPR",
            paths=["ppr"],
            rrf_weights={"ppr": 1.0},
            enable_rerank=False,
            min_rrf_score=0.0,
            path_overrides={
                "ppr": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": False,
                }
            },
        ),
        RetrievalProfile(
            name="full",
            description="Failure recovery profile combining PPR, semantic chunk retrieval, and lexical sparse retrieval",
            paths=["ppr", "qdrant_chunks_hybrid", "qdrant_sparse"],
            rrf_weights={
                "ppr": 1.3,
                "qdrant_chunks_hybrid": 0.8,
                "qdrant_sparse": 0.5,
            },
            enable_rerank=False,
            min_rrf_score=0.0,
            max_concurrent_paths=None,
            path_overrides={
                "ppr": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": False,
                },
                "qdrant_chunks_hybrid": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": True,
                },
                "qdrant_sparse": {"exclude_synonym_edges": True},
            },
        ),
        RetrievalProfile(
            name="full_v2",
            description="Agentic v2 failure recovery profile with stronger PPR weighting",
            paths=["ppr", "qdrant_chunks_hybrid", "qdrant_sparse"],
            rrf_weights={
                "ppr": 2.0,
                "qdrant_chunks_hybrid": 0.7,
                "qdrant_sparse": 0.4,
            },
            enable_rerank=False,
            min_rrf_score=0.0,
            max_concurrent_paths=None,
            path_overrides={
                "ppr": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": False,
                },
                "qdrant_chunks_hybrid": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": True,
                },
                "qdrant_sparse": {"exclude_synonym_edges": True},
            },
        ),
        RetrievalProfile(
            name="hybrid_experimental",
            description="Ablation/debug profile for LightRAG hybrid KG retrieval; not a classifier output",
            paths=["hybrid"],
            rrf_weights={"hybrid": 1.0},
            path_overrides={
                "hybrid": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": True,
                }
            },
        ),
        RetrievalProfile(
            name="gfm_multihop",
            description="Multi-hop reasoning via optional GFM graph retrieval plus protected PPR",
            paths=[
                # "gfm",  # uncomment to enable GFM graph retrieval
                "ppr",
            ],
            rrf_weights={
                # "gfm": 1.0,  # uncomment together with path above
                "ppr": 1.0,
            },
            enable_rerank=False,
            min_rrf_score=0.0,
            path_overrides={
                "ppr": {
                    "qdrant_retrieval_mode": "hybrid",
                    "exclude_synonym_edges": False,
                }
            },
        ),
    ]
}
