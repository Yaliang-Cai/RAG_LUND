# raganything/retrieval/router.py
import asyncio
import logging
from dataclasses import asdict

from lightrag.utils import apply_rerank_if_enabled

from .classifier import QueryClassifier
from .paths import run_path
from .profiles import PROFILE_REGISTRY

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    pass


class RetrievalRouter:
    def __init__(self, lightrag, llm_func=None):
        self._lightrag = lightrag
        self._classifier = QueryClassifier(llm_func or lightrag.llm_model_func)

    async def route(
        self,
        query: str,
        param,
        profile_name: str | None = None,
    ) -> tuple[list[dict], dict]:
        """Route → parallel retrieval → weighted RRF → rerank → threshold.

        Returns:
            (final_chunks, routing_trace)
        """
        # 1. Select profile
        if profile_name is not None:
            profile = PROFILE_REGISTRY.get(profile_name)
            if profile is None:
                raise ValueError(f"Unknown profile: {profile_name!r}")
            classifier_meta = {
                "confidence": 1.0,
                "reasoning": "explicit override",
                "latency": 0.0,
            }
        else:
            profile_name, classifier_meta = await self._classifier.classify(query)
            profile = PROFILE_REGISTRY[profile_name]

        # 2. Parallel path execution with optional semaphore
        sem = (
            asyncio.Semaphore(profile.max_concurrent_paths)
            if profile.max_concurrent_paths
            else None
        )

        async def _guarded(name: str):
            overrides = profile.path_overrides.get(name, {})
            if sem:
                async with sem:
                    return name, await run_path(name, query, param, self._lightrag, overrides)
            return name, await run_path(name, query, param, self._lightrag, overrides)

        results = await asyncio.gather(
            *[_guarded(n) for n in profile.paths],
            return_exceptions=True,
        )

        # 3. Collect results; skip failed paths
        chunks_by_path: dict[str, list[dict]] = {}
        latency_by_path: dict[str, float] = {}
        failed_paths: list[str] = []

        for item in results:
            if isinstance(item, BaseException):
                logger.warning("Retrieval path exception: %s", item)
                continue
            name, (chunks, latency) = item
            chunks_by_path[name] = chunks
            latency_by_path[name] = round(latency, 3)

        for name in profile.paths:
            if name not in chunks_by_path:
                failed_paths.append(name)

        if not chunks_by_path:
            raise RetrievalError("All retrieval paths failed")

        # 4. Weighted RRF — ranked lists enter intact, no pre-dedup
        merged = _weighted_rrf_merge(
            {n: chunks_by_path[n] for n in profile.paths if n in chunks_by_path},
            profile.rrf_weights,
            profile.rrf_k,
        )
        # Filter long-tail low-confidence candidates
        if profile.min_rrf_score > 0.0:
            merged = [c for c in merged if c.get("rrf_score", 0.0) >= profile.min_rrf_score]
        chunks_after_rrf = len(merged)

        # 5. Rerank (capped at rerank_candidate_cap)
        candidate_pool = merged[: profile.rerank_candidate_cap]
        try:
            global_config = asdict(self._lightrag)
        except Exception:
            global_config = {}

        if profile.enable_rerank:
            reranked = await apply_rerank_if_enabled(
                query=query,
                retrieved_docs=candidate_pool,
                global_config=global_config,
                enable_rerank=True,
                top_n=None,
                item_label="chunks",
            )
        else:
            reranked = candidate_pool
        chunks_after_rerank = len(reranked)

        # 6. Threshold filter — per-query override (param.min_rerank_score) wins
        #    over the profile default when explicitly set.
        param_min_rerank = getattr(param, "min_rerank_score", None)
        min_rerank_score = (
            param_min_rerank if param_min_rerank is not None else profile.min_rerank_score
        )
        if profile.enable_rerank:
            filtered = [
                c for c in reranked
                if c.get("rerank_score", 1.0) >= min_rerank_score
            ]
        else:
            filtered = reranked

        # 7. Final top-k
        chunk_top_k = getattr(param, "chunk_top_k", 10)
        final_chunks = filtered[:chunk_top_k]

        routing_trace = {
            "profile": profile_name,
            "confidence": classifier_meta["confidence"],
            "reasoning": classifier_meta["reasoning"],
            "paths_activated": list(chunks_by_path.keys()),
            "paths_failed": failed_paths,
            "chunks_per_path": {n: len(c) for n, c in chunks_by_path.items()},
            "chunks_after_rrf": chunks_after_rrf,
            "chunks_after_rerank": chunks_after_rerank,
            "chunks_after_threshold": len(final_chunks),
            "latency_per_path": {
                "classifier": round(classifier_meta["latency"], 3),
                **latency_by_path,
            },
        }

        return final_chunks, routing_trace


def _weighted_rrf_merge(
    chunks_by_path: dict[str, list[dict]],
    weights: dict[str, float],
    k: int,
) -> list[dict]:
    """Weighted RRF: Score(d) = Σ_p weight_p * 1/(k + rank(d,p)).

    Each path's ranked list is passed intact so that a chunk appearing
    in multiple paths accumulates cross-path rank signals.
    Output is deduplicated and sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for path_name, ranked in chunks_by_path.items():
        w = weights.get(path_name, 1.0)
        for rank, chunk in enumerate(ranked):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w / (k + rank + 1)
            if chunk_id not in meta:
                meta[chunk_id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    result = []
    for cid in sorted_ids:
        chunk = dict(meta[cid])
        chunk["rrf_score"] = round(scores[cid], 6)
        result.append(chunk)
    return result
