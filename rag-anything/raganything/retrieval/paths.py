# raganything/retrieval/paths.py
import copy
import dataclasses
import logging
import time

from raganything.constants import GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH
from raganything.retrieval.gfm_retriever import GFMRetrieverWrapper

logger = logging.getLogger(__name__)

# path_name → (lightrag_mode, qdrant_retrieval_mode_override_or_None)
_PATH_CONFIG: dict[str, tuple[str, str | None]] = {
    "naive":                ("naive",   None),
    "local_kg":             ("local",   None),   # entity-centric local KG BFS, no global/community
    "hybrid":               ("hybrid",  None),
    "mix":                  ("mix",     None),
    "ppr":                  ("ppr",     None),
    "qdrant_hybrid":        ("hybrid",  "hybrid"),
    "qdrant_sparse":        ("naive",   "bm25"),
    "qdrant_chunks_hybrid": ("naive",   "hybrid"),  # dense+BM25 chunk search, no KG traversal
    "gfm":                  ("gfm",     None),
}

KNOWN_PATHS: frozenset[str] = frozenset(_PATH_CONFIG.keys())


# Fields that are declared on QueryParam (version-safe lookup performed once)
def _known_fields(obj) -> set[str]:
    return {f.name for f in dataclasses.fields(obj)}


def _copy_param(param, mode: str, extra: dict):
    """Return a shallow copy of *param* with mode replaced and extra attrs set.

    Uses dataclasses.replace() for declared fields and setattr() for any
    dynamic attributes the running LightRAG version may not declare yet
    (e.g. qdrant_retrieval_mode, kg_chunk_selection_source).  The original
    param is never mutated.
    """
    known = _known_fields(param)

    # Separate extra into declared vs. dynamic
    declared_extra = {k: v for k, v in extra.items() if k in known}
    dynamic_extra  = {k: v for k, v in extra.items() if k not in known}

    path_param = dataclasses.replace(param, mode=mode, **declared_extra)

    for k, v in dynamic_extra.items():
        setattr(path_param, k, v)

    return path_param


async def run_path(
    name: str,
    query: str,
    param,           # QueryParam — never mutated
    lightrag,        # LightRAG instance
    overrides: dict,
) -> tuple[list[dict], float]:
    """Execute one retrieval path.

    Returns:
        (chunks, latency_seconds). chunks is an empty list on failure.
    """
    if name == "gfm":
        wrapper = GFMRetrieverWrapper.get_instance(GFM_DATA_DIR, GFM_DATA_NAME, GFM_MODEL_PATH)
        t0 = time.monotonic()
        chunks = await wrapper.retrieve(
            query, getattr(param, "chunk_top_k", 10), lightrag.text_chunks
        )
        latency = time.monotonic() - t0
        return chunks, latency

    if name not in _PATH_CONFIG:
        raise ValueError(f"Unknown path: {name!r}")

    mode, qdrant_mode = _PATH_CONFIG[name]

    # Build extra dict: qdrant_mode first (lower priority), then caller overrides
    extra: dict = {}
    if qdrant_mode is not None:
        extra["qdrant_retrieval_mode"] = qdrant_mode
    extra.update(overrides)

    # Build an isolated param copy; never mutate the caller's param
    path_param = _copy_param(param, mode, extra)

    t0 = time.monotonic()
    result = await lightrag.aquery_data(query, path_param)
    latency = time.monotonic() - t0

    chunks: list[dict] = []
    if result and result.get("status") == "success":
        chunks = result.get("data", {}).get("chunks", [])

    logger.debug("Path '%s': %d chunks in %.3fs", name, len(chunks), latency)
    return chunks, latency
