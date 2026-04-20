"""Global PPR engine: full-graph in-memory cache with scipy + fast-pagerank.

Loads the entire workspace entity graph from Neo4j once (lazily on first call),
builds a scipy CSR adjacency matrix, and extends it per-query with virtual chunk
nodes. pagerank_power from the fast-pagerank package is used for speed.

Install dependency:
    pip install fast-pagerank
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Literal

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

try:
    from fast_pagerank import pagerank_power  # type: ignore[import]

    _HAS_FAST_PAGERANK = True
except ImportError:
    _HAS_FAST_PAGERANK = False
    logger.warning(
        "fast-pagerank not installed - GlobalPPREngine unavailable. "
        "Run: pip install fast-pagerank"
    )

# Module-level registry: workspace name -> engine instance
_engines: dict[str, "GlobalPPREngine"] = {}


def get_engine(graph_storage) -> "GlobalPPREngine":
    """Return (and lazily create) the singleton engine for a graph storage instance."""
    workspace = getattr(graph_storage, "workspace", id(graph_storage))
    if workspace not in _engines:
        _engines[workspace] = GlobalPPREngine(graph_storage)
    return _engines[workspace]


def _is_synonym_edge(edge_data: dict[str, Any]) -> bool:
    edge_type = str(edge_data.get("edge_type", "")).upper()
    if edge_type == "SYNONYM":
        return True

    provenance = str(edge_data.get("provenance", "")).strip().lower()
    if provenance == "synonym_detection":
        return True

    source_id = str(edge_data.get("source_id", "") or "").strip()
    keywords = str(edge_data.get("keywords", "") or "").lower()
    return not source_id and ("synonym" in keywords or "alias" in keywords)


def _map_entity_edge_weight(
    edge_data: dict[str, Any],
    synonym_weight_mode: Literal["raw", "plus_one"],
) -> float:
    raw_weight = edge_data.get("weight", 1.0)
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        weight = 1.0

    if not math.isfinite(weight):
        weight = 1.0
    if weight < 0.0:
        weight = 0.0

    if _is_synonym_edge(edge_data) and synonym_weight_mode == "plus_one":
        return 1.0 + weight
    return weight


class GlobalPPREngine:
    """Full-graph PPR engine backed by an in-memory scipy CSR matrix."""

    def __init__(self, graph_storage) -> None:
        self._graph = graph_storage

        # Cached state (None = not loaded yet)
        self.node_index: dict[str, int] | None = None
        self.index_node: list[str] | None = None
        self._adj: csr_matrix | None = None  # raw mode N x N entity adjacency
        self._adj_by_mode: dict[tuple[str, bool], csr_matrix] = {}
        self._entity_edges_for_ppr: list[tuple[int, int, dict[str, Any]]] = []
        self._chunk_to_entities: dict[str, list[str]] = {}
        self._entity_to_chunks: dict[str, list[str]] = {}

        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        """Discard the cached graph; next query will reload from Neo4j."""
        self.node_index = None
        self.index_node = None
        self._adj = None
        self._adj_by_mode = {}
        self._entity_edges_for_ppr = []
        self._chunk_to_entities = {}
        self._entity_to_chunks = {}

    async def _ensure_loaded(self) -> None:
        if self._adj is not None:
            return
        async with self._lock:
            if self._adj is not None:
                return
            await self._load_graph()

    def _build_entity_adj(
        self,
        synonym_weight_mode: Literal["raw", "plus_one"],
        exclude_synonym_edges: bool = False,
    ) -> csr_matrix:
        index_node = self.index_node or []
        n_nodes = len(index_node)
        if n_nodes == 0:
            return csr_matrix((0, 0), dtype=np.float32)

        rows: list[int] = []
        cols: list[int] = []
        data_vals: list[float] = []
        for i, j, edge_data in self._entity_edges_for_ppr:
            if exclude_synonym_edges and _is_synonym_edge(edge_data):
                continue
            mapped_weight = _map_entity_edge_weight(edge_data, synonym_weight_mode)
            rows.extend([i, j])
            cols.extend([j, i])
            data_vals.extend([mapped_weight, mapped_weight])

        return csr_matrix((data_vals, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)

    def _get_entity_adj(
        self,
        synonym_weight_mode: Literal["raw", "plus_one"],
        exclude_synonym_edges: bool = False,
    ) -> csr_matrix:
        cache_key = (synonym_weight_mode, exclude_synonym_edges)
        cached = self._adj_by_mode.get(cache_key)
        if cached is not None:
            return cached

        built = self._build_entity_adj(synonym_weight_mode, exclude_synonym_edges)
        self._adj_by_mode[cache_key] = built
        return built

    async def _load_graph(self) -> None:
        """Pull all nodes + edges from Neo4j and build cached adjacency."""
        # Local import to avoid circular dependency at module load time
        from lightrag.utils import GRAPH_FIELD_SEP, split_string_by_multi_markers

        nodes, edges = await self._graph.get_all_nodes_and_edges()

        index_node: list[str] = []
        node_index: dict[str, int] = {}
        chunk_to_entities_set: dict[str, set[str]] = {}

        for node in nodes:
            entity_id = node.get("entity_id")
            if not entity_id:
                continue
            if entity_id not in node_index:
                node_index[entity_id] = len(index_node)
                index_node.append(entity_id)

            source_ids = node.get("source_id") or ""
            for chunk_id in split_string_by_multi_markers(source_ids, [GRAPH_FIELD_SEP]):
                chunk_id = chunk_id.strip()
                if chunk_id:
                    chunk_to_entities_set.setdefault(chunk_id, set()).add(entity_id)

        n_nodes = len(index_node)
        if n_nodes == 0:
            logger.warning("GlobalPPREngine: no entity nodes found - empty graph")
            self.node_index = {}
            self.index_node = []
            self._adj = csr_matrix((0, 0), dtype=np.float32)
            self._adj_by_mode = {("raw", False): self._adj}
            self._entity_edges_for_ppr = []
            self._chunk_to_entities = {}
            self._entity_to_chunks = {}
            return

        entity_edges_for_ppr: list[tuple[int, int, dict[str, Any]]] = []
        for edge in edges:
            src = edge.get("src")
            tgt = edge.get("tgt")
            i = node_index.get(src)
            j = node_index.get(tgt)
            if i is None or j is None:
                continue
            entity_edges_for_ppr.append((i, j, edge))

            edge_source_ids = edge.get("source_id") or ""
            if not edge_source_ids:
                continue
            for chunk_id in split_string_by_multi_markers(edge_source_ids, [GRAPH_FIELD_SEP]):
                chunk_id = chunk_id.strip()
                if not chunk_id:
                    continue
                if src:
                    chunk_to_entities_set.setdefault(chunk_id, set()).add(src)
                if tgt:
                    chunk_to_entities_set.setdefault(chunk_id, set()).add(tgt)

        chunk_to_entities = {
            chunk_id: sorted(entity_ids)
            for chunk_id, entity_ids in chunk_to_entities_set.items()
            if entity_ids
        }
        entity_to_chunks_set: dict[str, set[str]] = {}
        for chunk_id, entity_ids in chunk_to_entities.items():
            for entity_id in entity_ids:
                entity_to_chunks_set.setdefault(entity_id, set()).add(chunk_id)
        entity_to_chunks = {
            entity_id: sorted(chunk_ids)
            for entity_id, chunk_ids in entity_to_chunks_set.items()
        }

        self.node_index = node_index
        self.index_node = index_node
        self._entity_edges_for_ppr = entity_edges_for_ppr
        self._adj = self._build_entity_adj("raw")
        self._adj_by_mode = {("raw", False): self._adj}
        self._chunk_to_entities = chunk_to_entities
        self._entity_to_chunks = entity_to_chunks

        logger.info(
            "GlobalPPREngine: loaded %d entity nodes, %d edges, %d chunk mappings, %d entity->chunk mappings",
            n_nodes,
            len(entity_edges_for_ppr),
            len(chunk_to_entities),
            len(entity_to_chunks),
        )

    async def run_ppr(
        self,
        entity_seed_weights: dict[str, float],
        chunk_seed_weights: dict[str, float],
        chunk_to_entities: dict[str, list[str]] | None = None,
        damping: float = 0.5,
        top_k: int = 50,
        chunk_embeddings: dict[str, list[float]] | None = None,
        ppr_synonym_weight_mode: Literal["raw", "plus_one"] = "raw",
        exclude_synonym_edges: bool = False,
    ) -> list[tuple[str, float]]:
        """Run PPR on the full entity graph extended with virtual chunk nodes.

        Notes:
            - chunk-entity virtual edges are fixed to weight=1.0.
            - chunk_embeddings is kept only for API compatibility and ignored.
            - when exclude_synonym_edges=True, SYNONYM edges are removed before PPR.
        """
        if not _HAS_FAST_PAGERANK:
            raise RuntimeError(
                "fast-pagerank is not installed. Install it with: pip install fast-pagerank"
            )

        await self._ensure_loaded()

        if chunk_embeddings is not None:
            logger.debug(
                "GlobalPPREngine.run_ppr: chunk_embeddings provided but ignored; "
                "chunk-entity edge weight is fixed to 1.0."
            )

        node_index = self.node_index or {}
        index_node = self.index_node or []
        adj = self._get_entity_adj(
            ppr_synonym_weight_mode,
            exclude_synonym_edges=exclude_synonym_edges,
        )
        n_nodes = len(index_node)
        if n_nodes == 0:
            return []

        if chunk_to_entities is None:
            chunk_to_entities = {
                chunk_id: self._chunk_to_entities[chunk_id]
                for chunk_id in chunk_seed_weights
                if chunk_id in self._chunk_to_entities
            }

        chunk_ids = list(chunk_to_entities.keys())
        n_chunks = len(chunk_ids)
        if n_chunks == 0:
            return []

        chunk_index = {chunk_id: n_nodes + i for i, chunk_id in enumerate(chunk_ids)}
        total_nodes = n_nodes + n_chunks

        coo = adj.tocoo()
        ext_rows: list[int] = list(coo.row)
        ext_cols: list[int] = list(coo.col)
        ext_data: list[float] = [float(v) for v in coo.data]

        for chunk_id, entity_ids in chunk_to_entities.items():
            chunk_i = chunk_index[chunk_id]
            for entity_id in set(entity_ids):
                entity_i = node_index.get(entity_id)
                if entity_i is None:
                    continue
                ext_rows.extend([chunk_i, entity_i])
                ext_cols.extend([entity_i, chunk_i])
                ext_data.extend([1.0, 1.0])

        full_adj = csr_matrix(
            (ext_data, (ext_rows, ext_cols)),
            shape=(total_nodes, total_nodes),
            dtype=np.float32,
        )

        personalization = np.zeros(total_nodes, dtype=np.float32)
        for entity_id, weight in entity_seed_weights.items():
            idx = node_index.get(entity_id)
            if idx is not None:
                personalization[idx] = max(personalization[idx], float(weight))

        for chunk_id, weight in chunk_seed_weights.items():
            idx = chunk_index.get(chunk_id)
            if idx is not None:
                personalization[idx] = max(personalization[idx], float(weight))

        pers_total = float(personalization.sum())
        if pers_total <= 0.0:
            return []
        personalization /= pers_total

        scores = pagerank_power(full_adj, p=damping, personalize=personalization, tol=1e-6)
        ranked = [(chunk_id, float(scores[chunk_index[chunk_id]])) for chunk_id in chunk_ids]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
