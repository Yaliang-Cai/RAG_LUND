"""Global PPR Engine — full-graph in-memory cache with scipy + fast-pagerank.

Loads the entire workspace entity graph from Neo4j once (lazily on first call),
builds a scipy CSR adjacency matrix, and extends it per-query with virtual chunk
nodes.  pagerank_power from the fast-pagerank package is used for speed.

Install dependency:
    pip install fast-pagerank
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

try:
    from fast_pagerank import pagerank_power  # type: ignore[import]

    _HAS_FAST_PAGERANK = True
except ImportError:
    _HAS_FAST_PAGERANK = False
    logger.warning(
        "fast-pagerank not installed — GlobalPPREngine unavailable. "
        "Run: pip install fast-pagerank"
    )

# Module-level registry: workspace name → engine instance
_engines: dict[str, "GlobalPPREngine"] = {}


def get_engine(graph_storage) -> "GlobalPPREngine":
    """Return (and lazily create) the singleton engine for a graph storage instance."""
    workspace = getattr(graph_storage, "workspace", id(graph_storage))
    if workspace not in _engines:
        _engines[workspace] = GlobalPPREngine(graph_storage)
    return _engines[workspace]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GlobalPPREngine:
    """Full-graph PPR engine backed by an in-memory scipy CSR matrix.

    The entity–entity adjacency matrix is built once (lazily) from Neo4j and
    cached.  Per-query virtual chunk nodes are appended as a sparse extension
    before pagerank_power runs.

    Attributes:
        node_index:  entity_id → column/row index in the entity block.
        index_node:  int → entity_id (inverse of node_index).
    """

    def __init__(self, graph_storage) -> None:
        self._graph = graph_storage

        # Cached state (None = not loaded yet)
        self.node_index: dict[str, int] | None = None
        self.index_node: list[str] | None = None
        self._adj: csr_matrix | None = None  # N×N entity adjacency
        self._entity_embeddings: dict[str, list[float]] = {}
        self._chunk_to_entities: dict[str, list[str]] = {}
        self._entity_to_chunks: dict[str, list[str]] = {}  # inverse of _chunk_to_entities

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Discard the cached graph; next query will reload from Neo4j."""
        self.node_index = None
        self.index_node = None
        self._adj = None
        self._entity_embeddings = {}
        self._chunk_to_entities = {}
        self._entity_to_chunks = {}

    async def _ensure_loaded(self) -> None:
        if self._adj is not None:
            return
        async with self._lock:
            if self._adj is not None:  # double-checked locking
                return
            await self._load_graph()

    async def _load_graph(self) -> None:
        """Pull all nodes + edges from Neo4j and build the CSR matrix."""
        # Local import to avoid circular dependency at module load time
        from lightrag.utils import GRAPH_FIELD_SEP, split_string_by_multi_markers

        nodes, edges = await self._graph.get_all_nodes_and_edges()

        # --- Node index ---
        index_node: list[str] = []
        node_index: dict[str, int] = {}
        entity_embeddings: dict[str, list[float]] = {}
        chunk_to_entities: dict[str, list[str]] = {}

        for n in nodes:
            eid = n.get("entity_id")
            if not eid:
                continue
            if eid not in node_index:
                node_index[eid] = len(index_node)
                index_node.append(eid)

            emb = n.get("embedding")
            if emb is not None:
                entity_embeddings[eid] = emb

            # Chunks from node source_id (entity-defining chunks)
            source_ids = n.get("source_id") or ""
            for cid in split_string_by_multi_markers(source_ids, [GRAPH_FIELD_SEP]):
                cid = cid.strip()
                if cid:
                    chunk_to_entities.setdefault(cid, []).append(eid)

        N = len(index_node)

        # Build chunk_to_entities from edge source_id (relation chunks)
        # A relation chunk connects to both the src and tgt entity of that edge.
        for e in edges:
            edge_source_ids = e.get("source_id") or ""
            if not edge_source_ids:
                continue
            src, tgt = e.get("src"), e.get("tgt")
            for cid in split_string_by_multi_markers(edge_source_ids, [GRAPH_FIELD_SEP]):
                cid = cid.strip()
                if not cid:
                    continue
                if src:
                    chunk_to_entities.setdefault(cid, []).append(src)
                if tgt:
                    chunk_to_entities.setdefault(cid, []).append(tgt)

        # --- Adjacency matrix ---
        if N == 0:
            logger.warning("GlobalPPREngine: no entity nodes found — empty graph")
            self.node_index = {}
            self.index_node = []
            self._adj = csr_matrix((0, 0), dtype=np.float32)
            self._entity_embeddings = {}
            self._chunk_to_entities = {}
            return

        rows: list[int] = []
        cols: list[int] = []
        data_vals: list[float] = []

        for e in edges:
            src, tgt = e.get("src"), e.get("tgt")
            i = node_index.get(src)
            j = node_index.get(tgt)
            if i is None or j is None:
                continue
            w = float(e.get("weight", 1.0))
            # Undirected — add both directions
            rows.extend([i, j])
            cols.extend([j, i])
            data_vals.extend([w, w])

        adj = csr_matrix(
            (data_vals, (rows, cols)), shape=(N, N), dtype=np.float32
        )

        # Build reverse mapping: entity_id → [chunk_id, ...]
        entity_to_chunks: dict[str, list[str]] = {}
        for cid, eids in chunk_to_entities.items():
            for eid in eids:
                entity_to_chunks.setdefault(eid, []).append(cid)

        self.node_index = node_index
        self.index_node = index_node
        self._adj = adj
        self._entity_embeddings = entity_embeddings
        self._chunk_to_entities = chunk_to_entities
        self._entity_to_chunks = entity_to_chunks

        logger.info(
            f"GlobalPPREngine: loaded {N} entity nodes, {len(edges)} edges, "
            f"{len(chunk_to_entities)} chunk mappings, "
            f"{len(entity_to_chunks)} entity→chunk mappings"
        )

    # ------------------------------------------------------------------
    # PPR
    # ------------------------------------------------------------------

    async def run_ppr(
        self,
        entity_seed_weights: dict[str, float],
        chunk_seed_weights: dict[str, float],
        chunk_to_entities: dict[str, list[str]] | None = None,
        damping: float = 0.5,
        top_k: int = 50,
        chunk_embeddings: dict[str, list[float]] | None = None,
    ) -> list[tuple[str, float]]:
        """Run PPR on the full entity graph extended with virtual chunk nodes.

        Args:
            entity_seed_weights: {entity_id: weight} — already normalised to sum=1.
            chunk_seed_weights:  {chunk_id: weight}  — already scaled by passage_node_weight.
            chunk_to_entities:   {chunk_id: [entity_id, ...]} — edges from virtual chunk nodes
                                 to entity nodes.  If None, falls back to the pre-loaded
                                 mapping (filtered to keys present in chunk_seed_weights).
            damping:             PPR damping factor (α).
            top_k:               Return at most this many (chunk_id, score) pairs.
            chunk_embeddings:    Optional {chunk_id: embedding} for cosine-weighted
                                 chunk→entity edges.

        Returns:
            Sorted list of (chunk_id, ppr_score) tuples, descending by score.
        """
        if not _HAS_FAST_PAGERANK:
            raise RuntimeError(
                "fast-pagerank is not installed. "
                "Install it with: pip install fast-pagerank"
            )

        await self._ensure_loaded()

        node_index = self.node_index
        index_node = self.index_node
        adj = self._adj
        N = len(index_node)

        if N == 0:
            return []

        # Resolve chunk_to_entities: caller may pass a subset or None (→ pre-loaded)
        if chunk_to_entities is None:
            chunk_to_entities = {
                cid: self._chunk_to_entities[cid]
                for cid in chunk_seed_weights
                if cid in self._chunk_to_entities
            }

        # Virtual chunk node indices: N, N+1, …, N+M-1
        chunk_ids = list(chunk_to_entities.keys())
        M = len(chunk_ids)
        if M == 0:
            return []

        chunk_index = {cid: N + i for i, cid in enumerate(chunk_ids)}
        total = N + M

        # --- Extend adjacency matrix with chunk ↔ entity edges ---
        coo = adj.tocoo()
        ext_rows: list[int] = list(coo.row)
        ext_cols: list[int] = list(coo.col)
        ext_data: list[float] = list(coo.data)

        for cid, eids in chunk_to_entities.items():
            ci = chunk_index[cid]
            c_emb = chunk_embeddings.get(cid) if chunk_embeddings else None

            for eid in eids:
                ei = node_index.get(eid)
                if ei is None:
                    continue

                # Edge weight: cosine sim if both embeddings present, else 1.0
                if c_emb is not None and eid in self._entity_embeddings:
                    w = max(
                        0.0,
                        _cosine_similarity(c_emb, self._entity_embeddings[eid]),
                    )
                else:
                    w = 1.0

                ext_rows.extend([ci, ei])
                ext_cols.extend([ei, ci])
                ext_data.extend([w, w])

        full_adj = csr_matrix(
            (ext_data, (ext_rows, ext_cols)),
            shape=(total, total),
            dtype=np.float32,
        )

        # --- Personalisation vector ---
        pers = np.zeros(total, dtype=np.float32)

        for eid, w in entity_seed_weights.items():
            idx = node_index.get(eid)
            if idx is not None:
                pers[idx] = max(pers[idx], float(w))

        for cid, w in chunk_seed_weights.items():
            idx = chunk_index.get(cid)
            if idx is not None:
                pers[idx] = max(pers[idx], float(w))

        total_pers = float(pers.sum())
        if total_pers <= 0.0:
            return []
        pers /= total_pers

        # --- Run PPR ---
        scores = pagerank_power(full_adj, p=damping, personalize=pers, tol=1e-6)

        # --- Extract chunk scores only ---
        chunk_scores = [
            (cid, float(scores[chunk_index[cid]])) for cid in chunk_ids
        ]
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return chunk_scores[:top_k]
