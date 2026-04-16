"""V2: Synonym Linking 鈥?build SYNONYM edges between semantically similar entities.

HippoRAG2-aligned: uses pre-computed entity embeddings for exact cosine similarity
via local numpy matrix multiplication.  All embeddings are fetched once from the VDB
and pairwise similarities are computed in Python 鈥?zero per-entity VDB round-trips.

Computation is batched row-wise so the full N脳N matrix is never materialised in memory.

This module is ingestion-only and fully independent of V3 (PPR multi-hop).
When V3 is enabled, PPR naturally propagates along SYNONYM edges.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from lightrag.base import BaseVectorStorage, BaseGraphStorage
from lightrag.utils import logger, compute_entity_vdb_id

# Rows processed per matmul batch.  1 000 rows 脳 100 000 cols 脳 float32 鈮?400 MB peak.
# Reduce if memory is constrained; increase for throughput on large corpora.
_MATMUL_BATCH = 1_000

SYNONYM_EDGE_TYPE = "SYNONYM"
SYNONYM_EDGE_PROVENANCE = "synonym_detection"


def _is_synonym_edge(edge_data: dict[str, Any] | None) -> bool:
    """Best-effort synonym edge detection for compatibility with legacy records."""
    if not isinstance(edge_data, dict):
        return False

    edge_type = str(edge_data.get("edge_type", "")).upper()
    if edge_type == SYNONYM_EDGE_TYPE:
        return True

    provenance = str(edge_data.get("provenance", "")).strip().lower()
    if provenance == SYNONYM_EDGE_PROVENANCE:
        return True

    source_id = str(edge_data.get("source_id", "") or "").strip()
    keywords = str(edge_data.get("keywords", "") or "").lower()
    if not source_id and ("synonym" in keywords or "alias" in keywords):
        return True

    return False


def _graph_id_to_vdb_id(graph_id: str, enable_disambiguation: bool) -> str:
    """Convert a graph node ID to its corresponding VDB hash ID.

    Graph node IDs are composite (``"name|type"``) when disambiguation is on,
    or plain names when off.  This function reverses the split so we always
    call the canonical factory ``compute_entity_vdb_id``.
    """
    if enable_disambiguation and "|" in graph_id:
        name, etype = graph_id.rsplit("|", 1)
    else:
        name, etype = graph_id, ""
    return compute_entity_vdb_id(name, etype, enable_disambiguation)


def _passes_length_filter(graph_id: str, min_entity_len: int) -> bool:
    name = graph_id.split("|")[0] if "|" in graph_id else graph_id
    return len(re.sub(r'[^A-Za-z0-9\u4e00-\u9fff]', '', name)) > min_entity_len


async def build_synonym_edges(
    entities_vdb: BaseVectorStorage,
    knowledge_graph_inst: BaseGraphStorage,
    new_entity_ids: list[str] | None = None,
    synonymy_threshold: float = 0.8,
    synonymy_topk: int = 2048,  # unused 鈥?kept for API compatibility; exact pairwise replaces KNN
    min_entity_len: int = 2,
    enable_entity_disambiguation: bool = True,
) -> int:
    """Create SYNONYM edges between semantically similar entities.

    Fetches all entity embeddings from the VDB once, then computes cosine
    similarity locally via batched numpy matmul 鈥?aligned with HippoRAG2's
    exact matrix-multiply approach.

    Args:
        entities_vdb: Vector DB containing entity embeddings.
        knowledge_graph_inst: Graph storage for reading/writing edges.
        new_entity_ids: If given, only *query* from these entities (incremental
            mode).  They are compared against the full entity universe so that
            existing entities can also gain SYNONYM edges to new arrivals.
            Each ID is a graph-node ID (composite ``"name|type"`` when
            disambiguation is on, plain name when off).
        synonymy_threshold: Minimum cosine similarity to create a SYNONYM edge.
        synonymy_topk: Unused.  Kept for backward-compatibility with callers
            that pass this parameter.  Exact pairwise comparison supersedes KNN.
        min_entity_len: Minimum alphanumeric/CJK character count in the *query*
            entity name.  Short entities are skipped as query sources (aligned
            with HippoRAG2's ``>2 alphanumeric chars`` filter).
        enable_entity_disambiguation: Mirror of the V1 global toggle. Controls
            how graph-node IDs are mapped to VDB hash IDs.

    Returns:
        Number of SYNONYM edges created.
    """
    if new_entity_ids is not None and len(new_entity_ids) == 0:
        return 0

    # ------------------------------------------------------------------
    # 1. Determine query side and reference side
    # ------------------------------------------------------------------
    all_labels = await knowledge_graph_inst.get_all_labels()
    if not all_labels:
        return 0

    # Query side: entities we run similarity *from*
    if new_entity_ids is not None:
        query_ids = list(new_entity_ids)
    else:
        query_ids = all_labels  # full mode: query == reference

    # ------------------------------------------------------------------
    # 2. Fetch embeddings
    #    Reference side always = entire entity universe (so new entities
    #    can form edges with existing ones in incremental mode).
    #    Full mode: query_ids == all_labels 鈫?single fetch, shared dict.
    # ------------------------------------------------------------------
    ref_vdb_ids = [
        _graph_id_to_vdb_id(eid, enable_entity_disambiguation) for eid in all_labels
    ]
    ref_emb_dict: dict[str, list[float]] = await entities_vdb.get_vectors_by_ids(
        ref_vdb_ids
    )

    if new_entity_ids is None:
        # Full mode: query embeddings are the same dict
        query_emb_dict = ref_emb_dict
        query_vdb_ids = ref_vdb_ids
    else:
        query_vdb_ids = [
            _graph_id_to_vdb_id(eid, enable_entity_disambiguation) for eid in query_ids
        ]
        # Avoid a second VDB round-trip when query IDs are a subset of all_labels
        query_emb_dict = {
            vid: ref_emb_dict[vid]
            for vid in query_vdb_ids
            if vid in ref_emb_dict
        }
        missing = [vid for vid in query_vdb_ids if vid not in ref_emb_dict]
        if missing:
            extra = await entities_vdb.get_vectors_by_ids(missing)
            query_emb_dict.update(extra)

    # ------------------------------------------------------------------
    # 3. Build query matrix (apply length filter to query side only)
    # ------------------------------------------------------------------
    valid_query_ids: list[str] = []
    valid_query_embs: list[list[float]] = []
    for gid, vid in zip(query_ids, query_vdb_ids):
        if not _passes_length_filter(gid, min_entity_len):
            continue
        emb = query_emb_dict.get(vid)
        if emb is None:
            continue
        valid_query_ids.append(gid)
        valid_query_embs.append(emb)

    if not valid_query_embs:
        return 0

    # Build reference matrix (no length filter 鈥?candidates may be short)
    valid_ref_ids: list[str] = []
    valid_ref_embs: list[list[float]] = []
    for gid, vid in zip(all_labels, ref_vdb_ids):
        emb = ref_emb_dict.get(vid)
        if emb is None:
            continue
        valid_ref_ids.append(gid)
        valid_ref_embs.append(emb)

    if not valid_ref_embs:
        return 0

    # ------------------------------------------------------------------
    # 4. L2-normalise so dot product == cosine similarity
    # ------------------------------------------------------------------
    Q = np.array(valid_query_embs, dtype=np.float32)  # (n_query, dim)
    R = np.array(valid_ref_embs,   dtype=np.float32)  # (n_ref,   dim)

    q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    Q /= q_norms

    r_norms = np.linalg.norm(R, axis=1, keepdims=True)
    r_norms[r_norms == 0] = 1.0
    R /= r_norms

    is_symmetric = new_entity_ids is None  # full mode: Q rows == R rows

    # ------------------------------------------------------------------
    # 5. Batched matmul 鈫?collect canonical pairs above threshold
    # ------------------------------------------------------------------
    best_similarity_by_pair: dict[tuple[str, str], float] = {}
    n_query = len(valid_query_ids)

    for batch_start in range(0, n_query, _MATMUL_BATCH):
        batch_end = min(batch_start + _MATMUL_BATCH, n_query)
        q_batch = Q[batch_start:batch_end]  # (B, dim)
        sim_batch = q_batch @ R.T  # (B, n_ref)
        row_local, col_idx = np.where(sim_batch >= synonymy_threshold)

        for r_local, c in zip(row_local.tolist(), col_idx.tolist()):
            r = batch_start + r_local

            # Full mode: only keep upper triangle (pair-level dedupe baseline)
            if is_symmetric and r >= c:
                continue

            entity_id = valid_query_ids[r]
            candidate_id = valid_ref_ids[c]
            if entity_id == candidate_id:
                continue

            src_id, tgt_id = (
                (entity_id, candidate_id)
                if entity_id < candidate_id
                else (candidate_id, entity_id)
            )
            pair_key = (src_id, tgt_id)
            distance = float(sim_batch[r_local, c])
            prev = best_similarity_by_pair.get(pair_key)
            if prev is None or distance > prev:
                best_similarity_by_pair[pair_key] = distance

    if not best_similarity_by_pair:
        return 0

    sorted_pairs = sorted(best_similarity_by_pair.items(), key=lambda item: item[0])
    existing_edges = await knowledge_graph_inst.get_edges_batch(
        [{"src": src_id, "tgt": tgt_id} for (src_id, tgt_id), _ in sorted_pairs]
    )

    edges_to_upsert: list[tuple[str, str, dict[str, Any]]] = []
    for (src_id, tgt_id), distance in sorted_pairs:
        entity_name = src_id.split("|")[0] if "|" in src_id else src_id
        candidate_name = tgt_id.split("|")[0] if "|" in tgt_id else tgt_id
        edge_data: dict[str, Any] = {
            "weight": float(distance),
            "description": f"Synonym: {entity_name} 鈮?{candidate_name}",
            "keywords": "synonym,alias",
            # Keep source_id reserved for real chunk IDs only.
            # Synthetic edge provenance is tracked separately.
            "provenance": SYNONYM_EDGE_PROVENANCE,
            "edge_type": SYNONYM_EDGE_TYPE,
        }

        existing = existing_edges.get((src_id, tgt_id))
        if existing:
            # FACTUAL > SYNONYM: never overwrite factual edges.
            # Untyped legacy extraction edges are also treated as factual.
            if not _is_synonym_edge(existing):
                continue

            existing_weight = existing.get("weight")
            try:
                existing_weight_float = float(existing_weight)
            except (TypeError, ValueError):
                existing_weight_float = None

            if (
                existing_weight_float is not None
                and abs(existing_weight_float - distance) <= 1e-5
                and existing.get("provenance") == edge_data["provenance"]
            ):
                continue

        edges_to_upsert.append((src_id, tgt_id, edge_data))

    if not edges_to_upsert:
        return 0

    await knowledge_graph_inst.upsert_edges_batch(edges_to_upsert)
    created = len(edges_to_upsert)

    logger.info(
        "Synonym linking: upserted %d SYNONYM edges (candidates=%d)",
        created,
        len(best_similarity_by_pair),
    )
    return created
