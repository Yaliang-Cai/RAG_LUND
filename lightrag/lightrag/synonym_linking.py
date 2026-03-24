"""V2: Synonym Linking — build SYNONYM edges between semantically similar entities.

HippoRAG2-aligned: uses pre-computed entity embeddings for KNN search
(not text re-encoding). Each entity's VDB embedding vector is used directly
as the query vector, giving true embedding-space nearest-neighbour semantics.

This module is ingestion-only and fully independent of V3 (PPR multi-hop).
When V3 is enabled, PPR naturally propagates along SYNONYM edges.
"""

from __future__ import annotations

import re
from typing import Any

from lightrag.base import BaseVectorStorage, BaseGraphStorage
from lightrag.utils import logger, compute_mdhash_id


async def build_synonym_edges(
    entities_vdb: BaseVectorStorage,
    knowledge_graph_inst: BaseGraphStorage,
    new_entity_ids: list[str] | None = None,
    synonymy_threshold: float = 0.8,
    synonymy_topk: int = 100,
    min_entity_len: int = 2,
) -> int:
    """Create SYNONYM edges between semantically similar entities.

    Uses embedding-space KNN (aligned with HippoRAG2): retrieves each entity's
    pre-computed embedding vector via ``get_vectors_by_ids()``, then queries the
    VDB with that vector directly — no text re-encoding.

    Args:
        entities_vdb: Vector DB containing entity embeddings.
        knowledge_graph_inst: Graph storage for reading/writing edges.
        new_entity_ids: If given, only process these entities (incremental mode).
            Each ID is a composite graph-node ID (e.g. ``"苹果|COMPANY"``).
        synonymy_threshold: Minimum cosine similarity to create a SYNONYM edge.
        synonymy_topk: How many KNN neighbours to check per entity.
        min_entity_len: Minimum alphanumeric/CJK character count in entity name.
            Entities with fewer characters are skipped (aligned with HippoRAG2's
            ``>2 alphanumeric chars`` filter).

    Returns:
        Number of SYNONYM edges created.
    """
    if new_entity_ids is not None and len(new_entity_ids) == 0:
        return 0

    # Determine target entities
    if new_entity_ids is not None:
        target_ids = new_entity_ids
    else:
        target_ids = await knowledge_graph_inst.get_all_labels()

    if not target_ids:
        return 0

    # Batch-fetch pre-computed embeddings for all target entities
    entity_vdb_ids = [compute_mdhash_id(eid, prefix="ent-") for eid in target_ids]
    embeddings = await entities_vdb.get_vectors_by_ids(entity_vdb_ids)

    created = 0

    for entity_id, vdb_id in zip(target_ids, entity_vdb_ids):
        embedding = embeddings.get(vdb_id)
        if embedding is None:
            continue

        # Entity name length filter (aligned with HippoRAG2: >2 alphanumeric chars)
        entity_name = entity_id.split("|")[0] if "|" in entity_id else entity_id
        if len(re.sub(r'[^A-Za-z0-9\u4e00-\u9fff]', '', entity_name)) <= min_entity_len:
            continue

        # KNN query using the entity's own embedding vector (not text re-encoding)
        try:
            results = await entities_vdb.query(
                "", top_k=synonymy_topk + 1, query_embedding=embedding
            )
        except Exception as e:
            logger.debug(f"Synonym linking: VDB query failed for {entity_id}: {e}")
            continue

        for r in results:
            candidate_id = r.get("entity_id", r.get("entity_name", ""))
            if not candidate_id or candidate_id == entity_id:
                continue

            distance = r.get("distance", 0.0)
            if distance < synonymy_threshold:
                continue

            # Avoid duplicate edges (check both directions)
            if await knowledge_graph_inst.has_edge(entity_id, candidate_id):
                existing = await knowledge_graph_inst.get_edge(entity_id, candidate_id)
                if existing and existing.get("edge_type") == "SYNONYM":
                    continue
            if await knowledge_graph_inst.has_edge(candidate_id, entity_id):
                existing = await knowledge_graph_inst.get_edge(candidate_id, entity_id)
                if existing and existing.get("edge_type") == "SYNONYM":
                    continue

            candidate_name = r.get(
                "entity_name",
                candidate_id.split("|")[0] if "|" in candidate_id else candidate_id,
            )

            edge_data: dict[str, Any] = {
                "weight": float(distance),
                "description": f"Synonym: {entity_name} ≈ {candidate_name}",
                "keywords": "synonym,alias",
                "source_id": "synonym_detection",
                "edge_type": "SYNONYM",
            }

            await knowledge_graph_inst.upsert_edge(entity_id, candidate_id, edge_data)
            created += 1
            logger.debug(
                f"Synonym edge: {entity_id} <-> {candidate_id} (sim={distance:.3f})"
            )

    if created:
        logger.info(f"Synonym linking: created {created} SYNONYM edges")

    return created
