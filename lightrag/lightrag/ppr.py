"""V3: Personalized PageRank for multi-hop reasoning.

HippoRAG2-aligned: supports heterogeneous graphs with entity nodes + virtual
chunk nodes, dual-signal seed weights, and returns chunk-level PPR scores
for direct document ranking.

Pure networkx implementation — no extra dependencies beyond what LightRAG
already ships.  This module is retrieval-only and fully independent of V2
(synonym linking).  When V2 is enabled, SYNONYM edges naturally participate
in PPR propagation.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import networkx as nx
from lightrag.utils import logger


def _is_synonym_edge(edge_data: dict[str, Any]) -> bool:
    """Best-effort synonym edge detection (works for typed and legacy edges)."""
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
    """Map stored graph edge weights to retrieval-time PPR weights."""
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


def personalized_pagerank(
    entity_nodes: list[dict],
    entity_edges: list[dict],
    chunk_nodes: list[dict],
    chunk_entity_edges: list[dict],
    entity_seed_weights: dict[str, float],
    chunk_seed_weights: dict[str, float],
    damping: float = 0.5,
    top_k: int = 50,
    ppr_synonym_weight_mode: Literal["raw", "plus_one"] = "raw",
) -> list[tuple[str, float]]:
    """HippoRAG2-style PPR on entity + virtual chunk graph.

    Builds a heterogeneous graph with entity nodes (from graph storage) and
    virtual chunk nodes (constructed at retrieval time from source_id mappings).
    Dual-signal seed weights drive personalization: relation VDB scores seed
    entity nodes, chunks VDB scores seed chunk nodes.

    Args:
        entity_nodes: List of node dicts, each must contain ``entity_id``.
        entity_edges: List of edge dicts with ``src``, ``tgt``, and optional ``weight``.
        chunk_nodes: List of dicts with ``chunk_id`` for virtual chunk nodes.
        chunk_entity_edges: List of dicts with ``chunk_id`` and ``entity_id``.
        entity_seed_weights: Entity node ID → seed weight (from relation VDB scores).
        chunk_seed_weights: Chunk node ID → seed weight (from chunks VDB × passage_weight).
        damping: PPR damping factor (alpha). Lower = more spread, higher = stay near seeds.
        top_k: Return the top-K chunk nodes by PPR score.

    Returns:
        Sorted list of ``(chunk_id, ppr_score)`` tuples, descending by score.
        Only chunk nodes are included in the output.
    """
    G = nx.Graph()
    chunk_node_ids = set()

    # Add entity nodes
    for node in entity_nodes:
        nid = node.get("entity_id")
        if nid:
            G.add_node(nid, node_type="entity")

    # Add entity-entity edges
    for edge in entity_edges:
        src, tgt = edge.get("src"), edge.get("tgt")
        if src and tgt:
            edge_weight = _map_entity_edge_weight(edge, ppr_synonym_weight_mode)
            G.add_edge(src, tgt, weight=edge_weight)

    # Add virtual chunk nodes
    for chunk in chunk_nodes:
        cid = chunk.get("chunk_id")
        if cid:
            G.add_node(cid, node_type="chunk")
            chunk_node_ids.add(cid)

    # Add chunk-entity edges (weight=1.0, matching HippoRAG2's passage-entity edges)
    for ce in chunk_entity_edges:
        cid, eid = ce.get("chunk_id"), ce.get("entity_id")
        if cid and eid and cid in G and eid in G:
            G.add_edge(cid, eid, weight=1.0)

    if G.number_of_nodes() == 0:
        return []

    # Build combined seed weights (dual signal)
    personalization = {}
    for nid, w in entity_seed_weights.items():
        if nid in G:
            personalization[nid] = max(w, 1e-8)
    for cid, w in chunk_seed_weights.items():
        if cid in G:
            personalization[cid] = max(w, 1e-8)

    if not personalization:
        return []

    # Normalize
    total = sum(personalization.values())
    personalization = {k: v / total for k, v in personalization.items()}

    try:
        pr = nx.pagerank(G, alpha=damping, personalization=personalization, weight="weight")
    except nx.PowerIterationFailedConvergence:
        logger.warning("PPR did not converge; returning seed chunks only")
        seed_chunks = [
            (cid, w) for cid, w in chunk_seed_weights.items() if cid in chunk_node_ids
        ]
        seed_chunks.sort(key=lambda x: x[1], reverse=True)
        return seed_chunks[:top_k]

    # Extract ONLY chunk node scores (matching HippoRAG2: only passage scores matter)
    chunk_scores = [(nid, score) for nid, score in pr.items() if nid in chunk_node_ids]
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    return chunk_scores[:top_k]


def personalized_pagerank_simple(
    nodes: list[dict],
    edges: list[dict],
    seed_weights: dict[str, float],
    damping: float = 0.5,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Backward-compatible simple PPR on an entity-only subgraph.

    This is the original V3 signature, kept for basic usage scenarios.

    Args:
        nodes: List of node dicts, each must contain ``entity_id``.
        edges: List of edge dicts with ``src``, ``tgt``, and optional ``weight``.
        seed_weights: Mapping of seed node IDs → initial weight.
        damping: PPR damping factor (alpha).
        top_k: Return the top-K nodes by PPR score.

    Returns:
        Sorted list of ``(entity_id, ppr_score)`` tuples, descending by score.
    """
    G = nx.Graph()

    for node in nodes:
        nid = node.get("entity_id")
        if nid:
            G.add_node(nid)

    for edge in edges:
        src, tgt = edge.get("src"), edge.get("tgt")
        if src and tgt:
            w = float(edge.get("weight", 1.0))
            G.add_edge(src, tgt, weight=w)

    if G.number_of_nodes() == 0:
        return []

    personalization: dict[str, float] = {}
    for nid, w in seed_weights.items():
        if nid in G:
            personalization[nid] = max(w, 1e-8)

    if not personalization:
        return []

    total = sum(personalization.values())
    personalization = {nid: w / total for nid, w in personalization.items()}

    try:
        pr = nx.pagerank(G, alpha=damping, personalization=personalization, weight="weight")
    except nx.PowerIterationFailedConvergence:
        logger.warning("PPR did not converge; returning seed nodes only")
        return sorted(personalization.items(), key=lambda x: x[1], reverse=True)[:top_k]

    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
