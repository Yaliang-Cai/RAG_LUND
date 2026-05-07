import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIGHTRAG_ROOT = PROJECT_ROOT / "lightrag"
if str(LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTRAG_ROOT))

from lightrag import QueryParam
from lightrag.operate import _should_exclude_synonym_edges
from lightrag.ppr_engine import GlobalPPREngine


def test_synonym_filter_explicit_values_override_mode_auto_defaults():
    assert (
        _should_exclude_synonym_edges(
            QueryParam(mode="hybrid", exclude_synonym_edges=True)
        )
        is True
    )
    assert (
        _should_exclude_synonym_edges(
            QueryParam(mode="ppr", exclude_synonym_edges=False)
        )
        is False
    )
    assert (
        _should_exclude_synonym_edges(
            QueryParam(mode="hybrid", exclude_synonym_edges=False)
        )
        is False
    )
    assert (
        _should_exclude_synonym_edges(QueryParam(mode="ppr", exclude_synonym_edges=True))
        is True
    )

    assert (
        _should_exclude_synonym_edges(
            QueryParam(mode="hybrid", exclude_synonym_edges=None)
        )
        is True
    )
    assert (
        _should_exclude_synonym_edges(QueryParam(mode="ppr", exclude_synonym_edges=None))
        is False
    )


def test_global_ppr_engine_filters_synonym_edges_from_actual_adjacency():
    engine = GlobalPPREngine.__new__(GlobalPPREngine)
    engine.index_node = ["seed", "factual", "synonym"]
    engine._adj_by_mode = {}
    engine._entity_edges_for_ppr = [
        (
            0,
            1,
            {
                "weight": 1.0,
                "edge_type": "FACTUAL",
                "provenance": "relation_extraction",
                "source_id": "chunk-factual",
            },
        ),
        (
            0,
            2,
            {
                "weight": 1.0,
                "edge_type": "SYNONYM",
                "provenance": "synonym_detection",
                "keywords": "synonym,alias",
                "source_id": "",
            },
        ),
    ]

    include_adj = engine._get_entity_adj("raw", exclude_synonym_edges=False)
    exclude_adj = engine._get_entity_adj("raw", exclude_synonym_edges=True)

    assert include_adj.nnz == 4
    assert include_adj[0, 2] > 0
    assert include_adj[2, 0] > 0
    assert exclude_adj.nnz == 2
    assert exclude_adj[0, 2] == 0
    assert exclude_adj[2, 0] == 0
    assert ("raw", False) in engine._adj_by_mode
    assert ("raw", True) in engine._adj_by_mode
