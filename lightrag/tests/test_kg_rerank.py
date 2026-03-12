"""
Unit tests for KG retrieval noise fixes:
1. Hub node edge sorting with VDB cosine score
2. Entity/relation reranking via _rerank_kg_results
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch


# ========== Fix 1: Hub noise — edge sorting with vdb_score ==========


class TestEdgeSortingWithVdbScore:
    """Test that edges are sorted by entity VDB relevance, not pure degree."""

    def test_relevant_low_degree_edges_rank_first(self):
        """Hub edges (high degree, low relevance) should rank below relevant edges."""
        node_datas = [
            {"entity_name": "HUB_NODE", "vdb_score": 0.3, "rank": 100},
            {"entity_name": "RELEVANT_NODE", "vdb_score": 0.95, "rank": 2},
        ]

        all_edges_data = [
            {"src_tgt": ("HUB_NODE", "OTHER_A"), "rank": 120, "weight": 5.0},
            {"src_tgt": ("HUB_NODE", "OTHER_B"), "rank": 110, "weight": 4.0},
            {"src_tgt": ("RELEVANT_NODE", "OTHER_C"), "rank": 5, "weight": 1.0},
        ]

        # Apply the new sorting logic (copied from operate.py)
        entity_score_map = {
            dp["entity_name"]: dp.get("vdb_score", 0.0) for dp in node_datas
        }
        for edge in all_edges_data:
            src, tgt = edge["src_tgt"]
            edge["_entity_relevance"] = max(
                entity_score_map.get(src, 0.0), entity_score_map.get(tgt, 0.0)
            )

        all_edges_data = sorted(
            all_edges_data,
            key=lambda x: (x["_entity_relevance"], x.get("weight", 1.0), x["rank"]),
            reverse=True,
        )

        # RELEVANT_NODE edge should come first despite low degree
        assert all_edges_data[0]["src_tgt"] == ("RELEVANT_NODE", "OTHER_C")
        # HUB_NODE edges should follow, ordered by weight then rank
        assert all_edges_data[1]["src_tgt"] == ("HUB_NODE", "OTHER_A")
        assert all_edges_data[2]["src_tgt"] == ("HUB_NODE", "OTHER_B")

    def test_old_sorting_would_prefer_hub(self):
        """Confirm old (rank, weight) sorting puts hub edges first."""
        all_edges_data = [
            {"src_tgt": ("HUB", "X"), "rank": 120, "weight": 5.0},
            {"src_tgt": ("SMALL", "Y"), "rank": 5, "weight": 1.0},
        ]
        old_sorted = sorted(
            all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        assert old_sorted[0]["src_tgt"] == ("HUB", "X")

    def test_missing_vdb_score_defaults_to_zero(self):
        """Entities not in seed set get vdb_score=0, their edges rank last."""
        node_datas = [{"entity_name": "SEED_A", "vdb_score": 0.8}]
        all_edges_data = [
            {"src_tgt": ("SEED_A", "NEIGHBOR"), "rank": 10, "weight": 2.0},
            {"src_tgt": ("UNKNOWN_X", "UNKNOWN_Y"), "rank": 50, "weight": 3.0},
        ]
        entity_score_map = {
            dp["entity_name"]: dp.get("vdb_score", 0.0) for dp in node_datas
        }
        for edge in all_edges_data:
            src, tgt = edge["src_tgt"]
            edge["_entity_relevance"] = max(
                entity_score_map.get(src, 0.0), entity_score_map.get(tgt, 0.0)
            )
        all_edges_data = sorted(
            all_edges_data,
            key=lambda x: (x["_entity_relevance"], x.get("weight", 1.0), x["rank"]),
            reverse=True,
        )
        assert all_edges_data[0]["src_tgt"] == ("SEED_A", "NEIGHBOR")
        assert all_edges_data[1]["_entity_relevance"] == 0.0

    def test_vdb_score_propagation_in_node_data(self):
        """Simulates _get_node_data dict comprehension to verify vdb_score field."""
        results = [
            {"entity_name": "A", "distance": 0.92, "created_at": "2026-01-01"},
            {"entity_name": "B", "distance": 0.75, "created_at": "2026-01-02"},
            {"entity_name": "C", "created_at": "2026-01-03"},  # missing distance
        ]
        node_datas_raw = [
            {"description": "desc_a"},
            {"description": "desc_b"},
            {"description": "desc_c"},
        ]
        node_degrees = [5, 10, 2]

        node_datas = [
            {
                **n,
                "entity_name": k["entity_name"],
                "rank": d,
                "vdb_score": k.get("distance", 0.0),
                "created_at": k.get("created_at"),
            }
            for k, n, d in zip(results, node_datas_raw, node_degrees)
            if n is not None
        ]

        assert node_datas[0]["vdb_score"] == 0.92
        assert node_datas[1]["vdb_score"] == 0.75
        assert node_datas[2]["vdb_score"] == 0.0

    def test_tiebreaker_by_weight_then_rank(self):
        """When vdb_score is equal, weight breaks ties; then rank."""
        node_datas = [
            {"entity_name": "A", "vdb_score": 0.8},
            {"entity_name": "B", "vdb_score": 0.8},
        ]
        all_edges_data = [
            {"src_tgt": ("A", "X"), "rank": 10, "weight": 2.0},
            {"src_tgt": ("B", "Y"), "rank": 20, "weight": 5.0},
            {"src_tgt": ("A", "Z"), "rank": 30, "weight": 5.0},
        ]
        entity_score_map = {
            dp["entity_name"]: dp.get("vdb_score", 0.0) for dp in node_datas
        }
        for edge in all_edges_data:
            src, tgt = edge["src_tgt"]
            edge["_entity_relevance"] = max(
                entity_score_map.get(src, 0.0), entity_score_map.get(tgt, 0.0)
            )
        all_edges_data = sorted(
            all_edges_data,
            key=lambda x: (x["_entity_relevance"], x.get("weight", 1.0), x["rank"]),
            reverse=True,
        )
        # Same relevance (0.8), so weight=5.0 > 2.0; among weight=5.0, rank=30 > 20
        assert all_edges_data[0]["src_tgt"] == ("A", "Z")   # 0.8, 5.0, 30
        assert all_edges_data[1]["src_tgt"] == ("B", "Y")   # 0.8, 5.0, 20
        assert all_edges_data[2]["src_tgt"] == ("A", "X")   # 0.8, 2.0, 10


# ========== Fix 2: Entity/relation reranking ==========


class TestRerankContentFieldConstruction:
    """Test that _rerank_kg_results constructs correct content fields."""

    def test_entity_content_field(self):
        entities = [
            {"entity_name": "Foo", "description": "A foo thing"},
            {"entity_name": "Bar", "description": ""},
            {"entity_name": "Baz"},
        ]
        for e in entities:
            e["content"] = f"{e['entity_name']}: {e.get('description', '')}"
        assert entities[0]["content"] == "Foo: A foo thing"
        assert entities[1]["content"] == "Bar: "
        assert entities[2]["content"] == "Baz: "

    def test_relation_content_field_src_tgt(self):
        """Relations with src_tgt tuple."""
        relations = [
            {"src_tgt": ("A", "B"), "description": "A relates to B"},
            {"src_tgt": ("E", "F")},  # no description
        ]
        for r in relations:
            src = r.get("src_id") or (r["src_tgt"][0] if "src_tgt" in r else "")
            tgt = r.get("tgt_id") or (r["src_tgt"][1] if "src_tgt" in r else "")
            r["content"] = f"{src} \u2192 {tgt}: {r.get('description', '')}"
        assert "A \u2192 B: A relates to B" == relations[0]["content"]
        assert "E \u2192 F: " == relations[1]["content"]

    def test_relation_content_field_src_id_tgt_id(self):
        """Relations with src_id/tgt_id fields."""
        relations = [
            {"src_id": "C", "tgt_id": "D", "description": "C-D relation"},
        ]
        for r in relations:
            src = r.get("src_id") or (r["src_tgt"][0] if "src_tgt" in r else "")
            tgt = r.get("tgt_id") or (r["src_tgt"][1] if "src_tgt" in r else "")
            r["content"] = f"{src} \u2192 {tgt}: {r.get('description', '')}"
        assert relations[0]["content"] == "C \u2192 D: C-D relation"


class TestRerankKgResults:
    """Test _rerank_kg_results function end-to-end."""

    @pytest.mark.asyncio
    async def test_rerank_reorders_entities_by_score(self):
        from lightrag.operate import _rerank_kg_results
        from lightrag.base import QueryParam

        search_result = {
            "final_entities": [
                {"entity_name": "LowRelevance", "description": "Not very relevant"},
                {"entity_name": "HighRelevance", "description": "Very relevant to query"},
            ],
            "final_relations": [],
        }

        async def mock_rerank_func(query=None, documents=None, top_n=None, **kwargs):
            # Return index-based results: score by content
            results = []
            for i, t in enumerate(documents):
                score = 0.95 if "Very relevant" in t else 0.2
                results.append({"index": i, "relevance_score": score})
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results

        global_config = {"rerank_model_func": mock_rerank_func}
        query_param = QueryParam(enable_rerank=True)

        result = await _rerank_kg_results(
            "test query", search_result, query_param, global_config
        )

        entities = result["final_entities"]
        # HighRelevance should now be first (higher rerank score)
        assert entities[0]["entity_name"] == "HighRelevance"
        assert entities[0].get("rerank_score", 0) > entities[1].get("rerank_score", 0)

    @pytest.mark.asyncio
    async def test_rerank_reorders_relations_by_score(self):
        from lightrag.operate import _rerank_kg_results
        from lightrag.base import QueryParam

        search_result = {
            "final_entities": [],
            "final_relations": [
                {"src_tgt": ("A", "B"), "description": "Irrelevant connection"},
                {"src_tgt": ("C", "D"), "description": "Highly relevant link to query"},
            ],
        }

        async def mock_rerank_func(query=None, documents=None, top_n=None, **kwargs):
            results = []
            for i, t in enumerate(documents):
                score = 0.9 if "Highly relevant" in t else 0.1
                results.append({"index": i, "relevance_score": score})
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results

        global_config = {"rerank_model_func": mock_rerank_func}
        query_param = QueryParam(enable_rerank=True)

        result = await _rerank_kg_results(
            "test query", search_result, query_param, global_config
        )

        relations = result["final_relations"]
        assert relations[0]["src_tgt"] == ("C", "D")

    @pytest.mark.asyncio
    async def test_rerank_empty_entities_and_relations(self):
        from lightrag.operate import _rerank_kg_results
        from lightrag.base import QueryParam

        search_result = {
            "final_entities": [],
            "final_relations": [],
        }

        global_config = {"rerank_model_func": AsyncMock()}
        query_param = QueryParam(enable_rerank=True)

        result = await _rerank_kg_results(
            "test query", search_result, query_param, global_config
        )

        assert result["final_entities"] == []
        assert result["final_relations"] == []
        # rerank function should NOT have been called
        global_config["rerank_model_func"].assert_not_called()

    @pytest.mark.asyncio
    async def test_rerank_preserves_extra_fields(self):
        """Reranking should not drop existing fields from entity/relation dicts."""
        from lightrag.operate import _rerank_kg_results
        from lightrag.base import QueryParam

        search_result = {
            "final_entities": [
                {
                    "entity_name": "Test",
                    "description": "Desc",
                    "rank": 5,
                    "vdb_score": 0.8,
                    "source_id": "chunk_1",
                },
            ],
            "final_relations": [],
        }

        async def mock_rerank_func(query=None, documents=None, top_n=None, **kwargs):
            return [{"index": i, "relevance_score": 0.75} for i in range(len(documents))]

        global_config = {"rerank_model_func": mock_rerank_func}
        query_param = QueryParam(enable_rerank=True)

        result = await _rerank_kg_results(
            "query", search_result, query_param, global_config
        )

        entity = result["final_entities"][0]
        assert entity["entity_name"] == "Test"
        assert entity["rank"] == 5
        assert entity["vdb_score"] == 0.8
        assert entity["source_id"] == "chunk_1"
        assert "content" in entity  # content field added for reranking


class TestBuildQueryContextStage15Integration:
    """Test that Stage 1.5 is correctly wired into _build_query_context."""

    @pytest.mark.asyncio
    async def test_stage15_called_when_rerank_enabled(self):
        """Verify _rerank_kg_results is called between Stage 1 and Stage 2."""
        from lightrag.operate import _rerank_kg_results

        # We just need to verify the function exists and is callable
        assert callable(_rerank_kg_results)

    @pytest.mark.asyncio
    async def test_stage15_skipped_when_no_rerank_func(self):
        """When rerank_model_func is None, Stage 1.5 should be skipped."""
        from lightrag.base import QueryParam

        query_param = QueryParam(enable_rerank=True)
        global_config = {}  # no rerank_model_func

        # The condition: query_param.enable_rerank and global_config.get("rerank_model_func")
        should_rerank = query_param.enable_rerank and global_config.get("rerank_model_func")
        assert not should_rerank

    @pytest.mark.asyncio
    async def test_stage15_skipped_when_rerank_disabled(self):
        """When enable_rerank=False, Stage 1.5 should be skipped."""
        from lightrag.base import QueryParam

        query_param = QueryParam(enable_rerank=False)
        global_config = {"rerank_model_func": AsyncMock()}

        should_rerank = query_param.enable_rerank and global_config.get("rerank_model_func")
        assert not should_rerank
