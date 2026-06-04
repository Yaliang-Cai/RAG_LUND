import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_LIGHTRAG_ROOT = PROJECT_ROOT.parent / "lightrag"
if str(LOCAL_LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIGHTRAG_ROOT))

from lightrag.operate import _merge_nodes_then_upsert
from lightrag.utils import compute_entity_id


class _GraphWithExistingNode:
    def __init__(self, node):
        self.node = dict(node)
        self.upserted = None

    async def get_node(self, entity_name):
        return dict(self.node)

    async def upsert_node(self, entity_name, node_data):
        self.upserted = (entity_name, node_data)


@pytest.mark.asyncio
async def test_multimodal_existing_node_replay_preserves_entity_name_for_endpoint_remap():
    modal_name = "Contrastive Loss with Softmax Normalization (equation)"
    modal_type = "equation"
    chunk_id = "chunk-modal"
    composite_id = compute_entity_id(modal_name, modal_type, True)
    graph = _GraphWithExistingNode(
        {
            "entity_id": composite_id,
            "entity_type": modal_type,
            "description": "Existing multimodal main entity.",
            "source_id": chunk_id,
            "file_path": "paper.pdf",
            "created_at": 1,
            "truncate": "",
        }
    )

    result = await _merge_nodes_then_upsert(
        composite_id,
        [
            {
                "entity_name": modal_name,
                "entity_type": modal_type,
                "description": "",
                "source_id": chunk_id,
                "file_path": "paper.pdf",
            }
        ],
        graph,
        entity_vdb=None,
        global_config={
            "source_ids_limit_method": "KEEP",
            "max_source_ids_per_entity": 10,
        },
    )

    assert result["entity_id"] == composite_id
    assert result["entity_name"] == modal_name
    assert graph.upserted is None
