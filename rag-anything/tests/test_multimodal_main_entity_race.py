import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_LIGHTRAG_ROOT = PROJECT_ROOT.parent / "lightrag"
if str(LOCAL_LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIGHTRAG_ROOT))

from lightrag.constants import GRAPH_FIELD_SEP
from raganything.processor import ProcessorMixin


class _GraphStore:
    def __init__(self) -> None:
        self.nodes = {}
        self.waiters = 0
        self.write_gate = asyncio.Event()

    async def get_node(self, entity_id):
        node = self.nodes.get(entity_id)
        return dict(node) if isinstance(node, dict) else None

    async def upsert_node(self, entity_id, node_data):
        self.waiters += 1
        try:
            await self.write_gate.wait()
            self.nodes[entity_id] = dict(node_data)
        finally:
            self.waiters -= 1


class _EntityChunksStore:
    def __init__(self) -> None:
        self.records = {}

    async def get_by_id(self, entity_id):
        record = self.records.get(entity_id)
        return dict(record) if isinstance(record, dict) else None

    async def upsert(self, payload):
        for entity_id, value in payload.items():
            self.records[entity_id] = dict(value)


class _EntityVDB:
    def __init__(self) -> None:
        self.records = {}
        self.index_done_calls = 0

    async def upsert(self, payload):
        for entity_id, value in payload.items():
            self.records[entity_id] = dict(value)

    async def index_done_callback(self):
        self.index_done_calls += 1


class _ProcessorUnderTest(ProcessorMixin):
    def __init__(self, lightrag):
        self.lightrag = lightrag


@pytest.mark.asyncio
async def test_multimodal_main_entity_updates_merge_source_ids_without_lost_update(
    monkeypatch: pytest.MonkeyPatch,
):
    import raganything.processor as processor_module

    keyed_locks = {}

    @asynccontextmanager
    async def _keyed_lock(keys, namespace="default", enable_logging=False):
        if isinstance(keys, str):
            normalized_keys = (keys,)
        else:
            normalized_keys = tuple(keys)
        lock = keyed_locks.setdefault((namespace, normalized_keys), asyncio.Lock())
        async with lock:
            yield

    monkeypatch.setattr(
        processor_module, "get_storage_keyed_lock", _keyed_lock, raising=False
    )

    graph = _GraphStore()
    entity_chunks = _EntityChunksStore()
    entities_vdb = _EntityVDB()
    lightrag = SimpleNamespace(
        workspace="race_ws",
        enable_entity_disambiguation=True,
        max_source_ids_per_entity=0,
        source_ids_limit_method="KEEP",
        chunk_entity_relation_graph=graph,
        entity_chunks=entity_chunks,
        entities_vdb=entities_vdb,
    )
    processor = _ProcessorUnderTest(lightrag)

    shared_entity_id = "mm-entity-1"
    first = {
        "entity_id": shared_entity_id,
        "entity_name": "Figure 1 (image)",
        "entity_type": "image",
        "content": "first description",
        "file_path": "doc_a.pdf",
        "source_id": "chunk-a",
    }
    second = {
        "entity_id": shared_entity_id,
        "entity_name": "Figure 1 (image)",
        "entity_type": "image",
        "content": "second description",
        "file_path": "doc_b.pdf",
        "source_id": "chunk-b",
    }

    task_a = asyncio.create_task(
        processor._upsert_multimodal_main_entities_to_core_storage({"vdb-a": first})
    )
    task_b = asyncio.create_task(
        processor._upsert_multimodal_main_entities_to_core_storage({"vdb-b": second})
    )

    for _ in range(200):
        if graph.waiters >= 2:
            break
        await asyncio.sleep(0)

    graph.write_gate.set()
    await asyncio.gather(task_a, task_b)

    stored_node = graph.nodes[shared_entity_id]
    stored_chunks = entity_chunks.records[shared_entity_id]

    assert set(stored_node["source_id"].split(GRAPH_FIELD_SEP)) == {
        "chunk-a",
        "chunk-b",
    }
    assert set(stored_chunks["chunk_ids"]) == {"chunk-a", "chunk-b"}
    assert set(entities_vdb.records) == {"vdb-a", "vdb-b"}
    assert entities_vdb.index_done_calls == 2
