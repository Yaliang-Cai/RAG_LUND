import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAG_ANYTHING_ROOT = PROJECT_ROOT.parent / "rag-anything"
if str(RAG_ANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ANYTHING_ROOT))

from lightrag.base import DocStatus
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
    initialize_pipeline_status,
    initialize_share_data,
    set_default_workspace,
)
from lightrag.lightrag import LightRAG
import lightrag.lightrag as lightrag_module
from lightrag.operate import merge_nodes_and_edges
from lightrag.synonym_linking import build_synonym_edges
from lightrag.utils import (
    compute_entity_id,
    compute_entity_vdb_id,
    compute_mdhash_id,
    make_relation_chunk_key,
)

from raganything.processor import ProcessorMixin  # noqa: E402


pytestmark = pytest.mark.offline


class _NoopLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _DummyTokenizer:
    def encode(self, text: str) -> list[str]:
        return list(text or "")


class _InMemoryKVStorage:
    def __init__(self):
        self.data: dict[str, dict[str, Any]] = {}

    async def get_by_id(self, key: str):
        value = self.data.get(key)
        return copy.deepcopy(value) if value is not None else None

    async def get_by_ids(self, keys: list[str]):
        return [copy.deepcopy(self.data.get(key)) for key in keys]

    async def upsert(self, payload: dict[str, dict[str, Any]]):
        for key, value in payload.items():
            self.data[key] = copy.deepcopy(value)

    async def delete(self, keys: list[str]):
        for key in keys:
            self.data.pop(key, None)

    async def index_done_callback(self):
        return None


class _InMemoryVectorStorage:
    def __init__(self):
        self.payloads: dict[str, dict[str, Any]] = {}
        self.vectors: dict[str, list[float]] = {}

    async def upsert(self, payload: dict[str, dict[str, Any]]):
        for key, value in payload.items():
            self.payloads[key] = copy.deepcopy(value)

    async def delete(self, ids: list[str]):
        for item_id in ids:
            self.payloads.pop(item_id, None)
            self.vectors.pop(item_id, None)

    async def get_by_ids(self, ids: list[str]):
        return [copy.deepcopy(self.payloads.get(item_id)) for item_id in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        return {
            item_id: copy.deepcopy(self.vectors[item_id])
            for item_id in ids
            if item_id in self.vectors
        }

    async def index_done_callback(self):
        return None


class _InMemoryGraphStorage:
    def __init__(self):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _edge_key(src: str, tgt: str) -> tuple[str, str]:
        return tuple(sorted((src, tgt)))

    async def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    async def get_node(self, node_id: str):
        value = self.nodes.get(node_id)
        return copy.deepcopy(value) if value is not None else None

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        self.nodes[node_id] = copy.deepcopy(node_data)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {
            node_id: copy.deepcopy(self.nodes[node_id])
            for node_id in node_ids
            if node_id in self.nodes
        }

    async def get_all_labels(self) -> list[str]:
        return list(self.nodes.keys())

    async def has_edge(self, src: str, tgt: str) -> bool:
        return self._edge_key(src, tgt) in self.edges

    async def get_edge(self, src: str, tgt: str):
        key = self._edge_key(src, tgt)
        value = self.edges.get(key)
        if value is None:
            return None
        return copy.deepcopy(value)

    async def get_edges_batch(
        self, edge_pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for pair in edge_pairs:
            src = pair.get("src")
            tgt = pair.get("tgt")
            if not src or not tgt:
                continue
            key = self._edge_key(src, tgt)
            if key in self.edges:
                result[key] = copy.deepcopy(self.edges[key])
        return result

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict[str, Any]):
        key = self._edge_key(src, tgt)
        payload = copy.deepcopy(edge_data)
        payload.setdefault("source", key[0])
        payload.setdefault("target", key[1])
        self.edges[key] = payload

    async def upsert_edges_batch(self, edges: list[tuple[str, str, dict[str, Any]]]):
        for src, tgt, edge_data in edges:
            await self.upsert_edge(src, tgt, edge_data)

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        result = []
        for src, tgt in self.edges.keys():
            if node_id in (src, tgt):
                result.append((src, tgt))
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        return {node_id: await self.get_node_edges(node_id) for node_id in node_ids}

    async def remove_edges(self, edges: list[tuple[str, str]]):
        for src, tgt in edges:
            self.edges.pop(self._edge_key(src, tgt), None)

    async def remove_nodes(self, node_ids: list[str]):
        for node_id in node_ids:
            self.nodes.pop(node_id, None)
        for key in list(self.edges.keys()):
            if key[0] in node_ids or key[1] in node_ids:
                self.edges.pop(key, None)


class _DummyProcessor(ProcessorMixin):
    def __init__(self, lightrag_obj):
        self.lightrag = lightrag_obj
        self.config = SimpleNamespace(use_full_path=False)
        self.logger = _NoopLogger()


class _DummyLightRAG:
    adelete_by_doc_id = LightRAG.adelete_by_doc_id
    _update_delete_retry_state = LightRAG._update_delete_retry_state
    _get_existing_llm_cache_ids = LightRAG._get_existing_llm_cache_ids

    async def _insert_done(self):
        return None


async def _dummy_llm(*args, **kwargs):
    return "summary"


async def _append_doc_status_chunks(
    doc_status_storage: _InMemoryKVStorage,
    doc_id: str,
    file_path: str,
    new_chunk_ids: list[str],
):
    current = await doc_status_storage.get_by_id(doc_id)
    if current is None:
        merged_chunks = list(dict.fromkeys(new_chunk_ids))
        payload = {
            "status": DocStatus.PROCESSED.value,
            "file_path": file_path,
            "chunks_list": merged_chunks,
            "chunks_count": len(merged_chunks),
            "metadata": {},
            "created_at": "2026-04-16T00:00:00+00:00",
            "updated_at": "2026-04-16T00:00:00+00:00",
        }
    else:
        merged_chunks = list(dict.fromkeys([*(current.get("chunks_list", [])), *new_chunk_ids]))
        payload = {
            **current,
            "chunks_list": merged_chunks,
            "chunks_count": len(merged_chunks),
            "updated_at": "2026-04-16T00:00:00+00:00",
        }
    await doc_status_storage.upsert({doc_id: payload})


def _build_global_config(workspace: str) -> dict[str, Any]:
    return {
        "workspace": workspace,
        "enable_entity_disambiguation": True,
        "llm_model_max_async": 4,
        "tokenizer": _DummyTokenizer(),
        "summary_context_size": 4096,
        "summary_max_tokens": 512,
        "summary_length_recommended": 128,
        "force_llm_summary_on_merge": 3,
        "llm_model_func": _dummy_llm,
        "addon_params": {},
        "source_ids_limit_method": "KEEP",
        "max_source_ids_per_entity": 20,
        "max_source_ids_per_relation": 20,
        "max_file_paths": 20,
        "file_path_more_placeholder": "files omitted",
    }


async def _ingest_factual_doc(
    *,
    doc_id: str,
    file_path: str,
    src_name: str,
    src_type: str,
    tgt_name: str,
    tgt_type: str,
    relation_keywords: str,
    relation_description: str,
    graph: _InMemoryGraphStorage,
    entities_vdb: _InMemoryVectorStorage,
    relationships_vdb: _InMemoryVectorStorage,
    text_chunks: _InMemoryKVStorage,
    chunks_vdb: _InMemoryVectorStorage,
    full_docs: _InMemoryKVStorage,
    doc_status: _InMemoryKVStorage,
    full_entities: _InMemoryKVStorage,
    full_relations: _InMemoryKVStorage,
    entity_chunks: _InMemoryKVStorage,
    relation_chunks: _InMemoryKVStorage,
    pipeline_status: dict[str, Any],
    pipeline_status_lock,
    global_config: dict[str, Any],
) -> dict[str, Any]:
    chunk_content = f"{doc_id}:{src_name}-{tgt_name}:{relation_description}"
    chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")
    chunk_payload = {
        "content": chunk_content,
        "full_doc_id": doc_id,
        "file_path": file_path,
        "llm_cache_list": [],
        "tokens": len(chunk_content),
    }
    await text_chunks.upsert({chunk_id: chunk_payload})
    await chunks_vdb.upsert({chunk_id: chunk_payload})
    await full_docs.upsert({doc_id: {"content": f"doc content: {doc_id}"}})
    await _append_doc_status_chunks(doc_status, doc_id, file_path, [chunk_id])

    src_id = compute_entity_id(src_name, src_type, True)
    tgt_id = compute_entity_id(tgt_name, tgt_type, True)
    maybe_nodes = {
        src_name: [
            {
                "entity_name": src_name,
                "entity_type": src_type,
                "description": f"{src_name} description",
                "source_id": chunk_id,
                "file_path": file_path,
                "timestamp": 1,
            }
        ],
        tgt_name: [
            {
                "entity_name": tgt_name,
                "entity_type": tgt_type,
                "description": f"{tgt_name} description",
                "source_id": chunk_id,
                "file_path": file_path,
                "timestamp": 1,
            }
        ],
    }
    maybe_edges = {
        tuple(sorted((src_name, tgt_name))): [
            {
                "description": relation_description,
                "keywords": relation_keywords,
                "weight": 1.0,
                "source_id": chunk_id,
                "file_path": file_path,
                "timestamp": 1,
            }
        ]
    }
    await merge_nodes_and_edges(
        chunk_results=[(maybe_nodes, maybe_edges)],
        knowledge_graph_inst=graph,
        entity_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        global_config=global_config,
        full_entities_storage=full_entities,
        full_relations_storage=full_relations,
        doc_id=doc_id,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=None,
        entity_chunks_storage=entity_chunks,
        relation_chunks_storage=relation_chunks,
        current_file_number=1,
        total_files=1,
        file_path=file_path,
    )
    return {
        "chunk_id": chunk_id,
        "src_id": src_id,
        "tgt_id": tgt_id,
    }


@pytest.mark.asyncio
async def test_v1_relation_endpoint_remap_uses_chunk_local_entity_type():
    workspace = "test-v1-chunk-local-remap"
    initialize_share_data(workers=1)
    set_default_workspace(workspace)
    await initialize_pipeline_status(workspace=workspace)
    pipeline_status = await get_namespace_data("pipeline_status", workspace=workspace)
    pipeline_status_lock = get_pipeline_status_lock(workspace=workspace)

    graph = _InMemoryGraphStorage()
    entities_vdb = _InMemoryVectorStorage()
    relationships_vdb = _InMemoryVectorStorage()
    full_entities = _InMemoryKVStorage()
    full_relations = _InMemoryKVStorage()
    entity_chunks = _InMemoryKVStorage()
    relation_chunks = _InMemoryKVStorage()
    global_config = _build_global_config(workspace)
    global_config["strict_relation_endpoint_entity_match"] = True

    chunk_planet = "chunk-mercury-planet"
    chunk_element = "chunk-mercury-element"
    mercury_planet = compute_entity_id("Mercury", "planet", True)
    mercury_element = compute_entity_id("Mercury", "element", True)
    venus_planet = compute_entity_id("Venus", "planet", True)
    gold_element = compute_entity_id("Gold", "element", True)

    chunk_results = [
        (
            {
                "Mercury": [
                    {
                        "entity_name": "Mercury",
                        "entity_type": "planet",
                        "description": "Mercury as a planet.",
                        "source_id": chunk_planet,
                        "file_path": "doc.md",
                    }
                ],
                "Venus": [
                    {
                        "entity_name": "Venus",
                        "entity_type": "planet",
                        "description": "Venus as a planet.",
                        "source_id": chunk_planet,
                        "file_path": "doc.md",
                    }
                ],
            },
            {
                tuple(sorted(("Mercury", "Venus"))): [
                    {
                        "description": "Mercury and Venus are inner planets.",
                        "keywords": "planetary_neighbor",
                        "weight": 1.0,
                        "source_id": chunk_planet,
                        "file_path": "doc.md",
                    }
                ]
            },
        ),
        (
            {
                "Mercury": [
                    {
                        "entity_name": "Mercury",
                        "entity_type": "element",
                        "description": "Mercury as a chemical element.",
                        "source_id": chunk_element,
                        "file_path": "doc.md",
                    }
                ],
                "Gold": [
                    {
                        "entity_name": "Gold",
                        "entity_type": "element",
                        "description": "Gold as a chemical element.",
                        "source_id": chunk_element,
                        "file_path": "doc.md",
                    }
                ],
            },
            {
                tuple(sorted(("Mercury", "Gold"))): [
                    {
                        "description": "Mercury and gold are chemical elements.",
                        "keywords": "chemical_element",
                        "weight": 1.0,
                        "source_id": chunk_element,
                        "file_path": "doc.md",
                    }
                ]
            },
        ),
    ]

    await merge_nodes_and_edges(
        chunk_results=chunk_results,
        knowledge_graph_inst=graph,
        entity_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        global_config=global_config,
        full_entities_storage=full_entities,
        full_relations_storage=full_relations,
        doc_id="doc-mercury",
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=None,
        entity_chunks_storage=entity_chunks,
        relation_chunks_storage=relation_chunks,
        current_file_number=1,
        total_files=1,
        file_path="doc.md",
    )

    assert set(graph.nodes) == {
        mercury_planet,
        venus_planet,
        mercury_element,
        gold_element,
    }
    assert set(graph.edges) == {
        tuple(sorted((mercury_planet, venus_planet))),
        tuple(sorted((mercury_element, gold_element))),
    }


@pytest.mark.asyncio
async def test_mocked_three_doc_write_delete_and_edge_semantics():
    workspace = "test-doc-delete-mocked"
    initialize_share_data(workers=1)
    set_default_workspace(workspace)
    await initialize_pipeline_status(workspace=workspace)
    pipeline_status = await get_namespace_data("pipeline_status", workspace=workspace)
    pipeline_status_lock = get_pipeline_status_lock(workspace=workspace)

    graph = _InMemoryGraphStorage()
    entities_vdb = _InMemoryVectorStorage()
    relationships_vdb = _InMemoryVectorStorage()
    chunks_vdb = _InMemoryVectorStorage()
    text_chunks = _InMemoryKVStorage()
    full_docs = _InMemoryKVStorage()
    doc_status = _InMemoryKVStorage()
    full_entities = _InMemoryKVStorage()
    full_relations = _InMemoryKVStorage()
    entity_chunks = _InMemoryKVStorage()
    relation_chunks = _InMemoryKVStorage()
    llm_cache = _InMemoryKVStorage()

    global_config = _build_global_config(workspace)

    # Doc-1 factual
    doc1 = await _ingest_factual_doc(
        doc_id="doc-1",
        file_path="doc1.pdf",
        src_name="ALPHA",
        src_type="ORG",
        tgt_name="BETA",
        tgt_type="ORG",
        relation_keywords="contract,agreement",
        relation_description="ALPHA and BETA signed an agreement.",
        graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        full_docs=full_docs,
        doc_status=doc_status,
        full_entities=full_entities,
        full_relations=full_relations,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        global_config=global_config,
    )

    # Doc-2 factual
    doc2 = await _ingest_factual_doc(
        doc_id="doc-2",
        file_path="doc2.pdf",
        src_name="GAMMA",
        src_type="PERSON",
        tgt_name="DELTA",
        tgt_type="PRODUCT",
        relation_keywords="develops,owns",
        relation_description="GAMMA developed DELTA.",
        graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        full_docs=full_docs,
        doc_status=doc_status,
        full_entities=full_entities,
        full_relations=full_relations,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        global_config=global_config,
    )

    # Doc-3 factual
    doc3 = await _ingest_factual_doc(
        doc_id="doc-3",
        file_path="doc3.pdf",
        src_name="OMEGA",
        src_type="CITY",
        tgt_name="THETA",
        tgt_type="COUNTRY",
        relation_keywords="located_in",
        relation_description="OMEGA is located in THETA.",
        graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        full_docs=full_docs,
        doc_status=doc_status,
        full_entities=full_entities,
        full_relations=full_relations,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        global_config=global_config,
    )

    # FACTUAL write semantics: stored weight is log1p(weight_raw).
    doc1_edge = await graph.get_edge(doc1["src_id"], doc1["tgt_id"])
    assert doc1_edge is not None
    assert doc1_edge.get("weight_raw") == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert doc1_edge.get("weight") == pytest.approx(math.log1p(1.0), rel=1e-6, abs=1e-6)
    rel_doc1 = compute_mdhash_id(
        min(doc1["src_id"], doc1["tgt_id"]) + max(doc1["src_id"], doc1["tgt_id"]),
        prefix="rel-",
    )
    assert rel_doc1 in relationships_vdb.payloads
    assert relationships_vdb.payloads[rel_doc1].get("weight_raw") == pytest.approx(
        1.0, rel=1e-6, abs=1e-6
    )

    # Doc-2 multimodal chunk + main entity
    mm_chunk_content = "doc2 multimodal chunk: table summary"
    mm_chunk_id = compute_mdhash_id(mm_chunk_content, prefix="chunk-")
    mm_chunk_payload = {
        "content": mm_chunk_content,
        "full_doc_id": "doc-2",
        "file_path": "doc2.pdf",
        "llm_cache_list": [],
        "tokens": len(mm_chunk_content),
        "is_multimodal": True,
        "modal_entity_name": "DOC2_TABLE_MAIN",
        "original_type": "table",
    }
    await text_chunks.upsert({mm_chunk_id: mm_chunk_payload})
    await chunks_vdb.upsert({mm_chunk_id: mm_chunk_payload})
    await _append_doc_status_chunks(doc_status, "doc-2", "doc2.pdf", [mm_chunk_id])

    mm_entity_name = "DOC2_TABLE_MAIN"
    mm_entity_type = "table"
    mm_entity_id = compute_entity_id(mm_entity_name, mm_entity_type, True)
    mm_entity_vdb_id = compute_entity_vdb_id(mm_entity_name, mm_entity_type, True)
    mm_entities_to_store = {
        mm_entity_vdb_id: {
            "entity_name": mm_entity_name,
            "entity_type": mm_entity_type,
            "entity_id": mm_entity_id,
            "content": "Table-level multimodal summary for doc-2",
            "source_id": mm_chunk_id,
            "file_path": "doc2.pdf",
        }
    }
    processor_lightrag = SimpleNamespace(
        chunk_entity_relation_graph=graph,
        entity_chunks=entity_chunks,
        entities_vdb=entities_vdb,
        full_entities=full_entities,
        enable_entity_disambiguation=True,
        max_source_ids_per_entity=20,
        source_ids_limit_method="KEEP",
    )
    processor = _DummyProcessor(processor_lightrag)
    await processor._upsert_multimodal_main_entities_to_core_storage(mm_entities_to_store)
    await processor._store_multimodal_entities_to_full_entities(mm_entities_to_store, "doc-2")

    # SYNONYM write gate: factual edge must not be overwritten by synonym linking.
    all_labels = await graph.get_all_labels()
    dims = max(4, len(all_labels))
    for idx, label in enumerate(sorted(all_labels)):
        if label in {doc1["src_id"], doc1["tgt_id"]}:
            vector = [1.0] + [0.0] * (dims - 1)
        else:
            vector = [0.0] * dims
            vector[(idx % (dims - 1)) + 1] = 1.0
        if "|" in label:
            entity_name, entity_type = label.rsplit("|", 1)
        else:
            entity_name, entity_type = label, ""
        entities_vdb.vectors[compute_entity_vdb_id(entity_name, entity_type, True)] = vector

    created_syn = await build_synonym_edges(
        entities_vdb=entities_vdb,
        knowledge_graph_inst=graph,
        new_entity_ids=None,
        synonymy_threshold=0.99,
        min_entity_len=1,
        enable_entity_disambiguation=True,
    )
    alpha_beta_edge = await graph.get_edge(doc1["src_id"], doc1["tgt_id"])
    assert created_syn == 0
    assert alpha_beta_edge is not None
    assert alpha_beta_edge.get("edge_type") == "FACTUAL"
    assert alpha_beta_edge.get("provenance") == "relation_extraction"

    # Existing SYNONYM edge should be fully replaced by later factual write.
    syn_src_name, syn_src_type = "SYN_A", "CONCEPT"
    syn_tgt_name, syn_tgt_type = "SYN_B", "CONCEPT"
    syn_src_id = compute_entity_id(syn_src_name, syn_src_type, True)
    syn_tgt_id = compute_entity_id(syn_tgt_name, syn_tgt_type, True)
    await graph.upsert_node(
        syn_src_id,
        {
            "entity_id": syn_src_id,
            "entity_type": syn_src_type,
            "description": "syn node a",
            "source_id": doc2["chunk_id"],
            "file_path": "doc2.pdf",
            "created_at": 1,
        },
    )
    await graph.upsert_node(
        syn_tgt_id,
        {
            "entity_id": syn_tgt_id,
            "entity_type": syn_tgt_type,
            "description": "syn node b",
            "source_id": doc2["chunk_id"],
            "file_path": "doc2.pdf",
            "created_at": 1,
        },
    )
    await graph.upsert_edge(
        syn_src_id,
        syn_tgt_id,
        {
            "description": "Synonym only edge",
            "keywords": "synonym,alias",
            "weight": 0.99,
            "edge_type": "SYNONYM",
            "provenance": "synonym_detection",
        },
    )

    syn_fact_chunk_content = "doc2 factual overwrite for synonym edge"
    syn_fact_chunk_id = compute_mdhash_id(syn_fact_chunk_content, prefix="chunk-")
    syn_fact_chunk_payload = {
        "content": syn_fact_chunk_content,
        "full_doc_id": "doc-2",
        "file_path": "doc2.pdf",
        "llm_cache_list": [],
        "tokens": len(syn_fact_chunk_content),
    }
    await text_chunks.upsert({syn_fact_chunk_id: syn_fact_chunk_payload})
    await chunks_vdb.upsert({syn_fact_chunk_id: syn_fact_chunk_payload})
    await _append_doc_status_chunks(doc_status, "doc-2", "doc2.pdf", [syn_fact_chunk_id])

    syn_fact_nodes = {
        syn_src_name: [
            {
                "entity_name": syn_src_name,
                "entity_type": syn_src_type,
                "description": "syn factual source",
                "source_id": syn_fact_chunk_id,
                "file_path": "doc2.pdf",
                "timestamp": 2,
            }
        ],
        syn_tgt_name: [
            {
                "entity_name": syn_tgt_name,
                "entity_type": syn_tgt_type,
                "description": "syn factual target",
                "source_id": syn_fact_chunk_id,
                "file_path": "doc2.pdf",
                "timestamp": 2,
            }
        ],
    }
    syn_fact_edges = {
        tuple(sorted((syn_src_name, syn_tgt_name))): [
            {
                "description": "SYN_A factually relates to SYN_B.",
                "keywords": "evidence,relationship",
                "weight": 2.0,
                "source_id": syn_fact_chunk_id,
                "file_path": "doc2.pdf",
                "timestamp": 2,
            }
        ]
    }
    await merge_nodes_and_edges(
        chunk_results=[(syn_fact_nodes, syn_fact_edges)],
        knowledge_graph_inst=graph,
        entity_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        global_config=global_config,
        full_entities_storage=full_entities,
        full_relations_storage=full_relations,
        doc_id="doc-2",
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=None,
        entity_chunks_storage=entity_chunks,
        relation_chunks_storage=relation_chunks,
        current_file_number=1,
        total_files=1,
        file_path="doc2.pdf",
    )
    syn_edge_after_factual = await graph.get_edge(syn_src_id, syn_tgt_id)
    assert syn_edge_after_factual is not None
    assert syn_edge_after_factual.get("edge_type") == "FACTUAL"
    assert syn_edge_after_factual.get("provenance") == "relation_extraction"
    assert syn_fact_chunk_id in str(syn_edge_after_factual.get("source_id", ""))
    assert syn_edge_after_factual.get("weight_raw") == pytest.approx(2.0, rel=1e-6, abs=1e-6)
    assert syn_edge_after_factual.get("weight") == pytest.approx(
        math.log1p(2.0), rel=1e-6, abs=1e-6
    )

    # Write completeness checks before deletion.
    doc2_entities = await full_entities.get_by_id("doc-2")
    doc2_relations = await full_relations.get_by_id("doc-2")
    doc2_status_before_delete = await doc_status.get_by_id("doc-2")
    assert doc2_entities is not None and mm_entity_id in doc2_entities.get("entity_names", [])
    assert doc2_relations is not None and doc2_relations.get("count", 0) >= 1
    assert doc2_status_before_delete is not None
    assert mm_chunk_id in doc2_status_before_delete.get("chunks_list", [])
    assert syn_fact_chunk_id in doc2_status_before_delete.get("chunks_list", [])
    assert await text_chunks.get_by_id(mm_chunk_id) is not None
    assert mm_chunk_id in chunks_vdb.payloads
    assert await entity_chunks.get_by_id(mm_entity_id) is not None
    assert await relation_chunks.get_by_id(make_relation_chunk_key(doc1["src_id"], doc1["tgt_id"])) is not None
    assert await relation_chunks.get_by_id(make_relation_chunk_key(doc2["src_id"], doc2["tgt_id"])) is not None
    assert await relation_chunks.get_by_id(make_relation_chunk_key(doc3["src_id"], doc3["tgt_id"])) is not None

    rag = _DummyLightRAG()
    rag.workspace = workspace
    rag.enable_entity_disambiguation = True
    rag.doc_status = doc_status
    rag.full_docs = full_docs
    rag.full_entities = full_entities
    rag.full_relations = full_relations
    rag.entity_chunks = entity_chunks
    rag.relation_chunks = relation_chunks
    rag.entities_vdb = entities_vdb
    rag.relationships_vdb = relationships_vdb
    rag.chunks_vdb = chunks_vdb
    rag.text_chunks = text_chunks
    rag.chunk_entity_relation_graph = graph
    rag.llm_response_cache = llm_cache

    delete_result = await rag.adelete_by_doc_id("doc-1", delete_llm_cache=False)
    assert delete_result.status == "success"

    # Deleted doc-1 must be fully removed.
    assert await doc_status.get_by_id("doc-1") is None
    assert await full_docs.get_by_id("doc-1") is None
    assert await full_entities.get_by_id("doc-1") is None
    assert await full_relations.get_by_id("doc-1") is None
    assert await text_chunks.get_by_id(doc1["chunk_id"]) is None
    assert doc1["chunk_id"] not in chunks_vdb.payloads
    assert await graph.get_node(doc1["src_id"]) is None
    assert await graph.get_node(doc1["tgt_id"]) is None
    assert await graph.get_edge(doc1["src_id"], doc1["tgt_id"]) is None
    assert await entity_chunks.get_by_id(doc1["src_id"]) is None
    assert await entity_chunks.get_by_id(doc1["tgt_id"]) is None
    assert await relation_chunks.get_by_id(
        make_relation_chunk_key(doc1["src_id"], doc1["tgt_id"])
    ) is None

    doc1_src_vdb = compute_entity_vdb_id("ALPHA", "ORG", True)
    doc1_tgt_vdb = compute_entity_vdb_id("BETA", "ORG", True)
    assert doc1_src_vdb not in entities_vdb.payloads
    assert doc1_tgt_vdb not in entities_vdb.payloads
    rel_doc1_a = compute_mdhash_id(doc1["src_id"] + doc1["tgt_id"], prefix="rel-")
    rel_doc1_b = compute_mdhash_id(doc1["tgt_id"] + doc1["src_id"], prefix="rel-")
    assert rel_doc1_a not in relationships_vdb.payloads
    assert rel_doc1_b not in relationships_vdb.payloads

    # Other docs and their graph/vector/KV data must remain safe.
    assert await doc_status.get_by_id("doc-2") is not None
    assert await doc_status.get_by_id("doc-3") is not None
    assert await full_docs.get_by_id("doc-2") is not None
    assert await full_docs.get_by_id("doc-3") is not None
    assert await full_entities.get_by_id("doc-2") is not None
    assert await full_entities.get_by_id("doc-3") is not None
    assert await full_relations.get_by_id("doc-2") is not None
    assert await full_relations.get_by_id("doc-3") is not None

    assert await graph.get_node(doc2["src_id"]) is not None
    assert await graph.get_node(doc2["tgt_id"]) is not None
    assert await graph.get_node(doc3["src_id"]) is not None
    assert await graph.get_node(doc3["tgt_id"]) is not None
    assert await graph.get_node(mm_entity_id) is not None
    assert await graph.get_edge(doc2["src_id"], doc2["tgt_id"]) is not None
    assert await graph.get_edge(doc3["src_id"], doc3["tgt_id"]) is not None

    doc2_src_vdb = compute_entity_vdb_id("GAMMA", "PERSON", True)
    doc2_tgt_vdb = compute_entity_vdb_id("DELTA", "PRODUCT", True)
    doc3_src_vdb = compute_entity_vdb_id("OMEGA", "CITY", True)
    doc3_tgt_vdb = compute_entity_vdb_id("THETA", "COUNTRY", True)
    assert doc2_src_vdb in entities_vdb.payloads
    assert doc2_tgt_vdb in entities_vdb.payloads
    assert doc3_src_vdb in entities_vdb.payloads
    assert doc3_tgt_vdb in entities_vdb.payloads
    assert mm_entity_vdb_id in entities_vdb.payloads

    rel_doc2 = compute_mdhash_id(
        min(doc2["src_id"], doc2["tgt_id"]) + max(doc2["src_id"], doc2["tgt_id"]),
        prefix="rel-",
    )
    rel_doc3 = compute_mdhash_id(
        min(doc3["src_id"], doc3["tgt_id"]) + max(doc3["src_id"], doc3["tgt_id"]),
        prefix="rel-",
    )
    assert rel_doc2 in relationships_vdb.payloads
    assert rel_doc3 in relationships_vdb.payloads

    assert await text_chunks.get_by_id(doc2["chunk_id"]) is not None
    assert await text_chunks.get_by_id(doc3["chunk_id"]) is not None
    assert await text_chunks.get_by_id(mm_chunk_id) is not None
    assert doc2["chunk_id"] in chunks_vdb.payloads
    assert doc3["chunk_id"] in chunks_vdb.payloads
    assert mm_chunk_id in chunks_vdb.payloads

    assert await entity_chunks.get_by_id(doc2["src_id"]) is not None
    assert await entity_chunks.get_by_id(doc2["tgt_id"]) is not None
    assert await entity_chunks.get_by_id(doc3["src_id"]) is not None
    assert await entity_chunks.get_by_id(doc3["tgt_id"]) is not None
    assert await entity_chunks.get_by_id(mm_entity_id) is not None
    assert await relation_chunks.get_by_id(
        make_relation_chunk_key(doc2["src_id"], doc2["tgt_id"])
    ) is not None
    assert await relation_chunks.get_by_id(
        make_relation_chunk_key(doc3["src_id"], doc3["tgt_id"])
    ) is not None


@pytest.mark.asyncio
async def test_mocked_delete_retry_after_rebuild_interruption(monkeypatch):
    workspace = "test-doc-delete-rebuild-retry"
    initialize_share_data(workers=1)
    set_default_workspace(workspace)
    await initialize_pipeline_status(workspace=workspace)
    pipeline_status = await get_namespace_data("pipeline_status", workspace=workspace)
    pipeline_status_lock = get_pipeline_status_lock(workspace=workspace)

    graph = _InMemoryGraphStorage()
    entities_vdb = _InMemoryVectorStorage()
    relationships_vdb = _InMemoryVectorStorage()
    chunks_vdb = _InMemoryVectorStorage()
    text_chunks = _InMemoryKVStorage()
    full_docs = _InMemoryKVStorage()
    doc_status = _InMemoryKVStorage()
    full_entities = _InMemoryKVStorage()
    full_relations = _InMemoryKVStorage()
    entity_chunks = _InMemoryKVStorage()
    relation_chunks = _InMemoryKVStorage()
    llm_cache = _InMemoryKVStorage()
    global_config = _build_global_config(workspace)

    # Doc-1 and Doc-2 share one factual relation to force relationship rebuild path.
    doc1 = await _ingest_factual_doc(
        doc_id="doc-1",
        file_path="doc1.pdf",
        src_name="ALPHA",
        src_type="ORG",
        tgt_name="BETA",
        tgt_type="ORG",
        relation_keywords="contract",
        relation_description="ALPHA signed contract with BETA.",
        graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        full_docs=full_docs,
        doc_status=doc_status,
        full_entities=full_entities,
        full_relations=full_relations,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        global_config=global_config,
    )
    doc2 = await _ingest_factual_doc(
        doc_id="doc-2",
        file_path="doc2.pdf",
        src_name="ALPHA",
        src_type="ORG",
        tgt_name="BETA",
        tgt_type="ORG",
        relation_keywords="contract,renewal",
        relation_description="ALPHA renewed contract with BETA.",
        graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        full_docs=full_docs,
        doc_status=doc_status,
        full_entities=full_entities,
        full_relations=full_relations,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        global_config=global_config,
    )
    doc3 = await _ingest_factual_doc(
        doc_id="doc-3",
        file_path="doc3.pdf",
        src_name="GAMMA",
        src_type="PERSON",
        tgt_name="DELTA",
        tgt_type="PRODUCT",
        relation_keywords="develops",
        relation_description="GAMMA developed DELTA.",
        graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        full_docs=full_docs,
        doc_status=doc_status,
        full_entities=full_entities,
        full_relations=full_relations,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        global_config=global_config,
    )

    shared_edge_before = await graph.get_edge(doc1["src_id"], doc1["tgt_id"])
    assert shared_edge_before is not None
    assert shared_edge_before.get("weight_raw") == pytest.approx(2.0, rel=1e-6, abs=1e-6)

    rag = _DummyLightRAG()
    rag.workspace = workspace
    rag.enable_entity_disambiguation = True
    rag.doc_status = doc_status
    rag.full_docs = full_docs
    rag.full_entities = full_entities
    rag.full_relations = full_relations
    rag.entity_chunks = entity_chunks
    rag.relation_chunks = relation_chunks
    rag.entities_vdb = entities_vdb
    rag.relationships_vdb = relationships_vdb
    rag.chunks_vdb = chunks_vdb
    rag.text_chunks = text_chunks
    rag.chunk_entity_relation_graph = graph
    rag.llm_response_cache = llm_cache

    original_rebuild = lightrag_module.rebuild_knowledge_from_chunks
    rebuild_calls = {"count": 0}

    async def flaky_rebuild(*args, **kwargs):
        rebuild_calls["count"] += 1
        if rebuild_calls["count"] == 1:
            raise RuntimeError("simulated rebuild interruption")
        return await original_rebuild(*args, **kwargs)

    monkeypatch.setattr(lightrag_module, "rebuild_knowledge_from_chunks", flaky_rebuild)
    monkeypatch.setattr(lightrag_module, "asdict", lambda _: global_config)

    # First deletion fails in rebuild stage; retry metadata should be persisted.
    first_attempt = await rag.adelete_by_doc_id("doc-1", delete_llm_cache=False)
    assert first_attempt.status == "fail"
    doc1_status_after_fail = await doc_status.get_by_id("doc-1")
    assert doc1_status_after_fail is not None
    metadata = doc1_status_after_fail.get("metadata", {})
    assert metadata.get("deletion_failed") is True
    assert metadata.get("deletion_failure_stage") == "rebuild_knowledge_graph"

    # Second attempt succeeds and keeps remaining docs/graph complete.
    second_attempt = await rag.adelete_by_doc_id("doc-1", delete_llm_cache=False)
    assert second_attempt.status == "success"

    # Doc-1 data removed.
    assert await doc_status.get_by_id("doc-1") is None
    assert await full_docs.get_by_id("doc-1") is None
    assert await full_entities.get_by_id("doc-1") is None
    assert await full_relations.get_by_id("doc-1") is None
    assert await text_chunks.get_by_id(doc1["chunk_id"]) is None
    assert doc1["chunk_id"] not in chunks_vdb.payloads

    # Shared relation should be rebuilt from doc-2 only.
    shared_edge_after = await graph.get_edge(doc1["src_id"], doc1["tgt_id"])
    assert shared_edge_after is not None
    assert doc1["chunk_id"] not in str(shared_edge_after.get("source_id", ""))
    assert doc2["chunk_id"] in str(shared_edge_after.get("source_id", ""))
    assert shared_edge_after.get("weight_raw") == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert shared_edge_after.get("weight") == pytest.approx(
        math.log1p(1.0), rel=1e-6, abs=1e-6
    )

    # Doc-2 / Doc-3 stay intact.
    assert await doc_status.get_by_id("doc-2") is not None
    assert await doc_status.get_by_id("doc-3") is not None
    assert await graph.get_edge(doc2["src_id"], doc2["tgt_id"]) is not None
    assert await graph.get_edge(doc3["src_id"], doc3["tgt_id"]) is not None
    assert await text_chunks.get_by_id(doc2["chunk_id"]) is not None
    assert await text_chunks.get_by_id(doc3["chunk_id"]) is not None
