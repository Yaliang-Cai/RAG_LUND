import os
import sys
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from dotenv import load_dotenv

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
)
from lightrag.lightrag import LightRAG
from lightrag.operate import merge_nodes_and_edges
from lightrag.synonym_linking import build_synonym_edges
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_entity_id,
    compute_entity_vdb_id,
    compute_mdhash_id,
    make_relation_chunk_key,
)
from raganything.processor import ProcessorMixin

load_dotenv(dotenv_path=".env", override=False)

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


class _NoopLogger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in (content or "")]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class _DummyProcessor(ProcessorMixin):
    def __init__(self, lightrag_obj: Any):
        self.lightrag = lightrag_obj
        self.config = SimpleNamespace(use_full_path=False)
        self.logger = _NoopLogger()


def _stable_vector(text: str) -> list[float]:
    upper = (text or "").upper()
    if "ALPHA" in upper or "BETA" in upper:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if "SYN_A" in upper or "SYN_B" in upper:
        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    bucket = abs(hash(upper)) % 6 + 2
    vec = [0.0] * 8
    vec[bucket] = 1.0
    return vec


async def _mock_embedding(texts: list[str], **kwargs) -> np.ndarray:
    return np.array([_stable_vector(text) for text in texts], dtype=np.float32)


async def _dummy_llm(*args, **kwargs) -> str:
    return "summary"


def _require_env_and_connectivity() -> None:
    required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "QDRANT_URL"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing required env for integration test: {', '.join(missing)}")


async def _check_neo4j_reachable() -> None:
    from neo4j import AsyncGraphDatabase

    uri = os.getenv("NEO4J_URI", "")
    username = os.getenv("NEO4J_USERNAME", "")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE")
    driver = AsyncGraphDatabase.driver(
        uri,
        auth=(username, password),
        connection_timeout=3.0,
    )
    try:
        session_kwargs = {}
        if database:
            session_kwargs["database"] = database
        async with driver.session(**session_kwargs) as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            if not record or record.get("ok") != 1:
                pytest.skip("Neo4j connectivity check failed: unexpected query result")
    except Exception as exc:
        pytest.skip(f"Neo4j unreachable: {exc}")
    finally:
        await driver.close()


def _check_qdrant_reachable() -> None:
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=3)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant unreachable: {exc}")
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


async def _append_doc_status_chunks(
    doc_status_storage: Any,
    doc_id: str,
    file_path: str,
    new_chunk_ids: list[str],
) -> None:
    current = await doc_status_storage.get_by_id(doc_id)
    now_iso = datetime.now(timezone.utc).isoformat()

    if current is None:
        merged_chunks = list(dict.fromkeys(new_chunk_ids))
        payload = {
            "status": DocStatus.PROCESSED.value,
            "file_path": file_path,
            "chunks_list": merged_chunks,
            "chunks_count": len(merged_chunks),
            "metadata": {},
            "created_at": now_iso,
            "updated_at": now_iso,
        }
    else:
        merged_chunks = list(
            dict.fromkeys([*(current.get("chunks_list", [])), *new_chunk_ids])
        )
        payload = {
            **current,
            "chunks_list": merged_chunks,
            "chunks_count": len(merged_chunks),
            "updated_at": now_iso,
        }

    await doc_status_storage.upsert({doc_id: payload})


async def _ingest_factual_doc(
    *,
    rag: LightRAG,
    doc_id: str,
    file_path: str,
    src_name: str,
    src_type: str,
    tgt_name: str,
    tgt_type: str,
    relation_keywords: str,
    relation_description: str,
    pipeline_status: dict[str, Any],
    pipeline_status_lock: Any,
) -> dict[str, str]:
    chunk_content = f"{doc_id}:{src_name}-{tgt_name}:{relation_description}"
    chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")
    chunk_payload = {
        "content": chunk_content,
        "full_doc_id": doc_id,
        "file_path": file_path,
        "llm_cache_list": [],
        "tokens": len(chunk_content),
    }

    await rag.text_chunks.upsert({chunk_id: chunk_payload})
    await rag.chunks_vdb.upsert({chunk_id: chunk_payload})
    await rag.full_docs.upsert({doc_id: {"content": f"doc content: {doc_id}"}})
    await _append_doc_status_chunks(rag.doc_status, doc_id, file_path, [chunk_id])

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
        knowledge_graph_inst=rag.chunk_entity_relation_graph,
        entity_vdb=rag.entities_vdb,
        relationships_vdb=rag.relationships_vdb,
        global_config=rag.__dict__,
        full_entities_storage=rag.full_entities,
        full_relations_storage=rag.full_relations,
        doc_id=doc_id,
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=rag.llm_response_cache,
        entity_chunks_storage=rag.entity_chunks,
        relation_chunks_storage=rag.relation_chunks,
        current_file_number=1,
        total_files=1,
        file_path=file_path,
    )

    return {"chunk_id": chunk_id, "src_id": src_id, "tgt_id": tgt_id}


@pytest.fixture
async def rag_real(tmp_path: Path):
    pytest.importorskip("neo4j")
    pytest.importorskip("qdrant_client")
    _require_env_and_connectivity()
    await _check_neo4j_reachable()
    _check_qdrant_reachable()

    workspace = f"it-n4j-qdrant-{uuid4().hex[:10]}"
    working_dir = tmp_path / "rag_storage"

    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        graph_storage="Neo4JStorage",
        vector_storage="QdrantVectorDBStorage",
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=2048,
            func=_mock_embedding,
            model_name="it-mock-embed-8d",
        ),
        tokenizer=Tokenizer("it-mock-tokenizer", _SimpleTokenizerImpl()),
        enable_entity_disambiguation=True,
    )

    await rag.initialize_storages()

    try:
        yield rag
    finally:
        for storage in (
            rag.entities_vdb,
            rag.relationships_vdb,
            rag.chunks_vdb,
            rag.chunk_entity_relation_graph,
        ):
            if not storage:
                continue
            drop_fn = getattr(storage, "drop", None)
            if callable(drop_fn):
                try:
                    await drop_fn()
                except Exception:
                    pass
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_real_neo4j_qdrant_write_delete_chain(rag_real: LightRAG):
    rag = rag_real
    await initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    pipeline_status_lock = get_pipeline_status_lock(workspace=rag.workspace)

    doc1 = await _ingest_factual_doc(
        rag=rag,
        doc_id="doc-1",
        file_path="doc1.pdf",
        src_name="ALPHA",
        src_type="ORG",
        tgt_name="BETA",
        tgt_type="ORG",
        relation_keywords="contract,agreement",
        relation_description="ALPHA and BETA signed an agreement.",
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
    )

    doc2 = await _ingest_factual_doc(
        rag=rag,
        doc_id="doc-2",
        file_path="doc2.pdf",
        src_name="GAMMA",
        src_type="PERSON",
        tgt_name="DELTA",
        tgt_type="PRODUCT",
        relation_keywords="develops,owns",
        relation_description="GAMMA developed DELTA.",
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
    )

    doc3 = await _ingest_factual_doc(
        rag=rag,
        doc_id="doc-3",
        file_path="doc3.pdf",
        src_name="OMEGA",
        src_type="CITY",
        tgt_name="THETA",
        tgt_type="COUNTRY",
        relation_keywords="located_in",
        relation_description="OMEGA is located in THETA.",
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
    )

    doc1_edge = await rag.chunk_entity_relation_graph.get_edge(
        doc1["src_id"], doc1["tgt_id"]
    )
    assert doc1_edge is not None
    assert doc1_edge.get("weight_raw") == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert doc1_edge.get("weight") == pytest.approx(math.log1p(1.0), rel=1e-6, abs=1e-6)

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
    await rag.text_chunks.upsert({mm_chunk_id: mm_chunk_payload})
    await rag.chunks_vdb.upsert({mm_chunk_id: mm_chunk_payload})
    await _append_doc_status_chunks(rag.doc_status, "doc-2", "doc2.pdf", [mm_chunk_id])

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
    processor = _DummyProcessor(rag)
    await processor._upsert_multimodal_main_entities_to_core_storage(mm_entities_to_store)
    await processor._store_multimodal_entities_to_full_entities(mm_entities_to_store, "doc-2")

    created_syn = await build_synonym_edges(
        entities_vdb=rag.entities_vdb,
        knowledge_graph_inst=rag.chunk_entity_relation_graph,
        new_entity_ids=None,
        synonymy_threshold=0.99,
        min_entity_len=1,
        enable_entity_disambiguation=True,
    )
    alpha_beta_edge = await rag.chunk_entity_relation_graph.get_edge(
        doc1["src_id"], doc1["tgt_id"]
    )
    assert created_syn >= 0
    assert alpha_beta_edge is not None
    assert alpha_beta_edge.get("edge_type") == "FACTUAL"
    assert alpha_beta_edge.get("provenance") == "relation_extraction"

    syn_src_name, syn_src_type = "SYN_A", "CONCEPT"
    syn_tgt_name, syn_tgt_type = "SYN_B", "CONCEPT"
    syn_src_id = compute_entity_id(syn_src_name, syn_src_type, True)
    syn_tgt_id = compute_entity_id(syn_tgt_name, syn_tgt_type, True)
    await rag.chunk_entity_relation_graph.upsert_node(
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
    await rag.chunk_entity_relation_graph.upsert_node(
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
    await rag.chunk_entity_relation_graph.upsert_edge(
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
    await rag.text_chunks.upsert({syn_fact_chunk_id: syn_fact_chunk_payload})
    await rag.chunks_vdb.upsert({syn_fact_chunk_id: syn_fact_chunk_payload})
    await _append_doc_status_chunks(
        rag.doc_status, "doc-2", "doc2.pdf", [syn_fact_chunk_id]
    )

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
        knowledge_graph_inst=rag.chunk_entity_relation_graph,
        entity_vdb=rag.entities_vdb,
        relationships_vdb=rag.relationships_vdb,
        global_config=rag.__dict__,
        full_entities_storage=rag.full_entities,
        full_relations_storage=rag.full_relations,
        doc_id="doc-2",
        pipeline_status=pipeline_status,
        pipeline_status_lock=pipeline_status_lock,
        llm_response_cache=rag.llm_response_cache,
        entity_chunks_storage=rag.entity_chunks,
        relation_chunks_storage=rag.relation_chunks,
        current_file_number=1,
        total_files=1,
        file_path="doc2.pdf",
    )
    syn_edge_after_factual = await rag.chunk_entity_relation_graph.get_edge(
        syn_src_id, syn_tgt_id
    )
    assert syn_edge_after_factual is not None
    assert syn_edge_after_factual.get("edge_type") == "FACTUAL"
    assert syn_edge_after_factual.get("provenance") == "relation_extraction"
    assert syn_fact_chunk_id in str(syn_edge_after_factual.get("source_id", ""))
    assert syn_edge_after_factual.get("weight_raw") == pytest.approx(2.0, rel=1e-6, abs=1e-6)
    assert syn_edge_after_factual.get("weight") == pytest.approx(
        math.log1p(2.0), rel=1e-6, abs=1e-6
    )

    doc2_entities = await rag.full_entities.get_by_id("doc-2")
    doc2_relations = await rag.full_relations.get_by_id("doc-2")
    doc2_status_before_delete = await rag.doc_status.get_by_id("doc-2")
    assert doc2_entities is not None and mm_entity_id in doc2_entities.get("entity_names", [])
    assert doc2_relations is not None and doc2_relations.get("count", 0) >= 1
    assert doc2_status_before_delete is not None
    assert mm_chunk_id in doc2_status_before_delete.get("chunks_list", [])
    assert syn_fact_chunk_id in doc2_status_before_delete.get("chunks_list", [])
    assert await rag.text_chunks.get_by_id(mm_chunk_id) is not None
    assert await rag.chunks_vdb.get_by_id(mm_chunk_id) is not None
    assert await rag.entities_vdb.get_by_id(mm_entity_vdb_id) is not None
    assert await rag.entity_chunks.get_by_id(mm_entity_id) is not None
    assert (
        await rag.relation_chunks.get_by_id(
            make_relation_chunk_key(doc1["src_id"], doc1["tgt_id"])
        )
        is not None
    )
    assert (
        await rag.relation_chunks.get_by_id(
            make_relation_chunk_key(doc2["src_id"], doc2["tgt_id"])
        )
        is not None
    )
    assert (
        await rag.relation_chunks.get_by_id(
            make_relation_chunk_key(doc3["src_id"], doc3["tgt_id"])
        )
        is not None
    )

    delete_result = await rag.adelete_by_doc_id("doc-1", delete_llm_cache=False)
    assert delete_result.status == "success"

    assert await rag.doc_status.get_by_id("doc-1") is None
    assert await rag.full_docs.get_by_id("doc-1") is None
    assert await rag.full_entities.get_by_id("doc-1") is None
    assert await rag.full_relations.get_by_id("doc-1") is None
    assert await rag.text_chunks.get_by_id(doc1["chunk_id"]) is None
    assert await rag.chunks_vdb.get_by_id(doc1["chunk_id"]) is None
    assert await rag.chunk_entity_relation_graph.get_node(doc1["src_id"]) is None
    assert await rag.chunk_entity_relation_graph.get_node(doc1["tgt_id"]) is None
    assert (
        await rag.chunk_entity_relation_graph.get_edge(doc1["src_id"], doc1["tgt_id"])
        is None
    )
    assert await rag.entity_chunks.get_by_id(doc1["src_id"]) is None
    assert await rag.entity_chunks.get_by_id(doc1["tgt_id"]) is None
    assert (
        await rag.relation_chunks.get_by_id(
            make_relation_chunk_key(doc1["src_id"], doc1["tgt_id"])
        )
        is None
    )

    doc1_src_vdb = compute_entity_vdb_id("ALPHA", "ORG", True)
    doc1_tgt_vdb = compute_entity_vdb_id("BETA", "ORG", True)
    assert await rag.entities_vdb.get_by_id(doc1_src_vdb) is None
    assert await rag.entities_vdb.get_by_id(doc1_tgt_vdb) is None
    rel_doc1_a = compute_mdhash_id(doc1["src_id"] + doc1["tgt_id"], prefix="rel-")
    rel_doc1_b = compute_mdhash_id(doc1["tgt_id"] + doc1["src_id"], prefix="rel-")
    assert await rag.relationships_vdb.get_by_id(rel_doc1_a) is None
    assert await rag.relationships_vdb.get_by_id(rel_doc1_b) is None

    assert await rag.doc_status.get_by_id("doc-2") is not None
    assert await rag.doc_status.get_by_id("doc-3") is not None
    assert await rag.full_docs.get_by_id("doc-2") is not None
    assert await rag.full_docs.get_by_id("doc-3") is not None
    assert await rag.full_entities.get_by_id("doc-2") is not None
    assert await rag.full_entities.get_by_id("doc-3") is not None
    assert await rag.full_relations.get_by_id("doc-2") is not None
    assert await rag.full_relations.get_by_id("doc-3") is not None
    assert await rag.chunk_entity_relation_graph.get_node(doc2["src_id"]) is not None
    assert await rag.chunk_entity_relation_graph.get_node(doc2["tgt_id"]) is not None
    assert await rag.chunk_entity_relation_graph.get_node(doc3["src_id"]) is not None
    assert await rag.chunk_entity_relation_graph.get_node(doc3["tgt_id"]) is not None
    assert await rag.chunk_entity_relation_graph.get_node(mm_entity_id) is not None
    assert (
        await rag.chunk_entity_relation_graph.get_edge(doc2["src_id"], doc2["tgt_id"])
        is not None
    )
    assert (
        await rag.chunk_entity_relation_graph.get_edge(doc3["src_id"], doc3["tgt_id"])
        is not None
    )

    doc2_src_vdb = compute_entity_vdb_id("GAMMA", "PERSON", True)
    doc2_tgt_vdb = compute_entity_vdb_id("DELTA", "PRODUCT", True)
    doc3_src_vdb = compute_entity_vdb_id("OMEGA", "CITY", True)
    doc3_tgt_vdb = compute_entity_vdb_id("THETA", "COUNTRY", True)
    assert await rag.entities_vdb.get_by_id(doc2_src_vdb) is not None
    assert await rag.entities_vdb.get_by_id(doc2_tgt_vdb) is not None
    assert await rag.entities_vdb.get_by_id(doc3_src_vdb) is not None
    assert await rag.entities_vdb.get_by_id(doc3_tgt_vdb) is not None
    assert await rag.entities_vdb.get_by_id(mm_entity_vdb_id) is not None

    rel_doc2 = compute_mdhash_id(
        min(doc2["src_id"], doc2["tgt_id"]) + max(doc2["src_id"], doc2["tgt_id"]),
        prefix="rel-",
    )
    rel_doc3 = compute_mdhash_id(
        min(doc3["src_id"], doc3["tgt_id"]) + max(doc3["src_id"], doc3["tgt_id"]),
        prefix="rel-",
    )
    assert await rag.relationships_vdb.get_by_id(rel_doc2) is not None
    assert await rag.relationships_vdb.get_by_id(rel_doc3) is not None
    assert await rag.text_chunks.get_by_id(doc2["chunk_id"]) is not None
    assert await rag.text_chunks.get_by_id(doc3["chunk_id"]) is not None
    assert await rag.text_chunks.get_by_id(mm_chunk_id) is not None
    assert await rag.chunks_vdb.get_by_id(doc2["chunk_id"]) is not None
    assert await rag.chunks_vdb.get_by_id(doc3["chunk_id"]) is not None
    assert await rag.chunks_vdb.get_by_id(mm_chunk_id) is not None
    assert await rag.entity_chunks.get_by_id(doc2["src_id"]) is not None
    assert await rag.entity_chunks.get_by_id(doc2["tgt_id"]) is not None
    assert await rag.entity_chunks.get_by_id(doc3["src_id"]) is not None
    assert await rag.entity_chunks.get_by_id(doc3["tgt_id"]) is not None
    assert await rag.entity_chunks.get_by_id(mm_entity_id) is not None
    assert (
        await rag.relation_chunks.get_by_id(
            make_relation_chunk_key(doc2["src_id"], doc2["tgt_id"])
        )
        is not None
    )
    assert (
        await rag.relation_chunks.get_by_id(
            make_relation_chunk_key(doc3["src_id"], doc3["tgt_id"])
        )
        is not None
    )
