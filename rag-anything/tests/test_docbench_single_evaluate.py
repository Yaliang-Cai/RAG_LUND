import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_LIGHTRAG_ROOT = PROJECT_ROOT.parent / "lightrag"
if str(LOCAL_LIGHTRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIGHTRAG_ROOT))

if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")
    stub.CrossEncoder = object
    stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = stub

from evaluate_local.DocBench import evaluate
from raganything.constants import (
    DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES,
    DEFAULT_MIN_RERANK_SCORE,
)


pytestmark = pytest.mark.offline


def _write_docbench_doc(data_root: Path, doc_id: int, question_count: int = 1) -> None:
    doc_dir = data_root / str(doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4\n%mock\n")
    with (doc_dir / f"{doc_id}_qa.jsonl").open("w", encoding="utf-8") as f:
        for q_idx in range(question_count):
            f.write(
                json.dumps(
                    {
                        "question": f"What is value {doc_id}-{q_idx}?",
                        "answer": f"value {doc_id}-{q_idx}",
                        "type": "text-only",
                        "evidence": "mock evidence",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _patch_docbench_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    data_root = tmp_path / "docbench_data"
    output_dir = tmp_path / "docbench_results"
    working_dir_root = output_dir / "rag_workspaces"
    mineru_dir = output_dir / "mineru_outputs"
    for path in (data_root, output_dir, working_dir_root, mineru_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(evaluate, "DATA_ROOT", data_root)
    monkeypatch.setattr(evaluate, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(evaluate, "WORKING_DIR_ROOT", working_dir_root)
    monkeypatch.setattr(evaluate, "OUTPUT_MD_DIR", mineru_dir)
    monkeypatch.setattr(evaluate, "PROMPT_DUMP_DIR", output_dir / "prompt_dumps")
    monkeypatch.setattr(
        evaluate, "FINAL_MESSAGES_DUMP_DIR", output_dir / "final_vlm_messages"
    )
    monkeypatch.setattr(evaluate, "GENERATION_CONFIG_FILE", output_dir / "generation_config.json")
    monkeypatch.setattr(
        evaluate, "SINGLE_INGEST_MANIFEST_FILE", output_dir / "single_ingest_manifest.json"
    )
    monkeypatch.setattr(
        evaluate, "SINGLE_INGEST_FAILURES_FILE",
        output_dir / "single_ingest_failures.jsonl",
    )
    return data_root, output_dir


def test_docbench_single_settings_pin_v0_index_profile_and_bm25(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE", raising=False)
    monkeypatch.delenv("CONTEXT_ZERO_WINDOW_CONTENT_TYPES", raising=False)

    settings = evaluate._build_docbench_settings()

    assert settings.enable_entity_disambiguation is False
    assert settings.enable_synonym_linking is False
    assert settings.enable_multi_hop is False
    assert settings.enable_entity_surface_normalization is True
    assert settings.enable_keyword_case_normalization is True
    assert settings.strict_relation_endpoint_entity_match is True
    assert settings.qdrant_enable_sparse_bm25 is True
    assert settings.qdrant_retrieval_mode == "dense"
    assert settings.synonymy_threshold == 0.8
    assert settings.min_rerank_score == DEFAULT_MIN_RERANK_SCORE == 0.3
    assert evaluate.os.environ["ENABLE_TYPE_BASED_CONTEXT_WINDOW_OVERRIDE"] == "true"
    assert (
        evaluate.os.environ["CONTEXT_ZERO_WINDOW_CONTENT_TYPES"]
        == str(DEFAULT_CONTEXT_ZERO_WINDOW_CONTENT_TYPES)
    )


def test_docbench_single_index_profile_records_v0_synonym_postprocess():
    settings = evaluate._build_docbench_settings()
    profile = evaluate._build_docbench_index_profile(
        settings,
        apply_synonym_edges=True,
        synonymy_threshold=0.8,
    )

    assert profile["base_profile"] == "v0"
    assert profile["enable_entity_disambiguation"] is False
    assert profile["enable_synonym_linking"] is False
    assert profile["enable_multi_hop"] is False
    assert profile["synonym_edges_postprocess_enabled"] is True
    assert profile["synonym_edges_threshold"] == 0.8


def test_docbench_single_query_params_include_v4_window_controls():
    params = evaluate._build_docbench_query_params(
        query_mode="naive",
        top_k=20,
        chunk_top_k=10,
        naive_top_k=20,
    )

    assert params["mode"] == "naive"
    assert params["top_k"] == 20
    assert params["chunk_top_k"] == 10
    assert params["naive_top_k"] == 20


@pytest.mark.asyncio
async def test_docbench_single_resume_processed_doc_rejects_profile_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, output_dir = _patch_docbench_paths(monkeypatch, tmp_path)
    _write_docbench_doc(data_root, 0)
    (output_dir / "system_answers.jsonl").write_text(
        json.dumps({"doc_id": "0"}) + "\n",
        encoding="utf-8",
    )

    current_profile = evaluate._build_docbench_index_profile(
        evaluate._build_docbench_settings(),
        apply_synonym_edges=True,
        synonymy_threshold=0.8,
    )
    stale_profile = {
        **current_profile,
        "synonym_edges_threshold": 0.9,
    }
    workspace_id = evaluate._workspace_id_for_doc("0")
    profile_path = evaluate._workspace_profile_path(workspace_id)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(stale_profile), encoding="utf-8")
    evaluate._save_json_file(
        evaluate.SINGLE_INGEST_MANIFEST_FILE,
        {
            "schema_version": "docbench_single_ingest_manifest_v1",
            "docs": {
                "0": {
                    "status": "ok",
                    "workspace_id": workspace_id,
                    "index_profile": stale_profile,
                }
            },
        },
    )

    with pytest.raises(RuntimeError, match="processed-doc index profile mismatch"):
        await evaluate.generate_answers(
            start_id=0,
            end_id=1,
            resume=True,
            doc_flush_every=0,
        )


def test_docbench_single_query_params_auto_filter_synonym_edges_for_non_ppr():
    params = evaluate._build_docbench_query_params(
        query_mode="hybrid",
        entity_retrieval_mode="bm25",
        chunk_retrieval_mode="hybrid",
        exclude_synonym_edges=None,
        max_total_tokens=12345,
        multimodal_top_k=4,
        enable_rerank=False,
        enable_kg_rerank=True,
    )

    assert params["mode"] == "hybrid"
    assert params["entity_qdrant_retrieval_mode"] == "bm25"
    assert params["chunk_qdrant_retrieval_mode"] == "hybrid"
    assert params["exclude_synonym_edges"] is True
    assert params["max_total_tokens"] == 12345
    assert params["multimodal_top_k"] == 4
    assert params["enable_rerank"] is False
    assert params["enable_kg_rerank"] is True


def test_docbench_single_query_params_auto_keep_synonym_edges_for_ppr():
    params = evaluate._build_docbench_query_params(
        query_mode="ppr",
        exclude_synonym_edges=None,
        ppr_top_k=50,
        ppr_qa_top_k=20,
        recognition_top_k=7,
        ppr_post_rerank_fusion="raw_rrf",
    )

    assert params["mode"] == "ppr"
    assert params["exclude_synonym_edges"] is False
    assert params["ppr_top_k"] == 50
    assert params["ppr_qa_top_k"] == 20
    assert params["recognition_top_k"] == 7
    assert params["answer_context_mode"] == "chunk_only_prompt"
    assert params["ppr_post_rerank_fusion"] == "raw_rrf"


@pytest.mark.parametrize("explicit", [True, False])
def test_docbench_single_query_params_explicit_synonym_override(explicit: bool):
    params = evaluate._build_docbench_query_params(
        query_mode="ppr",
        exclude_synonym_edges=explicit,
    )

    assert params["exclude_synonym_edges"] is explicit


def test_docbench_single_query_params_reject_invalid_ppr_limits():
    with pytest.raises(ValueError, match="ppr_qa_top_k"):
        evaluate._build_docbench_query_params(
            query_mode="ppr",
            ppr_top_k=10,
            ppr_qa_top_k=11,
        )


@pytest.mark.asyncio
async def test_docbench_single_generate_uses_two_ingest_services_and_doc_scoped_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, output_dir = _patch_docbench_paths(monkeypatch, tmp_path)
    _write_docbench_doc(data_root, 0, question_count=2)
    _write_docbench_doc(data_root, 1, question_count=2)

    class _MockService:
        instances: list["_MockService"] = []
        ingest_service_ids: set[int] = set()
        active_ingests = 0
        max_active_ingests = 0
        active_query_docs: set[str] = set()
        max_active_query_docs = 0
        active_queries_by_doc: dict[str, int] = {}
        max_active_queries_by_doc: dict[str, int] = {}

        def __init__(self, settings: Any):
            self.settings = settings
            self.instance_id = len(self.instances)
            self.instances.append(self)

        async def ingest(
            self,
            file_path: str,
            output_dir: str | None = None,
            workspace_id: str | None = None,
            **kwargs: Any,
        ) -> str:
            self.ingest_service_ids.add(self.instance_id)
            type(self).active_ingests += 1
            type(self).max_active_ingests = max(
                type(self).max_active_ingests,
                type(self).active_ingests,
            )
            await evaluate.asyncio.sleep(0.01)
            type(self).active_ingests -= 1
            return str(workspace_id)

        async def get_rag(self, workspace_id: str):
            class _LightRag:
                enable_synonym_linking = False
                synonymy_threshold = 0.0
                synonymy_topk = 0
                synonymy_min_entity_len = 0

                async def rebuild_synonym_edges(self, reset_existing: bool = True):
                    return {
                        "success": True,
                        "skipped": False,
                        "cleared_edges": 0,
                        "created_edges": 3,
                    }

            return types.SimpleNamespace(lightrag=_LightRag())

        async def query(
            self,
            workspace_id: str,
            query: str,
            **kwargs: Any,
        ) -> str:
            type(self).active_query_docs.add(workspace_id)
            type(self).max_active_query_docs = max(
                type(self).max_active_query_docs,
                len(type(self).active_query_docs),
            )
            type(self).active_queries_by_doc[workspace_id] = (
                type(self).active_queries_by_doc.get(workspace_id, 0) + 1
            )
            type(self).max_active_queries_by_doc[workspace_id] = max(
                type(self).max_active_queries_by_doc.get(workspace_id, 0),
                type(self).active_queries_by_doc[workspace_id],
            )
            await evaluate.asyncio.sleep(0.01)
            type(self).active_queries_by_doc[workspace_id] -= 1
            if type(self).active_queries_by_doc[workspace_id] == 0:
                type(self).active_query_docs.discard(workspace_id)
            return f"answer for {query}"

    async def _noop_cleanup(*args: Any, **kwargs: Any) -> None:
        return None

    async def _noop_finalize(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(evaluate, "LocalRagService", _MockService)
    monkeypatch.setattr(evaluate, "_cleanup_rag_instance", _noop_cleanup)
    monkeypatch.setattr(evaluate, "_finalize_local_rag_service", _noop_finalize)
    monkeypatch.setattr(evaluate, "_ensure_master_log_handler", lambda: None)
    monkeypatch.setattr(evaluate, "_bridge_lightrag_logs_to_run_file", lambda: None)

    await evaluate.generate_answers(
        start_id=0,
        end_id=2,
        resume=False,
        max_async_generate=2,
        max_async_ingest_docs=2,
        max_async_query_docs=1,
        doc_flush_every=0,
        one_sentence=False,
    )

    assert len(_MockService.ingest_service_ids) == 2
    assert _MockService.max_active_ingests == 2
    assert _MockService.max_active_query_docs == 1
    assert max(_MockService.max_active_queries_by_doc.values()) == 2

    output_rows = [
        json.loads(line)
        for line in (output_dir / "system_answers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(output_rows) == 4
    assert {row["doc_id"] for row in output_rows} == {"0", "1"}

    manifest = json.loads(
        (output_dir / "single_ingest_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["docs"]["0"]["synonym_edges"]["applied"] is True
    assert manifest["docs"]["0"]["synonym_edges"]["threshold"] == 0.8
