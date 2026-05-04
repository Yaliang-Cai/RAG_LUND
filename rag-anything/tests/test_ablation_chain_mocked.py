import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys
import types

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

from evaluate_local.ablation_flags import AblationFlags
from evaluate_local import run_ablation_evals
from evaluate_local.DocBench import evaluate_shared
from evaluate_local.SurGE import evaluate_surge_fast


pytestmark = pytest.mark.offline


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _patch_docbench_paths(module: Any, output_dir: Path, data_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir_root = output_dir / "rag_workspaces"
    output_md_dir = output_dir / "mineru_outputs"
    working_dir_root.mkdir(parents=True, exist_ok=True)
    output_md_dir.mkdir(parents=True, exist_ok=True)

    module.DATA_ROOT = data_root
    module.OUTPUT_DIR = output_dir
    module.WORKING_DIR_ROOT = working_dir_root
    module.OUTPUT_MD_DIR = output_md_dir
    module.SYSTEM_ANSWERS_FILE = output_dir / "system_answers.jsonl"
    module.EVAL_RESULTS_FILE = output_dir / "eval_results.jsonl"
    module.STATS_FILE = output_dir / "statistics.json"
    module.RERANK_CHUNK_STATS_FILE = output_dir / "rerank_chunk_stats.jsonl"
    module.RERANK_CHUNK_SUMMARY_FILE = output_dir / "rerank_chunk_summary.json"
    module.GENERATION_CONFIG_FILE = output_dir / "generation_config.json"
    module.INGEST_MANIFEST_FILE = output_dir / "shared_ingest_manifest.json"
    module.INGEST_FAILURES_FILE = output_dir / "shared_ingest_failures.jsonl"


def _patch_surge_paths(module: Any, output_root: Path) -> None:
    module.OUTPUT_ROOT_DIR = output_root
    module.RETRIEVAL_DIR = output_root / "retrieval_results_fast"
    module.SURVEY_DIR = output_root / "survey_results_fast"
    module.LOG_DIR = output_root / "logs"
    module.RAG_STORAGE_DIR = output_root / "rag_storage"
    module.RAG_OUTPUT_DIR = output_root / "rag_outputs"
    module.PER_QUERY_FILE = module.RETRIEVAL_DIR / "retrieval_per_query.jsonl"
    module.SUMMARY_FILE = module.RETRIEVAL_DIR / "retrieval_summary.json"
    module.INDEX_SUMMARY_FILE = module.RETRIEVAL_DIR / "index_summary.json"
    module.RERANK_STATS_FILE = module.RETRIEVAL_DIR / "rerank_chunk_stats.jsonl"
    module.RERANK_SUMMARY_FILE = module.RETRIEVAL_DIR / "rerank_chunk_summary.json"
    module.WARNINGS_FILE = module.RETRIEVAL_DIR / "mapping_warnings.jsonl"
    module.INGEST_MANIFEST = module.RETRIEVAL_DIR / "shared_ingest_manifest_fast.json"
    module.INGEST_FAILURES = module.RETRIEVAL_DIR / "shared_ingest_failures_fast.jsonl"
    module.CHUNK_SOURCE_MAP_FILE = module.RETRIEVAL_DIR / "chunk_source_map.json"
    module.SURVEY_STATUS = module.SURVEY_DIR / "survey_mode_status.json"
    module.SURVEY_PER_FILE = module.SURVEY_DIR / "survey_retrieval_per_survey.jsonl"
    module.SURVEY_SUMMARY_FILE = module.SURVEY_DIR / "survey_retrieval_summary.json"
    module.SURVEY_RERANK_STATS_FILE = module.SURVEY_DIR / "survey_rerank_chunk_stats.jsonl"
    module.SURVEY_RERANK_SUMMARY_FILE = module.SURVEY_DIR / "survey_rerank_chunk_summary.json"
    module.SURVEY_WARNINGS_FILE = module.SURVEY_DIR / "survey_mapping_warnings.jsonl"
    module.ensure_dirs()


@pytest.mark.asyncio
async def test_docbench_generate_answers_shared_mocked_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "docbench_data"
    doc_dir = data_root / "0"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "0.pdf").write_bytes(b"%PDF-1.4\n%mock\n")
    _write_jsonl(
        doc_dir / "0_qa.jsonl",
        [
            {
                "question": "What value appears in the table?",
                "answer": "42",
                "type": "multimodal-f",
                "evidence": "A table on the page contains value 42.",
            }
        ],
    )

    output_dir = tmp_path / "docbench_output"
    _patch_docbench_paths(evaluate_shared, output_dir, data_root)

    class _MockDocBenchService:
        def __init__(self, settings: Any):
            self.settings = settings
            self.ingested_files: list[str] = []
            self.queries: list[str] = []
            self.finalize_calls: list[dict[str, Any]] = []

        async def ingest(
            self,
            file_path: str,
            output_dir: str | None = None,
            workspace_id: str | None = None,
            serialize_by_workspace_id: bool | None = None,
            **kwargs: Any,
        ) -> str:
            self.ingested_files.append(file_path)
            return str(workspace_id or "mock_ws")

        async def query_with_trace(
            self,
            workspace_id: str,
            query: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.queries.append(query)
            return {
                "answer": "The table value is 42.",
                "trace": {
                    "metadata": {
                        "rerank_chunk_debug": {
                            "scope": "all",
                            "scores_all": [0.93],
                            "scores_after_threshold": [0.93],
                            "scores_final": [0.93],
                            "count_input": 1,
                            "count_after_rerank": 1,
                            "count_after_threshold": 1,
                            "count_final": 1,
                        }
                    },
                    "data": {
                        "chunks": [
                            {"chunk_id": "chunk-docbench-0", "rerank_score": 0.93},
                        ]
                    },
                },
            }

        async def finalize_workspace_synonyms(
            self,
            workspace_id: str,
            *,
            force: bool = False,
            reset_existing: bool = True,
        ) -> dict[str, Any]:
            self.finalize_calls.append(
                {
                    "workspace_id": workspace_id,
                    "force": force,
                    "reset_existing": reset_existing,
                }
            )
            return {
                "success": True,
                "skipped": False,
                "cleared_edges": 2,
                "created_edges": 5,
            }

    async def _keep_service(
        service: Any,
        settings: Any,
        shared_workspace_id: str,
        clear_model_cache: bool = True,
    ) -> Any:
        return service

    def _mock_build_shared_settings(*, ablation_flags: AblationFlags) -> Any:
        return SimpleNamespace(working_dir_root=str(evaluate_shared.WORKING_DIR_ROOT))

    monkeypatch.setattr(evaluate_shared, "_recycle_local_rag_service", _keep_service)
    monkeypatch.setattr(evaluate_shared, "_build_shared_settings", _mock_build_shared_settings)
    monkeypatch.setattr(evaluate_shared, "_refresh_master_logging", lambda: None)

    created_services: list[_MockDocBenchService] = []

    def _mock_service_factory(settings: Any) -> _MockDocBenchService:
        service = _MockDocBenchService(settings)
        created_services.append(service)
        return service

    monkeypatch.setattr(evaluate_shared, "LocalRagService", _mock_service_factory)

    flags = AblationFlags(
        enable_entity_disambiguation=True,
        enable_synonym_linking=True,
        enable_multi_hop=False,
        multi_hop_depth=2,
        ppr_damping=0.5,
        ppr_top_k=50,
        passage_node_weight=0.05,
    )
    query_params = evaluate_shared._build_query_params(
        one_sentence=False,
        ablation_flags=flags,
        query_mode="hybrid",
        recognition_top_k=20,
    )

    await evaluate_shared.generate_answers_shared(
        start_id=0,
        end_id=1,
        resume=False,
        max_async_ingest=1,
        max_async_generate=1,
        one_sentence=False,
        profile_name="v0_mocked",
        eval_prompt_filename=evaluate_shared.RAGANYTHING_EVAL_PROMPT_FILENAME,
        shared_workspace_id="docbench_ws_mocked",
        retry_failed_only=False,
        clear_failures_on_success=True,
        ablation_flags=flags,
        query_params=query_params,
        experiment_id="exp_mock_docbench",
        allow_legacy_index_profile_adoption=True,
    )

    assert evaluate_shared.SYSTEM_ANSWERS_FILE.exists()
    with open(evaluate_shared.SYSTEM_ANSWERS_FILE, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "0"
    assert rows[0]["sys_ans"] == "The table value is 42."

    manifest = json.loads(evaluate_shared.INGEST_MANIFEST_FILE.read_text(encoding="utf-8"))
    assert manifest["shared_workspace_id"] == "docbench_ws_mocked"
    assert manifest["ingested_doc_ids"] == ["0"]

    assert evaluate_shared.RERANK_CHUNK_STATS_FILE.exists()
    with open(evaluate_shared.RERANK_CHUNK_STATS_FILE, "r", encoding="utf-8") as f:
        rerank_rows = [json.loads(line) for line in f if line.strip()]
    assert len(rerank_rows) == 1
    assert rerank_rows[0]["counts"]["final"] == 1
    assert len(created_services) == 1
    assert created_services[0].finalize_calls == [
        {
            "workspace_id": "docbench_ws_mocked",
            "force": False,
            "reset_existing": True,
        }
    ]


@pytest.mark.asyncio
async def test_docbench_index_only_ingests_without_answering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "docbench_data"
    doc_dir = data_root / "0"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "0.pdf").write_bytes(b"%PDF-1.4\n%mock\n")
    _write_jsonl(
        doc_dir / "0_qa.jsonl",
        [
            {
                "question": "What value appears in the table?",
                "answer": "42",
                "type": "multimodal-f",
                "evidence": "A table on the page contains value 42.",
            }
        ],
    )

    output_dir = tmp_path / "docbench_output"
    _patch_docbench_paths(evaluate_shared, output_dir, data_root)

    class _MockDocBenchService:
        def __init__(self, settings: Any):
            self.settings = settings
            self.ingested_files: list[str] = []
            self.queries: list[str] = []
            self.finalize_calls: list[dict[str, Any]] = []

        async def ingest(
            self,
            file_path: str,
            output_dir: str | None = None,
            workspace_id: str | None = None,
            serialize_by_workspace_id: bool | None = None,
            **kwargs: Any,
        ) -> str:
            self.ingested_files.append(file_path)
            return str(workspace_id or "mock_ws")

        async def query_with_trace(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.queries.append(str(args or kwargs))
            raise AssertionError("index-only mode must not query")

        async def finalize_workspace_synonyms(
            self,
            workspace_id: str,
            *,
            force: bool = False,
            reset_existing: bool = True,
        ) -> dict[str, Any]:
            self.finalize_calls.append(
                {
                    "workspace_id": workspace_id,
                    "force": force,
                    "reset_existing": reset_existing,
                }
            )
            return {"success": True, "skipped": False, "cleared_edges": 1, "created_edges": 2}

    async def _keep_service(
        service: Any,
        settings: Any,
        shared_workspace_id: str,
        clear_model_cache: bool = True,
    ) -> Any:
        return service

    def _mock_build_shared_settings(*, ablation_flags: AblationFlags) -> Any:
        return SimpleNamespace(working_dir_root=str(evaluate_shared.WORKING_DIR_ROOT))

    monkeypatch.setattr(evaluate_shared, "_recycle_local_rag_service", _keep_service)
    monkeypatch.setattr(evaluate_shared, "_build_shared_settings", _mock_build_shared_settings)
    monkeypatch.setattr(evaluate_shared, "_refresh_master_logging", lambda: None)

    created_services: list[_MockDocBenchService] = []

    def _mock_service_factory(settings: Any) -> _MockDocBenchService:
        service = _MockDocBenchService(settings)
        created_services.append(service)
        return service

    monkeypatch.setattr(evaluate_shared, "LocalRagService", _mock_service_factory)

    flags = AblationFlags(enable_entity_disambiguation=True, enable_synonym_linking=True)
    query_params = evaluate_shared._build_query_params(
        one_sentence=False,
        ablation_flags=flags,
        query_mode="hybrid",
        recognition_top_k=20,
    )

    await evaluate_shared.generate_answers_shared(
        start_id=0,
        end_id=1,
        resume=False,
        max_async_ingest=1,
        max_async_generate=1,
        one_sentence=False,
        profile_name="v0_mocked",
        eval_prompt_filename=evaluate_shared.RAGANYTHING_EVAL_PROMPT_FILENAME,
        shared_workspace_id="docbench_ws_index_only",
        retry_failed_only=False,
        clear_failures_on_success=True,
        ablation_flags=flags,
        query_params=query_params,
        experiment_id="exp_mock_docbench_index",
        allow_legacy_index_profile_adoption=True,
        index_only=True,
    )

    assert len(created_services) == 1
    assert len(created_services[0].ingested_files) == 1
    assert created_services[0].queries == []
    assert not evaluate_shared.SYSTEM_ANSWERS_FILE.exists()
    manifest = json.loads(evaluate_shared.INGEST_MANIFEST_FILE.read_text(encoding="utf-8"))
    assert manifest["ingested_doc_ids"] == ["0"]


@pytest.mark.asyncio
async def test_surge_run_retrieval_mocked_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "surge_output"
    _patch_surge_paths(evaluate_surge_fast, output_root)

    subset_dir = "subset_output"
    subset_root = tmp_path / "surge_data" / subset_dir
    subset_root.mkdir(parents=True, exist_ok=True)

    chunks_rows = [
        {"doc_id": 1, "chunk_id": "d1#0", "text": "Graph retrieval with image modality."},
        {"doc_id": 2, "chunk_id": "d2#0", "text": "Table evidence for retrieval benchmark."},
        {"doc_id": 3, "chunk_id": "d3#0", "text": "Background section with formula."},
    ]
    _write_jsonl(subset_root / "subset_chunks.jsonl", chunks_rows)

    queries = [
        {
            "query_id": 101,
            "prefix_titles_query": "Retrieve graph and table evidence.",
            "cites": [1, 2],
            "category": "mixed",
        }
    ]
    (subset_root / "subset_queries.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (subset_root / "subset_surveys.json").write_text("[]", encoding="utf-8")
    (subset_root / "subset_corpus.json").write_text("{}", encoding="utf-8")

    class _MockQueryParam:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

    monkeypatch.setattr(
        evaluate_surge_fast,
        "import_rag_dependencies",
        lambda: (_MockQueryParam, None, None),
    )

    @asynccontextmanager
    async def _mock_prepared_workspace_service(
        args: argparse.Namespace,
        source_records: dict[str, dict[str, Any]],
        *,
        stage: str,
    ):
        class _MockSurgeService:
            async def lightrag_aquery_data(
                self,
                workspace_id: str,
                query: str,
                param: Any,
            ) -> dict[str, Any]:
                ordered_docs = sorted(
                    source_records.values(),
                    key=lambda row: int(row["source_doc_id"]),
                )
                chosen = ordered_docs[:2]
                chunks = [
                    {
                        "chunk_id": row["lightrag_chunk_id"],
                        "rerank_score": 0.9 - idx * 0.1,
                    }
                    for idx, row in enumerate(chosen)
                ]
                return {
                    "metadata": {
                        "rerank_chunk_debug": {
                            "scope": "all",
                            "scores_all": [chunk["rerank_score"] for chunk in chunks],
                            "scores_after_threshold": [chunk["rerank_score"] for chunk in chunks],
                            "scores_final": [chunk["rerank_score"] for chunk in chunks],
                            "count_input": len(chunks),
                            "count_after_rerank": len(chunks),
                            "count_after_threshold": len(chunks),
                            "count_final": len(chunks),
                        }
                    },
                    "data": {"chunks": chunks},
                }

        ingest_summary = {
            "missing_full_set": [],
            "missing_doc_status_set": [],
            "status_not_processed_set": [],
            "missing_chunk_set": [],
            "missing_vdb_set": [],
            "chunk_count_mismatch_set": [],
        }
        yield (
            _MockSurgeService(),
            evaluate_surge_fast.get_ablation_flags(args),
            {"profile_version": 1},
            ingest_summary,
        )

    monkeypatch.setattr(
        evaluate_surge_fast,
        "prepared_workspace_service",
        _mock_prepared_workspace_service,
    )

    args = argparse.Namespace(
        data_root=str(tmp_path / "surge_data"),
        subset_dir=subset_dir,
        queries_file="subset_queries.json",
        surveys_file="subset_surveys.json",
        chunks_file="subset_chunks.jsonl",
        corpus_file="subset_corpus.json",
        workspace_id="surge_ws_mocked",
        query_mode="hybrid",
        top_k=40,
        chunk_top_k=0,
        k_list="1,2,5",
        survey_k_list="5,10",
        enable_rerank=True,
        batch_doc_concurrency=1,
        ingest_batch_size=8,
        llm_model_max_async=2,
        max_concurrency=1,
        max_retries=0,
        limit=0,
        enable_entity_disambiguation=True,
        enable_synonym_linking=True,
        enable_multi_hop=False,
        multi_hop_depth=2,
        ppr_damping=0.5,
        ppr_top_k=50,
        passage_node_weight=0.05,
        allow_legacy_index_profile_adoption=True,
        recognition_top_k=20,
    )

    code = await evaluate_surge_fast.run_retrieval(args)
    assert code == 0

    assert evaluate_surge_fast.PER_QUERY_FILE.exists()
    with open(evaluate_surge_fast.PER_QUERY_FILE, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["retrieved_doc_ids"] == [1, 2]
    assert rows[0]["error"] is None

    summary = json.loads(evaluate_surge_fast.SUMMARY_FILE.read_text(encoding="utf-8"))
    assert summary["query_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["avg_recall_at_k"]["2"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_surge_run_index_mocked_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "surge_output"
    _patch_surge_paths(evaluate_surge_fast, output_root)

    subset_dir = "subset_output"
    subset_root = tmp_path / "surge_data" / subset_dir
    subset_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        subset_root / "subset_chunks.jsonl",
        [{"doc_id": 1, "chunk_id": "d1#0", "text": "Graph retrieval with image modality."}],
    )
    (subset_root / "subset_queries.json").write_text("[]", encoding="utf-8")
    (subset_root / "subset_surveys.json").write_text("[]", encoding="utf-8")
    (subset_root / "subset_corpus.json").write_text("{}", encoding="utf-8")

    prepared_calls: list[str] = []

    @asynccontextmanager
    async def _mock_prepared_workspace_service(
        args: argparse.Namespace,
        source_records: dict[str, dict[str, Any]],
        *,
        stage: str,
    ):
        prepared_calls.append(stage)
        ingest_summary = {
            "missing_full_set": [],
            "missing_doc_status_set": [],
            "status_not_processed_set": [],
            "missing_chunk_set": [],
            "missing_vdb_set": [],
            "chunk_count_mismatch_set": [],
        }
        yield (
            object(),
            evaluate_surge_fast.get_ablation_flags(args),
            {"profile_version": 1, "enable_entity_disambiguation": True},
            ingest_summary,
        )

    monkeypatch.setattr(
        evaluate_surge_fast,
        "prepared_workspace_service",
        _mock_prepared_workspace_service,
    )

    args = argparse.Namespace(
        data_root=str(tmp_path / "surge_data"),
        subset_dir=subset_dir,
        queries_file="subset_queries.json",
        surveys_file="subset_surveys.json",
        chunks_file="subset_chunks.jsonl",
        corpus_file="subset_corpus.json",
        workspace_id="surge_ws_index_only",
        enable_entity_disambiguation=True,
        enable_synonym_linking=True,
        enable_multi_hop=False,
        multi_hop_depth=2,
        ppr_damping=0.5,
        ppr_top_k=50,
        ppr_qa_top_k=50,
        passage_node_weight=0.05,
    )

    code = await evaluate_surge_fast.run_index(args)

    assert code == 0
    assert prepared_calls == ["index build"]
    summary = json.loads((evaluate_surge_fast.RETRIEVAL_DIR / "index_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "index"
    assert summary["workspace_id"] == "surge_ws_index_only"


def test_run_ablation_evals_index_only_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, list[str]]] = []

    def _record_stage(**kwargs: Any) -> tuple[bool, float]:
        recorded.append((kwargs["stage_name"], list(kwargs["command"])))
        return True, 0.0

    monkeypatch.setattr(run_ablation_evals, "_run_one_stage", _record_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ablation_evals",
            "--run-id",
            "index_only_mock",
            "--runs-root",
            str(tmp_path / "runs"),
            "--profiles",
            "v0_v1_v2",
            "v0",
            "v0_v1",
            "--tasks",
            "both",
            "--index-only",
        ],
    )

    assert run_ablation_evals.main() == 0
    stages = [stage for stage, _ in recorded]
    assert stages == [
        "shared_index",
        "surge_index",
        "shared_index",
        "surge_index",
        "shared_index",
        "surge_index",
    ]
    assert all("--mode" in command and "index" in command for _, command in recorded)
    assert all("generate" not in command and "retrieval" not in command for _, command in recorded)
