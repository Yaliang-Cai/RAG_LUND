import json
import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import patch

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

from evaluate_local.DocBench import evaluate_shared
from evaluate_local.SurGE import evaluate_surge_fast
from evaluate_local.ablation_flags import AblationFlags
from evaluate_local import run_retrieval_ablation
from raganything.constants import DEFAULT_KG_CHUNK_SELECTION_SOURCE


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_docbench_build_query_params_ppr_forces_chunk_only_prompt():
    flags = AblationFlags(
        enable_entity_disambiguation=True,
        enable_synonym_linking=True,
        enable_multi_hop=False,
        multi_hop_depth=2,
        ppr_damping=0.5,
        ppr_top_k=50,
        ppr_qa_top_k=20,
        passage_node_weight=0.05,
    )

    params = evaluate_shared._build_query_params(
        one_sentence=False,
        ablation_flags=flags,
        query_mode="ppr",
        recognition_top_k=20,
        keyword_fanout_mode="per_keyword_rrf",
        keyword_entity_rrf_k=10,
        keyword_relation_rrf_k=20,
        entity_retrieval_mode="hybrid",
        chunk_retrieval_mode="bm25",
        exclude_synonym_edges=False,
        answer_context_mode="kg_prompt",
        kg_chunk_selection_source="untruncated",
        max_total_tokens=45000,
        enable_rerank=False,
        bypass_query_cache=True,
        bypass_keywords_cache=False,
    )

    assert params["mode"] == "ppr"
    assert params["keyword_fanout_mode"] == "per_keyword_rrf"
    assert params["keyword_entity_rrf_k"] == 10
    assert params["keyword_relation_rrf_k"] == 20
    assert params["entity_qdrant_retrieval_mode"] == "hybrid"
    assert params["chunk_qdrant_retrieval_mode"] == "bm25"
    assert params["answer_context_mode"] == "chunk_only_prompt"
    assert params["kg_chunk_selection_source"] == "untruncated"
    assert params["max_total_tokens"] == 45000
    assert params["enable_rerank"] is False
    assert params["ppr_top_k"] == 50
    assert params["ppr_qa_top_k"] == 20
    assert params["bypass_query_cache"] is True
    assert params["bypass_keywords_cache"] is False


def test_docbench_build_query_params_accepts_top_chunk_and_naive_limits():
    params = evaluate_shared._build_query_params(
        query_mode="naive",
        top_k=20,
        chunk_top_k=10,
        naive_top_k=20,
    )

    assert params["mode"] == "naive"
    assert params["top_k"] == 20
    assert params["chunk_top_k"] == 10
    assert params["naive_top_k"] == 20
    assert params["answer_context_mode"] == "chunk_only_prompt"


def test_docbench_build_query_params_hybrid_keeps_chunk_and_answer_axes():
    params = evaluate_shared._build_query_params(
        one_sentence=False,
        ablation_flags=AblationFlags(),
        query_mode="hybrid",
        recognition_top_k=20,
        keyword_fanout_mode="per_keyword_rrf",
        keyword_entity_rrf_k=10,
        keyword_relation_rrf_k=20,
        entity_retrieval_mode="dense",
        chunk_retrieval_mode="hybrid",
        exclude_synonym_edges=True,
        answer_context_mode="chunk_only_prompt",
        kg_chunk_selection_source="untruncated",
        max_total_tokens=45000,
        enable_rerank=True,
        bypass_query_cache=True,
        bypass_keywords_cache=False,
    )

    assert params["mode"] == "hybrid"
    assert params["keyword_entity_rrf_k"] == 10
    assert params["keyword_relation_rrf_k"] == 20
    assert params["chunk_qdrant_retrieval_mode"] == "hybrid"
    assert params["answer_context_mode"] == "chunk_only_prompt"
    assert params["kg_chunk_selection_source"] == "untruncated"
    assert params["max_total_tokens"] == 45000


def test_docbench_rerank_payload_includes_after_chunk_top_k():
    payload = evaluate_shared._extract_rerank_chunk_payload(
        {
            "metadata": {
                "rerank_chunk_debug": {
                    "scores_all": [0.9, 0.8, 0.7],
                    "scores_after_threshold": [0.9, 0.8],
                    "scores_final": [0.9],
                    "chunk_ids_all": ["c1", "c2", "c3"],
                    "chunk_ids_after_threshold": ["c1", "c2"],
                    "chunk_ids_after_chunk_top_k": ["c1", "c2"],
                    "chunk_ids_final": ["c1"],
                    "count_input": 3,
                    "count_after_rerank": 3,
                    "count_after_threshold": 2,
                    "count_after_chunk_top_k": 2,
                    "count_final": 1,
                }
            },
            "data": {"chunks": [{"chunk_id": "c1", "rerank_score": 0.9}]},
        },
        query_params={},
    )

    assert payload["counts"]["after_chunk_top_k"] == 2
    assert payload["chunk_ids"]["after_chunk_top_k"] == ["c1", "c2"]


def test_surge_build_query_params_hybrid_keeps_chunk_axis_but_not_answer_axis():
    args = run_retrieval_ablation.build_parser().parse_args(
        [
            "--shared-workspace-id",
            "docbench_shared_graphbm25_20260421_v0_v1_v2",
            "--surge-workspace-id",
            "surge_fast_graphbm25_20260421_v0_v1_v2",
            "--query-modes",
            "hybrid",
        ]
    )
    args.query_mode = "hybrid"
    args.top_k = 40
    args.chunk_top_k = 20
    args.keyword_fanout_mode = "per_keyword_rrf"
    args.keyword_entity_rrf_k = 10
    args.keyword_relation_rrf_k = 20
    args.entity_retrieval_mode = "dense"
    args.chunk_retrieval_mode = "hybrid"
    args.exclude_synonym_edges = True
    args.answer_context_mode = "kg_prompt"
    args.bypass_query_cache = True
    args.bypass_keywords_cache = False
    args.enable_rerank = False
    args.enable_entity_disambiguation = True
    args.enable_synonym_linking = True
    args.enable_multi_hop = False
    args.multi_hop_depth = 2
    args.ppr_damping = 0.5
    args.ppr_top_k = 50
    args.ppr_qa_top_k = 50
    args.passage_node_weight = 0.05

    params = evaluate_surge_fast.build_query_params(args, chunk_top_k=20)

    assert params["mode"] == "hybrid"
    assert params["keyword_fanout_mode"] == "per_keyword_rrf"
    assert params["keyword_entity_rrf_k"] == 10
    assert params["keyword_relation_rrf_k"] == 20
    assert params["entity_qdrant_retrieval_mode"] == "dense"
    assert params["chunk_qdrant_retrieval_mode"] == "hybrid"
    assert params["max_total_tokens"] == 45000
    assert params["enable_rerank"] is False
    assert "answer_context_mode" not in params


def test_surge_parser_uses_default_kg_chunk_selection_source_constant():
    args = evaluate_surge_fast.build_parser().parse_args([])

    assert args.kg_chunk_selection_source == DEFAULT_KG_CHUNK_SELECTION_SOURCE


def test_surge_build_query_params_passes_kg_chunk_selection_source():
    args = evaluate_surge_fast.build_parser().parse_args(
        ["--kg-chunk-selection-source", "untruncated"]
    )

    params = evaluate_surge_fast.build_query_params(args, chunk_top_k=0)

    assert params["kg_chunk_selection_source"] == "untruncated"


def test_retrieval_ablation_matrix_excludes_ppr_local_and_answer_axis_for_ppr():
    experiments = run_retrieval_ablation.build_full_experiment_matrix(
        query_modes=["hybrid", "ppr"],
        keyword_fanout_modes=["joined", "per_keyword_rrf"],
        retrieval_modes=["dense", "hybrid"],
        exclude_synonym_edges_values=[True, False],
        kg_chunk_selection_sources=["truncated", "untruncated"],
        answer_context_modes=["kg_prompt", "chunk_only_prompt"],
    )

    assert experiments
    assert all(item["query_mode"] != "ppr_local" for item in experiments)

    ppr_rows = [item for item in experiments if item["query_mode"] == "ppr"]
    hybrid_rows = [item for item in experiments if item["query_mode"] == "hybrid"]

    assert ppr_rows
    assert hybrid_rows
    assert all(item["answer_context_mode"] == "chunk_only_prompt" for item in ppr_rows)
    assert all("kg_chunk_selection_source" not in item for item in ppr_rows)
    assert all(item["entity_retrieval_mode"] == item["chunk_retrieval_mode"] for item in experiments)
    assert all("kg_chunk_selection_source" in item for item in hybrid_rows)


def test_retrieval_ablation_reduced_docbench_matrix_has_named_groups():
    experiments = run_retrieval_ablation.build_reduced_experiment_matrix("shared")

    assert [item["name"] for item in experiments] == [
        "baseline_kg",
        "per_keyword_kg",
        "per_keyword_no_kg_rerank_kg",
        "retrieval_hybrid_kg",
        "untruncated_kg",
        "baseline_chunk_only",
        "untruncated_chunk_only",
        "ppr_dense_rerank",
        "ppr_dense_no_rerank",
        "ppr_hybrid_per_keyword",
    ]
    assert len(experiments) == 10
    assert all(item["task"] == "shared" for item in experiments)
    assert all(item["exclude_synonym_edges"] is True for item in experiments if item["query_mode"] != "ppr")
    assert all(item["exclude_synonym_edges"] is False for item in experiments if item["query_mode"] == "ppr")
    assert all("kg_chunk_selection_source" not in item for item in experiments if item["query_mode"] == "ppr")
    ppr_dense = next(item for item in experiments if item["name"] == "ppr_dense_rerank")
    ppr_dense_no_rerank = next(item for item in experiments if item["name"] == "ppr_dense_no_rerank")
    assert ppr_dense["enable_rerank"] is True
    assert ppr_dense["ppr_top_k"] == 50
    assert ppr_dense["ppr_qa_top_k"] == 20
    assert ppr_dense_no_rerank["enable_rerank"] is False
    assert ppr_dense_no_rerank["ppr_top_k"] == 50
    assert ppr_dense_no_rerank["ppr_qa_top_k"] == 20


def test_retrieval_ablation_reduced_surge_matrix_has_named_groups():
    experiments = run_retrieval_ablation.build_reduced_experiment_matrix("surge")

    assert [item["name"] for item in experiments] == [
        "baseline",
        "per_keyword",
        "per_keyword_no_kg_rerank",
        "retrieval_hybrid",
        "ppr_dense_rerank",
        "ppr_dense_no_rerank",
        "ppr_hybrid_per_keyword",
    ]
    assert len(experiments) == 7
    assert all(item["task"] == "surge" for item in experiments)
    assert all(item.get("kg_chunk_selection_source") == "untruncated" for item in experiments if item["query_mode"] != "ppr")
    assert all("kg_chunk_selection_source" not in item for item in experiments if item["query_mode"] == "ppr")
    ppr_dense = next(item for item in experiments if item["name"] == "ppr_dense_rerank")
    ppr_dense_no_rerank = next(item for item in experiments if item["name"] == "ppr_dense_no_rerank")
    assert ppr_dense["enable_rerank"] is True
    assert ppr_dense["ppr_top_k"] == 50
    assert ppr_dense["ppr_qa_top_k"] == 50
    assert ppr_dense_no_rerank["enable_rerank"] is False
    assert ppr_dense_no_rerank["ppr_top_k"] == 50
    assert ppr_dense_no_rerank["ppr_qa_top_k"] == 50


def test_retrieval_ablation_parser_exposes_boolean_cache_toggles():
    parser = run_retrieval_ablation.build_parser()

    defaults = parser.parse_args([])
    disabled = parser.parse_args(
        ["--no-bypass-query-cache", "--bypass-keywords-cache"]
    )

    assert defaults.bypass_query_cache is True
    assert defaults.bypass_keywords_cache is False
    assert defaults.keyword_entity_rrf_k == 10
    assert defaults.keyword_relation_rrf_k == 20
    assert defaults.matrix_mode == "reduced"
    assert defaults.shared_chunk_top_k == 20
    assert defaults.surge_chunk_top_k == 0
    assert defaults.max_total_tokens == 45000
    assert disabled.bypass_query_cache is False
    assert disabled.bypass_keywords_cache is True


def test_runner_commands_map_unified_retrieval_mode_and_docbench_limits():
    args = run_retrieval_ablation.build_parser().parse_args([])
    shared_exp = next(
        item
        for item in run_retrieval_ablation.build_reduced_experiment_matrix("shared")
        if item["name"] == "ppr_dense_no_rerank"
    )
    surge_exp = next(
        item
        for item in run_retrieval_ablation.build_reduced_experiment_matrix("surge")
        if item["name"] == "ppr_hybrid_per_keyword"
    )

    shared_cmd = run_retrieval_ablation._shared_command(args, shared_exp, Path("out"))
    surge_cmd = run_retrieval_ablation._surge_command(args, surge_exp, Path("out"))

    assert shared_cmd[shared_cmd.index("--entity_retrieval_mode") + 1] == "dense"
    assert shared_cmd[shared_cmd.index("--chunk_retrieval_mode") + 1] == "dense"
    assert shared_cmd[shared_cmd.index("--max_total_tokens") + 1] == "45000"
    assert shared_cmd[shared_cmd.index("--enable_rerank") + 1] == "false"
    assert shared_cmd[shared_cmd.index("--keyword_entity_rrf_k") + 1] == "10"
    assert shared_cmd[shared_cmd.index("--keyword_relation_rrf_k") + 1] == "20"
    assert shared_cmd[shared_cmd.index("--ppr_top_k") + 1] == "50"
    assert shared_cmd[shared_cmd.index("--ppr_qa_top_k") + 1] == "20"

    assert surge_cmd[surge_cmd.index("--entity_retrieval_mode") + 1] == "hybrid"
    assert surge_cmd[surge_cmd.index("--chunk_retrieval_mode") + 1] == "hybrid"
    assert surge_cmd[surge_cmd.index("--chunk-top-k") + 1] == "0"
    assert surge_cmd[surge_cmd.index("--max-total-tokens") + 1] == "45000"
    assert surge_cmd[surge_cmd.index("--enable-rerank") + 1] == "true"
    assert surge_cmd[surge_cmd.index("--keyword_entity_rrf_k") + 1] == "10"
    assert surge_cmd[surge_cmd.index("--keyword_relation_rrf_k") + 1] == "20"
    assert surge_cmd[surge_cmd.index("--ppr-top-k") + 1] == "50"
    assert surge_cmd[surge_cmd.index("--ppr-qa-top-k") + 1] == "50"


def test_runner_workspace_env_overrides_use_run_root():
    parser = run_retrieval_ablation.build_parser()
    args = parser.parse_args(
        [
            "--run-root",
            "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/graphbm25_20260421",
        ]
    )

    env = run_retrieval_ablation._build_workspace_env(
        {"PYTHONPATH": "existing"},
        args,
        shared_layout={
            "working_dir_root": args.run_root,
            "manifest_file": "",
            "failures_file": "",
        },
        surge_layout={"storage_root": args.run_root},
    )

    assert env["DOCBENCH_SHARED_WORKING_DIR_ROOT"] == args.run_root
    assert env["SURGE_FAST_RAG_STORAGE_DIR"] == args.run_root
    assert "existing" in env["PYTHONPATH"]


def test_debug_script_resolves_working_dir_from_run_root():
    script_path = PROJECT_ROOT / "scripts" / "debug_retrieval_ablation.py"
    spec = importlib.util.spec_from_file_location(
        "debug_retrieval_ablation_testmod",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with patch.object(
        module,
        "resolve_shared_workspace_layout",
        return_value={
            "workspace_dir": "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/graphbm25_20260421/docbench_shared_graphbm25_20260421_v0_v1_v2"
        },
    ):
        resolved = module._resolve_working_dir(
            "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/graphbm25_20260421",
            "docbench_shared_graphbm25_20260421_v0_v1_v2",
            None,
        )

    assert (
        resolved
        == "/data/y50056788/Yaliang/projects/rag-anything/evaluate_local/ablation_runs/graphbm25_20260421/docbench_shared_graphbm25_20260421_v0_v1_v2"
    )


def test_shared_workspace_layout_uses_outer_workspace_when_nested_kv_dir_exists(tmp_path):
    workspace_id = "ws"
    run_root = tmp_path / "graphbm25_20260421"
    outer = (
        run_root
        / "_workspace_cache"
        / "docbench_shared"
        / "v0_v1_v2"
        / "rag_workspaces"
        / workspace_id
    )
    inner = outer / workspace_id
    inner.mkdir(parents=True)
    (inner / "kv_store_doc_status.json").write_text("{}", encoding="utf-8")

    layout = run_retrieval_ablation.resolve_shared_workspace_layout(
        run_root=run_root,
        workspace_id=workspace_id,
        require_existing=True,
    )

    assert layout["workspace_dir"] == str(outer.resolve())
    assert layout["working_dir_root"] == str(outer.parent.resolve())


def test_surge_workspace_layout_resolves_workspace_cache_storage(tmp_path):
    workspace_id = "ws"
    run_root = tmp_path / "graphbm25_20260504"
    outer = (
        run_root
        / "_workspace_cache"
        / "surge_fast"
        / "v0"
        / "rag_storage"
        / workspace_id
    )
    inner = outer / workspace_id
    inner.mkdir(parents=True)
    (inner / "kv_store_doc_status.json").write_text("{}", encoding="utf-8")

    layout = run_retrieval_ablation.resolve_surge_workspace_layout(
        run_root=run_root,
        workspace_id=workspace_id,
        require_existing=True,
    )

    assert layout["workspace_dir"] == str(outer.resolve())
    assert layout["storage_root"] == str(outer.parent.resolve())
    assert layout["state_dir"] == str(outer.parent.parent.resolve())


def test_debug_script_group_matrix_matches_runner_group_names():
    script_path = PROJECT_ROOT / "scripts" / "debug_retrieval_ablation.py"
    spec = importlib.util.spec_from_file_location(
        "debug_retrieval_ablation_matrix_testmod",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert [item["name"] for item in module._debug_group_matrix("docbench")] == [
        item["name"]
        for item in run_retrieval_ablation.build_reduced_experiment_matrix("shared")
    ]
    assert [item["name"] for item in module._debug_group_matrix("surge")] == [
        item["name"]
        for item in run_retrieval_ablation.build_reduced_experiment_matrix("surge")
    ]


def test_debug_script_surge_kwargs_do_not_add_answer_context_axis():
    script_path = PROJECT_ROOT / "scripts" / "debug_retrieval_ablation.py"
    spec = importlib.util.spec_from_file_location(
        "debug_retrieval_ablation_kwargs_testmod",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module.DATASET = "surge"
    kwargs = module._query_kwargs(module._debug_group_matrix("surge")[0])

    assert "answer_context_mode" not in kwargs
    assert kwargs["keyword_entity_rrf_k"] == 10
    assert kwargs["keyword_relation_rrf_k"] == 20


def test_debug_script_ppr_kwargs_include_rerank_and_ppr_limits():
    script_path = PROJECT_ROOT / "scripts" / "debug_retrieval_ablation.py"
    spec = importlib.util.spec_from_file_location(
        "debug_retrieval_ablation_ppr_kwargs_testmod",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module.DATASET = "docbench"
    kwargs = module._query_kwargs(
        next(
            item
            for item in module._debug_group_matrix("docbench")
            if item["name"] == "ppr_dense_no_rerank"
        )
    )

    assert kwargs["mode"] == "ppr"
    assert kwargs["enable_rerank"] is False
    assert kwargs["keyword_entity_rrf_k"] == 10
    assert kwargs["keyword_relation_rrf_k"] == 20
    assert kwargs["ppr_top_k"] == 50
    assert kwargs["ppr_qa_top_k"] == 20
    assert kwargs["answer_context_mode"] == "chunk_only_prompt"


def test_query_params_reject_ppr_qa_top_k_larger_than_ppr_top_k():
    flags = AblationFlags(
        enable_entity_disambiguation=True,
        enable_synonym_linking=True,
        enable_multi_hop=False,
        multi_hop_depth=2,
        ppr_damping=0.5,
        ppr_top_k=50,
        ppr_qa_top_k=60,
        passage_node_weight=0.05,
    )

    with pytest.raises(ValueError, match="ppr_qa_top_k"):
        evaluate_shared._build_query_params(
            one_sentence=False,
            ablation_flags=flags,
            query_mode="ppr",
        )

    args = evaluate_surge_fast.build_parser().parse_args(
        [
            "--query-mode",
            "ppr",
            "--ppr-top-k",
            "50",
            "--ppr-qa-top-k",
            "60",
        ]
    )
    args.enable_entity_disambiguation = True
    args.enable_synonym_linking = True
    args.enable_multi_hop = False

    with pytest.raises(ValueError, match="ppr_qa_top_k"):
        evaluate_surge_fast.build_query_params(args, chunk_top_k=0)


def test_surge_query_params_accept_naive_top_k():
    args = evaluate_surge_fast.build_parser().parse_args(
        [
            "--query-mode",
            "naive",
            "--top-k",
            "20",
            "--chunk-top-k",
            "50",
            "--naive-top-k",
            "75",
        ]
    )

    params = evaluate_surge_fast.build_query_params(args, chunk_top_k=50)

    assert params["mode"] == "naive"
    assert params["top_k"] == 20
    assert params["chunk_top_k"] == 50
    assert params["naive_top_k"] == 75


def test_retrieval_ablation_reduced_v2_docbench_matrix_has_expected_groups():
    experiments = run_retrieval_ablation.build_reduced_v2_experiment_matrix("shared")

    assert [item["name"] for item in experiments] == [
        "baseline_non_ppr",
        "non_ppr_per_keyword",
        "non_ppr_kg_rerank",
        "non_ppr_retrieval_hybrid",
        "non_ppr_untruncated",
        "non_ppr_chunk_only",
        "ppr_default",
        "ppr_per_keyword_no_rerank",
        "ppr_hybrid_no_rerank",
        "ppr_rerank",
        "ppr_raw_rerank_rrf",
    ]
    assert len(experiments) == 11

    baseline = next(item for item in experiments if item["name"] == "baseline_non_ppr")
    kg_rerank = next(item for item in experiments if item["name"] == "non_ppr_kg_rerank")
    chunk_only = next(item for item in experiments if item["name"] == "non_ppr_chunk_only")
    ppr_default = next(item for item in experiments if item["name"] == "ppr_default")
    ppr_rrf = next(item for item in experiments if item["name"] == "ppr_raw_rerank_rrf")

    assert baseline["query_mode"] == "hybrid"
    assert baseline["keyword_fanout_mode"] == "joined"
    assert baseline["retrieval_mode"] == "dense"
    assert baseline["enable_rerank"] is True
    assert baseline["enable_kg_rerank"] is False
    assert baseline["kg_chunk_selection_source"] == "truncated"
    assert baseline["answer_context_mode"] == "kg_prompt"
    assert baseline["exclude_synonym_edges"] is True

    assert {
        key: value
        for key, value in kg_rerank.items()
        if key != "name"
    } == {
        key: value
        for key, value in baseline.items()
        if key not in {"name", "enable_kg_rerank"}
    } | {"enable_kg_rerank": True}

    assert {
        key: value
        for key, value in chunk_only.items()
        if key != "name"
    } == {
        key: value
        for key, value in baseline.items()
        if key not in {"name", "answer_context_mode"}
    } | {"answer_context_mode": "chunk_only_prompt"}

    assert ppr_default["enable_rerank"] is False
    assert ppr_default["ppr_top_k"] == 50
    assert ppr_default["ppr_qa_top_k"] == 20
    assert ppr_default["enable_kg_rerank"] is False
    assert ppr_default["exclude_synonym_edges"] is False
    assert next(item for item in experiments if item["name"] == "ppr_rerank")["enable_rerank"] is True

    assert ppr_rrf["ppr_post_rerank_fusion"] == "raw_rrf"
    assert ppr_rrf["ppr_post_rerank_rrf_k"] == 60


def test_retrieval_ablation_reduced_v2_surge_matrix_has_expected_groups():
    experiments = run_retrieval_ablation.build_reduced_v2_experiment_matrix("surge")

    assert [item["name"] for item in experiments] == [
        "baseline_non_ppr",
        "non_ppr_per_keyword",
        "non_ppr_kg_rerank",
        "non_ppr_retrieval_hybrid",
        "non_ppr_truncated",
        "ppr_default",
        "ppr_per_keyword_no_rerank",
        "ppr_hybrid_no_rerank",
        "ppr_rerank",
        "ppr_raw_rerank_rrf",
    ]
    assert len(experiments) == 10

    baseline = next(item for item in experiments if item["name"] == "baseline_non_ppr")
    ppr_default = next(item for item in experiments if item["name"] == "ppr_default")
    ppr_rrf = next(item for item in experiments if item["name"] == "ppr_raw_rerank_rrf")

    assert baseline["query_mode"] == "hybrid"
    assert baseline["kg_chunk_selection_source"] == "untruncated"
    assert baseline["enable_kg_rerank"] is False
    assert baseline["exclude_synonym_edges"] is True
    assert "answer_context_mode" not in baseline

    assert ppr_default["enable_rerank"] is False
    assert ppr_default["ppr_top_k"] == 100
    assert ppr_default["ppr_qa_top_k"] == 50
    assert ppr_default["enable_kg_rerank"] is False
    assert ppr_default["exclude_synonym_edges"] is False
    assert next(item for item in experiments if item["name"] == "ppr_rerank")["enable_rerank"] is True
    assert ppr_rrf["ppr_post_rerank_fusion"] == "raw_rrf"
    assert ppr_rrf["ppr_post_rerank_rrf_k"] == 60


def test_retrieval_ablation_parser_accepts_reduced_v2():
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v2"])

    assert args.matrix_mode == "reduced_v2"


def test_retrieval_ablation_parser_accepts_reduced_v4():
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v4"])

    assert args.matrix_mode == "reduced_v4"


def test_reduced_v4_defaults_point_to_20260504_v0_workspaces():
    argv = ["--matrix-mode", "reduced_v4"]
    args = run_retrieval_ablation.build_parser().parse_args(argv)

    run_retrieval_ablation._apply_matrix_mode_defaults(args, argv)

    assert args.run_root.endswith("graphbm25_20260504")
    assert args.shared_workspace_id == "docbench_shared_graphbm25_20260504_v0"
    assert args.surge_workspace_id == "surge_fast_graphbm25_20260504_v0"


def test_reduced_v4_defaults_keep_explicit_workspace_overrides():
    argv = [
        "--matrix-mode",
        "reduced_v4",
        "--shared-workspace-id",
        "custom_docbench_v0",
        "--surge-workspace-id",
        "custom_surge_v0",
    ]
    args = run_retrieval_ablation.build_parser().parse_args(argv)

    run_retrieval_ablation._apply_matrix_mode_defaults(args, argv)

    assert args.shared_workspace_id == "custom_docbench_v0"
    assert args.surge_workspace_id == "custom_surge_v0"


def test_runner_commands_include_ppr_post_rerank_fusion_flags():
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v2"])
    shared_exp = next(
        item
        for item in run_retrieval_ablation.build_reduced_v2_experiment_matrix("shared")
        if item["name"] == "ppr_raw_rerank_rrf"
    )
    surge_exp = next(
        item
        for item in run_retrieval_ablation.build_reduced_v2_experiment_matrix("surge")
        if item["name"] == "ppr_raw_rerank_rrf"
    )

    shared_cmd = run_retrieval_ablation._shared_command(args, shared_exp, Path("out"))
    surge_cmd = run_retrieval_ablation._surge_command(args, surge_exp, Path("out"))

    assert shared_cmd[shared_cmd.index("--ppr_post_rerank_fusion") + 1] == "raw_rrf"
    assert shared_cmd[shared_cmd.index("--ppr_post_rerank_rrf_k") + 1] == "60"
    assert surge_cmd[surge_cmd.index("--ppr_post_rerank_fusion") + 1] == "raw_rrf"
    assert surge_cmd[surge_cmd.index("--ppr_post_rerank_rrf_k") + 1] == "60"


def test_selected_experiments_support_reduced_v2():
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v2"])

    shared, surge = run_retrieval_ablation._selected_experiments(args)

    assert len(shared) == 11
    assert len(surge) == 10
    assert next(item for item in shared if item["name"] == "ppr_default")["ppr_qa_top_k"] == 20
    assert next(item for item in surge if item["name"] == "ppr_default")["ppr_top_k"] == 100


def test_retrieval_ablation_parser_accepts_reduced_v3_and_survey_ppr_defaults():
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v3"])

    assert args.matrix_mode == "reduced_v3"
    assert args.surge_survey_ppr_top_k == 500
    assert args.surge_survey_ppr_qa_top_k == 500


def test_retrieval_ablation_reduced_v3_docbench_matrix_is_cartesian():
    experiments = run_retrieval_ablation.build_reduced_v3_experiment_matrix("shared")

    hybrid_rows = [item for item in experiments if item["query_mode"] == "hybrid"]
    ppr_rows = [item for item in experiments if item["query_mode"] == "ppr"]

    assert len(experiments) == 45
    assert len(hybrid_rows) == 32
    assert len(ppr_rows) == 13
    assert all(item["exclude_synonym_edges"] is True for item in hybrid_rows)
    assert all(item["enable_rerank"] is True for item in hybrid_rows)
    assert {item["enable_kg_rerank"] for item in hybrid_rows} == {False, True}
    assert {item["kg_chunk_selection_source"] for item in hybrid_rows} == {
        "truncated",
        "untruncated",
    }
    assert {item["answer_context_mode"] for item in hybrid_rows} == {
        "kg_prompt",
        "chunk_only_prompt",
    }

    no_synonym_ppr = next(
        item for item in ppr_rows if item["name"] == "v3_ppr_default_no_synonym_edges"
    )
    assert no_synonym_ppr["keyword_fanout_mode"] == "joined"
    assert no_synonym_ppr["retrieval_mode"] == "dense"
    assert no_synonym_ppr["enable_rerank"] is False
    assert no_synonym_ppr["exclude_synonym_edges"] is True
    assert sum(item["exclude_synonym_edges"] is True for item in ppr_rows) == 1
    assert all(item["enable_kg_rerank"] is False for item in ppr_rows)
    assert all("kg_chunk_selection_source" not in item for item in ppr_rows)
    assert {item["enable_rerank"] for item in ppr_rows} == {False, True}
    assert {item["ppr_post_rerank_fusion"] for item in ppr_rows} == {"none", "raw_rrf"}
    assert all(item["ppr_top_k"] == 50 for item in ppr_rows)
    assert all(item["ppr_qa_top_k"] == 20 for item in ppr_rows)


def test_retrieval_ablation_reduced_v3_surge_matrix_is_query_survey_ready():
    experiments = run_retrieval_ablation.build_reduced_v3_experiment_matrix("surge")

    hybrid_rows = [item for item in experiments if item["query_mode"] == "hybrid"]
    ppr_rows = [item for item in experiments if item["query_mode"] == "ppr"]

    assert len(experiments) == 29
    assert len(hybrid_rows) == 16
    assert len(ppr_rows) == 13
    assert all(item["exclude_synonym_edges"] is True for item in hybrid_rows)
    assert all(item["enable_rerank"] is True for item in hybrid_rows)
    assert {item["kg_chunk_selection_source"] for item in hybrid_rows} == {
        "truncated",
        "untruncated",
    }
    assert all("answer_context_mode" not in item for item in experiments)

    no_synonym_ppr = next(
        item for item in ppr_rows if item["name"] == "v3_ppr_default_no_synonym_edges"
    )
    assert no_synonym_ppr["keyword_fanout_mode"] == "joined"
    assert no_synonym_ppr["retrieval_mode"] == "dense"
    assert no_synonym_ppr["enable_rerank"] is False
    assert no_synonym_ppr["exclude_synonym_edges"] is True
    assert sum(item["exclude_synonym_edges"] is True for item in ppr_rows) == 1
    assert all(item["enable_kg_rerank"] is False for item in ppr_rows)
    assert all("kg_chunk_selection_source" not in item for item in ppr_rows)
    assert all(item["ppr_top_k"] == 100 for item in ppr_rows)
    assert all(item["ppr_qa_top_k"] == 50 for item in ppr_rows)


def test_reduced_v3_selected_experiment_groups_include_surge_survey_overrides():
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v3"])

    groups = run_retrieval_ablation._selected_experiment_groups(args)

    assert len(groups["shared"]) == 45
    assert len(groups["surge_query"]) == 29
    assert len(groups["surge_survey"]) == 29
    assert next(item for item in groups["surge_query"] if item["query_mode"] == "ppr")[
        "ppr_top_k"
    ] == 100
    assert next(item for item in groups["surge_query"] if item["query_mode"] == "ppr")[
        "ppr_qa_top_k"
    ] == 50
    assert next(item for item in groups["surge_survey"] if item["query_mode"] == "ppr")[
        "ppr_top_k"
    ] == 500
    assert next(item for item in groups["surge_survey"] if item["query_mode"] == "ppr")[
        "ppr_qa_top_k"
    ] == 500


def test_reduced_v3_surge_survey_command_uses_survey_stage_and_ppr_500():
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v3"])
    groups = run_retrieval_ablation._selected_experiment_groups(args)
    survey_ppr = next(item for item in groups["surge_survey"] if item["query_mode"] == "ppr")
    survey_hybrid = next(
        item
        for item in groups["surge_survey"]
        if item["query_mode"] == "hybrid"
        and item["kg_chunk_selection_source"] == "untruncated"
    )

    ppr_cmd = run_retrieval_ablation._surge_command(
        args,
        survey_ppr,
        Path("out"),
        mode="survey",
    )
    hybrid_cmd = run_retrieval_ablation._surge_command(
        args,
        survey_hybrid,
        Path("out"),
        mode="survey",
    )

    assert _flag_value(ppr_cmd, "--mode") == "survey"
    assert _flag_value(ppr_cmd, "--survey-stage") == "retrieval"
    assert _flag_value(ppr_cmd, "--ppr-top-k") == "500"
    assert _flag_value(ppr_cmd, "--ppr-qa-top-k") == "500"
    assert "--kg-chunk-selection-source" not in ppr_cmd
    assert _flag_value(hybrid_cmd, "--kg-chunk-selection-source") == "untruncated"


def test_reduced_v3_dry_run_writes_config_with_survey_groups(tmp_path):
    output_root = tmp_path / "retrieval_outputs"
    run_root = tmp_path / "ablation_runs" / "graphbm25_20260431"

    code = run_retrieval_ablation.main(
        [
            "--matrix-mode",
            "reduced_v3",
            "--run-id",
            "dry_v3",
            "--run-root",
            str(run_root),
            "--output-root",
            str(output_root),
            "--shared-workspace-id",
            "docbench_shared_graphbm25_20260431_v0_v1_v2",
            "--surge-workspace-id",
            "surge_fast_graphbm25_20260431_v0_v1_v2",
            "--no-require-existing-workspaces",
            "--dry-run",
        ]
    )

    assert code == 0
    config = json.loads((output_root / "dry_v3" / "config.json").read_text(encoding="utf-8"))
    progress_rows = [
        json.loads(line)
        for line in (output_root / "dry_v3" / "progress.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(config["shared_experiments"]) == 45
    assert len(config["surge_query_experiments"]) == 29
    assert len(config["surge_survey_experiments"]) == 29
    assert {row["task"] for row in progress_rows if row["status"] == "completed"} == {
        "shared",
        "surge_query",
        "surge_survey",
    }


def test_retrieval_ablation_reduced_v4_docbench_matrix_has_expected_groups():
    experiments = run_retrieval_ablation.build_reduced_v4_experiment_matrix("shared")

    assert [item["name"] for item in experiments] == [
        "naive_dense",
        "baseline_non_ppr",
        "non_ppr_per_keyword",
        "non_ppr_kg_rerank",
        "non_ppr_retrieval_hybrid",
        "non_ppr_untruncated",
        "non_ppr_chunk_only",
        "ppr_default",
        "ppr_per_keyword_no_rerank",
        "ppr_hybrid_no_rerank",
        "ppr_rerank",
        "ppr_raw_rerank_rrf",
    ]
    assert len(experiments) == 12

    naive = next(item for item in experiments if item["name"] == "naive_dense")
    ppr_rows = [item for item in experiments if item["query_mode"] == "ppr"]

    assert naive["query_mode"] == "naive"
    assert naive["chunk_retrieval_mode"] == "dense"
    assert naive["top_k"] == 20
    assert naive["chunk_top_k"] == 10
    assert naive["naive_top_k"] == 20
    assert all(item["exclude_synonym_edges"] is True for item in ppr_rows)
    assert all(item["ppr_top_k"] == 50 for item in ppr_rows)
    assert all(item["ppr_qa_top_k"] == 10 for item in ppr_rows)


def test_retrieval_ablation_reduced_v4_surge_query_matrix_has_expected_groups():
    experiments = run_retrieval_ablation.build_reduced_v4_experiment_matrix("surge_query")

    assert [item["name"] for item in experiments] == [
        "naive_dense",
        "baseline_non_ppr",
        "non_ppr_per_keyword",
        "non_ppr_kg_rerank",
        "non_ppr_retrieval_hybrid",
        "non_ppr_untruncated",
        "ppr_default",
        "ppr_per_keyword_no_rerank",
        "ppr_hybrid_no_rerank",
        "ppr_rerank",
        "ppr_raw_rerank_rrf",
    ]
    assert len(experiments) == 11

    non_ppr_rows = [item for item in experiments if item["query_mode"] == "hybrid"]
    ppr_rows = [item for item in experiments if item["query_mode"] == "ppr"]

    assert {
        item["name"]: item["kg_chunk_selection_source"]
        for item in non_ppr_rows
    } == {
        "baseline_non_ppr": "truncated",
        "non_ppr_per_keyword": "truncated",
        "non_ppr_kg_rerank": "truncated",
        "non_ppr_retrieval_hybrid": "truncated",
        "non_ppr_untruncated": "untruncated",
    }
    assert all("answer_context_mode" not in item for item in experiments)
    assert all(item["top_k"] == 20 for item in experiments)
    assert all(item["chunk_top_k"] == 50 for item in experiments)
    assert all(item["naive_top_k"] == 75 for item in experiments)
    assert all(item["exclude_synonym_edges"] is True for item in ppr_rows)
    assert all(item["ppr_top_k"] == 100 for item in ppr_rows)
    assert all(item["ppr_qa_top_k"] == 50 for item in ppr_rows)


def test_retrieval_ablation_reduced_v4_surge_survey_matrix_uses_larger_candidate_pool():
    experiments = run_retrieval_ablation.build_reduced_v4_experiment_matrix("surge_survey")
    ppr_rows = [item for item in experiments if item["query_mode"] == "ppr"]

    assert len(experiments) == 11
    assert all(item["top_k"] == 20 for item in experiments)
    assert all(item["chunk_top_k"] == 500 for item in experiments)
    assert all(item["naive_top_k"] == 750 for item in experiments)
    assert all(item["exclude_synonym_edges"] is True for item in ppr_rows)
    assert all(item["ppr_top_k"] == 750 for item in ppr_rows)
    assert all(item["ppr_qa_top_k"] == 500 for item in ppr_rows)


def test_reduced_v4_selected_groups_and_commands_use_expected_windows():
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v4"])

    groups = run_retrieval_ablation._selected_experiment_groups(args)
    shared_ppr = next(item for item in groups["shared"] if item["name"] == "ppr_default")
    shared_naive = next(item for item in groups["shared"] if item["name"] == "naive_dense")
    surge_query_ppr = next(
        item for item in groups["surge_query"] if item["name"] == "ppr_default"
    )
    surge_survey_ppr = next(
        item for item in groups["surge_survey"] if item["name"] == "ppr_default"
    )

    shared_cmd = run_retrieval_ablation._shared_command(args, shared_ppr, Path("out"))
    shared_naive_cmd = run_retrieval_ablation._shared_command(
        args, shared_naive, Path("out")
    )
    surge_query_cmd = run_retrieval_ablation._surge_command(
        args, surge_query_ppr, Path("out"), mode="retrieval"
    )
    surge_survey_cmd = run_retrieval_ablation._surge_command(
        args, surge_survey_ppr, Path("out"), mode="survey"
    )

    assert len(groups["shared"]) == 12
    assert len(groups["surge_query"]) == 11
    assert len(groups["surge_survey"]) == 11

    assert _flag_value(shared_cmd, "--top_k") == "20"
    assert _flag_value(shared_cmd, "--chunk_top_k") == "10"
    assert _flag_value(shared_cmd, "--naive_top_k") == "20"
    assert _flag_value(shared_cmd, "--ppr_top_k") == "50"
    assert _flag_value(shared_cmd, "--ppr_qa_top_k") == "10"
    assert _flag_value(shared_naive_cmd, "--query_mode") == "naive"
    assert _flag_value(shared_naive_cmd, "--naive_top_k") == "20"

    assert _flag_value(surge_query_cmd, "--top-k") == "20"
    assert _flag_value(surge_query_cmd, "--chunk-top-k") == "50"
    assert _flag_value(surge_query_cmd, "--naive-top-k") == "75"
    assert _flag_value(surge_query_cmd, "--ppr-top-k") == "100"
    assert _flag_value(surge_query_cmd, "--ppr-qa-top-k") == "50"

    assert _flag_value(surge_survey_cmd, "--mode") == "survey"
    assert _flag_value(surge_survey_cmd, "--top-k") == "20"
    assert _flag_value(surge_survey_cmd, "--chunk-top-k") == "500"
    assert _flag_value(surge_survey_cmd, "--naive-top-k") == "750"
    assert _flag_value(surge_survey_cmd, "--ppr-top-k") == "750"
    assert _flag_value(surge_survey_cmd, "--ppr-qa-top-k") == "500"


def test_reduced_v4_dry_run_writes_config_with_survey_groups(tmp_path):
    output_root = tmp_path / "retrieval_outputs"
    run_root = tmp_path / "ablation_runs" / "graphbm25_20260504"

    code = run_retrieval_ablation.main(
        [
            "--matrix-mode",
            "reduced_v4",
            "--run-id",
            "dry_v4",
            "--run-root",
            str(run_root),
            "--output-root",
            str(output_root),
            "--shared-workspace-id",
            "docbench_shared_graphbm25_20260504_v0",
            "--surge-workspace-id",
            "surge_fast_graphbm25_20260504_v0",
            "--no-require-existing-workspaces",
            "--dry-run",
        ]
    )

    assert code == 0
    config = json.loads((output_root / "dry_v4" / "config.json").read_text(encoding="utf-8"))
    progress_rows = [
        json.loads(line)
        for line in (output_root / "dry_v4" / "progress.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert config["ablation_flags"]["enable_entity_disambiguation"] is False
    assert config["ablation_flags"]["enable_synonym_linking"] is False
    assert config["ablation_flags"]["enable_multi_hop"] is False
    assert len(config["shared_experiments"]) == 12
    assert len(config["surge_query_experiments"]) == 11
    assert len(config["surge_survey_experiments"]) == 11
    assert {row["task"] for row in progress_rows if row["status"] == "completed"} == {
        "shared",
        "surge_query",
        "surge_survey",
    }


def test_synonym_edge_write_policy_preserves_factual_edges():
    edge = {
        "edge_type": "FACTUAL",
        "provenance": "relation_extraction",
        "weight": 1.0,
    }
    synonym = {
        "edge_type": "SYNONYM",
        "provenance": "synonym_detection",
        "weight": 0.9,
    }

    assert run_retrieval_ablation._format_synonym_threshold_token(0.8) == "0p80"
    assert run_retrieval_ablation._parse_synonym_thresholds("0.8,0.9,0.95") == [
        0.8,
        0.9,
        0.95,
    ]
    from lightrag.synonym_linking import _should_upsert_synonym_edge

    assert _should_upsert_synonym_edge(None, synonym) is True
    assert _should_upsert_synonym_edge(edge, synonym) is False
    assert _should_upsert_synonym_edge(synonym, {**synonym, "weight": 0.95}) is True


def test_retrieval_ablation_reduced_v6_docbench_matrix_has_all_on_baseline_and_single_switch_ablations():
    experiments = run_retrieval_ablation.build_reduced_v6_experiment_matrix("shared")

    assert [item["name"] for item in experiments] == [
        "baseline_non_ppr_all_on",
        "non_ppr_no_chunk_only",
        "non_ppr_no_kg_rerank",
        "non_ppr_no_per_keyword",
        "non_ppr_no_retrieval_hybrid",
    ]
    assert len(experiments) == 5
    assert all(item["query_mode"] == "hybrid" for item in experiments)
    assert all(item["kg_chunk_selection_source"] == "truncated" for item in experiments)
    assert all(item["top_k"] == 20 for item in experiments)
    assert all(item["chunk_top_k"] == 10 for item in experiments)
    assert all(item["naive_top_k"] == 20 for item in experiments)
    assert all("ppr_top_k" not in item for item in experiments)

    by_name = {item["name"]: item for item in experiments}
    baseline = by_name["baseline_non_ppr_all_on"]
    assert baseline["answer_context_mode"] == "chunk_only_prompt"
    assert baseline["enable_kg_rerank"] is True
    assert baseline["keyword_fanout_mode"] == "per_keyword_rrf"
    assert baseline["retrieval_mode"] == "hybrid"
    assert baseline["entity_retrieval_mode"] == "hybrid"
    assert baseline["chunk_retrieval_mode"] == "hybrid"

    switch_keys = {
        "answer_context_mode",
        "enable_kg_rerank",
        "keyword_fanout_mode",
        "retrieval_mode",
        "entity_retrieval_mode",
        "chunk_retrieval_mode",
    }
    assert {
        key
        for key in switch_keys
        if by_name["non_ppr_no_chunk_only"][key] != baseline[key]
    } == {"answer_context_mode"}
    assert {
        key
        for key in switch_keys
        if by_name["non_ppr_no_kg_rerank"][key] != baseline[key]
    } == {"enable_kg_rerank"}
    assert {
        key
        for key in switch_keys
        if by_name["non_ppr_no_per_keyword"][key] != baseline[key]
    } == {"keyword_fanout_mode"}
    assert {
        key
        for key in switch_keys
        if by_name["non_ppr_no_retrieval_hybrid"][key] != baseline[key]
    } == {"retrieval_mode", "entity_retrieval_mode", "chunk_retrieval_mode"}


def test_reduced_v6_selected_groups_and_dry_run_are_docbench_only(tmp_path):
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v6"])
    run_retrieval_ablation._apply_matrix_mode_defaults(args, ["--matrix-mode", "reduced_v6"])
    groups = run_retrieval_ablation._selected_experiment_groups(args)

    assert args.run_root.endswith("graphbm25_20260504")
    assert args.shared_workspace_id == "docbench_shared_graphbm25_20260504_v0"
    assert len(groups["shared"]) == 5
    assert groups["surge_query"] == []
    assert groups["surge_survey"] == []

    output_root = tmp_path / "retrieval_outputs"
    run_root = tmp_path / "ablation_runs" / "graphbm25_20260504"
    code = run_retrieval_ablation.main(
        [
            "--matrix-mode",
            "reduced_v6",
            "--run-id",
            "dry_v6",
            "--run-root",
            str(run_root),
            "--output-root",
            str(output_root),
            "--shared-workspace-id",
            "docbench_shared_graphbm25_20260504_v0",
            "--no-require-existing-workspaces",
            "--dry-run",
        ]
    )

    assert code == 0
    config = json.loads(
        (output_root / "dry_v6" / "config.json").read_text(encoding="utf-8")
    )
    progress_rows = [
        json.loads(line)
        for line in (output_root / "dry_v6" / "progress.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert config["matrix_mode"] == "reduced_v6"
    assert config["run_root"] == str(run_root)
    assert config["shared_workspace_id"] == "docbench_shared_graphbm25_20260504_v0"
    assert len(config["shared_experiments"]) == 5
    assert config["surge_query_experiments"] == []
    assert config["surge_survey_experiments"] == []
    assert {row["task"] for row in progress_rows if row["status"] == "completed"} == {
        "shared"
    }


def test_reduced_v6_rejects_surge_only_tasks():
    with pytest.raises(ValueError, match="reduced_v6 is DocBench-only"):
        run_retrieval_ablation.main(
            [
                "--matrix-mode",
                "reduced_v6",
                "--tasks",
                "surge",
                "--no-require-existing-workspaces",
                "--dry-run",
            ]
        )


def test_retrieval_ablation_reduced_v7_docbench_matrix_has_ppr_all_on_and_single_switch_ablations():
    experiments = run_retrieval_ablation.build_reduced_v7_experiment_matrix("shared")

    assert [item["name"] for item in experiments] == [
        "baseline_ppr_all_on",
        "ppr_no_rerank",
        "ppr_no_per_keyword",
        "ppr_no_retrieval_hybrid",
        "ppr_no_synonym_edges",
    ]
    assert len(experiments) == 5
    assert all(item["task"] == "shared" for item in experiments)
    assert all(item["query_mode"] == "ppr" for item in experiments)
    assert all(item["answer_context_mode"] == "chunk_only_prompt" for item in experiments)
    assert all(item["enable_kg_rerank"] is False for item in experiments)
    assert all(item["ppr_post_rerank_fusion"] == "none" for item in experiments)
    assert all(item["ppr_post_rerank_rrf_k"] == 60 for item in experiments)
    assert all(item["ppr_top_k"] == 50 for item in experiments)
    assert all(item["ppr_qa_top_k"] == 10 for item in experiments)
    assert all(item["top_k"] == 20 for item in experiments)
    assert all(item["chunk_top_k"] == 10 for item in experiments)
    assert all(item["naive_top_k"] == 20 for item in experiments)
    assert all("kg_chunk_selection_source" not in item for item in experiments)

    by_name = {item["name"]: item for item in experiments}
    baseline = by_name["baseline_ppr_all_on"]
    assert baseline["exclude_synonym_edges"] is False
    assert baseline["keyword_fanout_mode"] == "per_keyword_rrf"
    assert baseline["retrieval_mode"] == "hybrid"
    assert baseline["entity_retrieval_mode"] == "hybrid"
    assert baseline["chunk_retrieval_mode"] == "hybrid"
    assert baseline["enable_rerank"] is True

    switch_keys = {
        "exclude_synonym_edges",
        "enable_rerank",
        "keyword_fanout_mode",
        "retrieval_mode",
        "entity_retrieval_mode",
        "chunk_retrieval_mode",
    }
    assert {
        key
        for key in switch_keys
        if by_name["ppr_no_rerank"][key] != baseline[key]
    } == {"enable_rerank"}
    assert {
        key
        for key in switch_keys
        if by_name["ppr_no_per_keyword"][key] != baseline[key]
    } == {"keyword_fanout_mode"}
    assert {
        key
        for key in switch_keys
        if by_name["ppr_no_retrieval_hybrid"][key] != baseline[key]
    } == {"retrieval_mode", "entity_retrieval_mode", "chunk_retrieval_mode"}
    assert {
        key
        for key in switch_keys
        if by_name["ppr_no_synonym_edges"][key] != baseline[key]
    } == {"exclude_synonym_edges"}


def test_reduced_v7_selected_groups_commands_and_dry_run_apply_synonyms_once(tmp_path):
    args = run_retrieval_ablation.build_parser().parse_args(["--matrix-mode", "reduced_v7"])
    run_retrieval_ablation._apply_matrix_mode_defaults(args, ["--matrix-mode", "reduced_v7"])
    groups = run_retrieval_ablation._selected_experiment_groups(args)

    baseline = groups["shared"][0]
    no_synonym_edges = groups["shared"][-1]
    baseline_cmd = run_retrieval_ablation._shared_command(args, baseline, Path("out"))
    no_synonym_cmd = run_retrieval_ablation._shared_command(
        args, no_synonym_edges, Path("out")
    )

    assert args.run_root.endswith("graphbm25_20260504")
    assert args.shared_workspace_id == "docbench_shared_graphbm25_20260504_v0"
    assert len(groups["shared"]) == 5
    assert groups["surge_query"] == []
    assert groups["surge_survey"] == []
    assert _flag_value(baseline_cmd, "--top_k") == "20"
    assert _flag_value(baseline_cmd, "--chunk_top_k") == "10"
    assert _flag_value(baseline_cmd, "--naive_top_k") == "20"
    assert _flag_value(baseline_cmd, "--ppr_top_k") == "50"
    assert _flag_value(baseline_cmd, "--ppr_qa_top_k") == "10"
    assert _flag_value(baseline_cmd, "--exclude_synonym_edges") == "false"
    assert "--answer_context_mode" not in baseline_cmd
    assert _flag_value(no_synonym_cmd, "--exclude_synonym_edges") == "true"

    output_root = tmp_path / "retrieval_outputs"
    run_root = tmp_path / "ablation_runs" / "graphbm25_20260504"
    code = run_retrieval_ablation.main(
        [
            "--matrix-mode",
            "reduced_v7",
            "--run-id",
            "dry_v7",
            "--run-root",
            str(run_root),
            "--output-root",
            str(output_root),
            "--shared-workspace-id",
            "docbench_shared_graphbm25_20260504_v0",
            "--no-require-existing-workspaces",
            "--dry-run",
        ]
    )

    assert code == 0
    config = json.loads(
        (output_root / "dry_v7" / "config.json").read_text(encoding="utf-8")
    )
    progress_rows = [
        json.loads(line)
        for line in (output_root / "dry_v7" / "progress.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    synonym_logs = sorted((output_root / "dry_v7" / "_synonym_ops").glob("*.json"))

    assert config["matrix_mode"] == "reduced_v7"
    assert config["run_root"] == str(run_root)
    assert config["shared_workspace_id"] == "docbench_shared_graphbm25_20260504_v0"
    assert [item["name"] for item in config["shared_experiments"]] == [
        "baseline_ppr_all_on",
        "ppr_no_rerank",
        "ppr_no_per_keyword",
        "ppr_no_retrieval_hybrid",
        "ppr_no_synonym_edges",
    ]
    assert config["surge_query_experiments"] == []
    assert config["surge_survey_experiments"] == []
    assert {row["task"] for row in progress_rows if row["status"] == "completed"} == {
        "shared"
    }
    assert len(synonym_logs) == 1
    synonym_payload = json.loads(synonym_logs[0].read_text(encoding="utf-8"))
    assert synonym_payload["synonymy_threshold"] == 0.8
    assert synonym_payload["workspace_label"] == "docbench_shared_graphbm25_20260504_v0"


def test_reduced_v7_dry_run_does_not_require_surge_workspace(tmp_path):
    output_root = tmp_path / "retrieval_outputs"
    run_root = tmp_path / "ablation_runs" / "graphbm25_20260504"
    workspace_id = "docbench_shared_graphbm25_20260504_v0"
    workspace_dir = (
        run_root
        / "_workspace_cache"
        / "docbench_shared"
        / "v0"
        / "rag_workspaces"
        / workspace_id
    )
    workspace_dir.mkdir(parents=True)

    code = run_retrieval_ablation.main(
        [
            "--matrix-mode",
            "reduced_v7",
            "--run-id",
            "dry_v7_docbench_only",
            "--run-root",
            str(run_root),
            "--output-root",
            str(output_root),
            "--shared-workspace-id",
            workspace_id,
            "--surge-workspace-id",
            "missing_surge_workspace",
            "--dry-run",
        ]
    )

    assert code == 0
    config = json.loads(
        (output_root / "dry_v7_docbench_only" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["shared_workspace_layout"]["workspace_dir"] == str(
        workspace_dir.resolve()
    )
    assert config["surge_query_experiments"] == []
    assert config["surge_survey_experiments"] == []


def test_reduced_v7_rejects_surge_only_tasks():
    with pytest.raises(ValueError, match="reduced_v7 is DocBench-only"):
        run_retrieval_ablation.main(
            [
                "--matrix-mode",
                "reduced_v7",
                "--tasks",
                "surge",
                "--no-require-existing-workspaces",
                "--dry-run",
            ]
        )


def test_retrieval_ablation_v5_synonym_matrices_have_three_thresholded_ppr_defaults():
    shared = run_retrieval_ablation.build_v5_synonym_experiment_matrix(
        "shared", thresholds=[0.8, 0.9, 0.95]
    )
    surge_query = run_retrieval_ablation.build_v5_synonym_experiment_matrix(
        "surge_query", thresholds=[0.8, 0.9, 0.95]
    )
    surge_survey = run_retrieval_ablation.build_v5_synonym_experiment_matrix(
        "surge_survey", thresholds=[0.8, 0.9, 0.95]
    )

    assert [item["name"] for item in shared] == [
        "ppr_default__syn_0p80",
        "ppr_default__syn_0p90",
        "ppr_default__syn_0p95",
    ]
    assert all(item["query_mode"] == "ppr" for item in shared)
    assert all(item["exclude_synonym_edges"] is False for item in shared)
    assert [item["synonymy_threshold"] for item in shared] == [0.8, 0.9, 0.95]
    assert all(item["ppr_top_k"] == 50 for item in shared)
    assert all(item["ppr_qa_top_k"] == 10 for item in shared)
    assert all(item["ppr_top_k"] == 100 for item in surge_query)
    assert all(item["ppr_qa_top_k"] == 50 for item in surge_query)
    assert all(item["ppr_top_k"] == 750 for item in surge_survey)
    assert all(item["ppr_qa_top_k"] == 500 for item in surge_survey)


def test_retrieval_ablation_v5_synonym_selected_groups_and_dry_run(tmp_path):
    args = run_retrieval_ablation.build_parser().parse_args(
        [
            "--matrix-mode",
            "v5_synonym",
            "--v5-synonym-thresholds",
            "0.8,0.9,0.95",
        ]
    )
    groups = run_retrieval_ablation._selected_experiment_groups(args)

    assert len(groups["shared"]) == 3
    assert len(groups["surge_query"]) == 3
    assert len(groups["surge_survey"]) == 3
    assert [item["synonymy_threshold"] for item in groups["shared"]] == [
        0.8,
        0.9,
        0.95,
    ]

    output_root = tmp_path / "retrieval_outputs"
    run_root = tmp_path / "ablation_runs" / "graphbm25_20260504"
    code = run_retrieval_ablation.main(
        [
            "--matrix-mode",
            "v5_synonym",
            "--run-id",
            "dry_v5_synonym",
            "--run-root",
            str(run_root),
            "--output-root",
            str(output_root),
            "--shared-workspace-id",
            "docbench_shared_graphbm25_20260504_v0",
            "--surge-workspace-id",
            "surge_fast_graphbm25_20260504_v0",
            "--v5-synonym-thresholds",
            "0.8,0.9,0.95",
            "--no-require-existing-workspaces",
            "--dry-run",
        ]
    )

    assert code == 0
    config = json.loads(
        (output_root / "dry_v5_synonym" / "config.json").read_text(encoding="utf-8")
    )
    progress_rows = [
        json.loads(line)
        for line in (output_root / "dry_v5_synonym" / "progress.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    synonym_logs = sorted(
        (output_root / "dry_v5_synonym" / "_synonym_ops").glob("*.json")
    )

    assert config["matrix_mode"] == "v5_synonym"
    assert len(config["shared_experiments"]) == 3
    assert len(config["surge_query_experiments"]) == 3
    assert len(config["surge_survey_experiments"]) == 3
    assert {row["task"] for row in progress_rows if row["status"] == "completed"} == {
        "shared",
        "surge_query",
        "surge_survey",
    }
    assert len(synonym_logs) == 6
    assert any("docbench_shared_graphbm25_20260504_v0__syn_0p80" in str(path) for path in synonym_logs)
    assert any("surge_fast_graphbm25_20260504_v0__syn_0p95" in str(path) for path in synonym_logs)
