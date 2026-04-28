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
