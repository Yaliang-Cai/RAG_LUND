import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_docbench_single_retrieval_subset.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("docbench_single_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docbench_single_runner_defines_seven_groups_with_expected_defaults():
    runner = _load_runner()
    groups = runner.build_docbench_single_experiments()

    assert [group["name"] for group in groups] == [
        "v4_naive_dense",
        "v4_baseline_non_ppr",
        "v4_non_ppr_chunk_only",
        "v6_baseline_non_ppr_all_on",
        "v4_ppr_default",
        "v7_baseline_ppr_all_on",
        "v7_ppr_no_synonym_edges",
    ]
    assert all(group["top_k"] == 20 for group in groups)
    assert all(group["chunk_top_k"] == 10 for group in groups)
    assert all(group["naive_top_k"] == 20 for group in groups)

    by_name = {group["name"]: group for group in groups}
    assert by_name["v6_baseline_non_ppr_all_on"]["keyword_fanout_mode"] == "per_keyword_rrf"
    assert by_name["v6_baseline_non_ppr_all_on"]["retrieval_mode"] == "hybrid"
    assert by_name["v6_baseline_non_ppr_all_on"]["answer_context_mode"] == "chunk_only_prompt"
    assert by_name["v6_baseline_non_ppr_all_on"]["enable_kg_rerank"] is True

    assert by_name["v4_ppr_default"]["query_mode"] == "ppr"
    assert by_name["v4_ppr_default"]["exclude_synonym_edges"] is True
    assert by_name["v4_ppr_default"]["ppr_top_k"] == 50
    assert by_name["v4_ppr_default"]["ppr_qa_top_k"] == 10

    assert by_name["v7_baseline_ppr_all_on"]["exclude_synonym_edges"] is False
    assert by_name["v7_ppr_no_synonym_edges"]["exclude_synonym_edges"] is True


def test_docbench_single_runner_builds_isolated_output_and_shared_index_paths(tmp_path: Path):
    runner = _load_runner()
    args = runner.build_parser().parse_args(
        [
            "--run-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    paths = runner.resolve_run_paths(args)

    assert paths.run_root == tmp_path.resolve()
    assert paths.working_dir_root == tmp_path.resolve() / "index" / "rag_workspaces"
    assert paths.index_state_dir == tmp_path.resolve() / "index" / "state"
    assert paths.output_dir_for("v4_naive_dense") == (
        tmp_path.resolve() / "docbench_single__v4_naive_dense" / "evaluate"
    )
    assert args.max_async_generate == 6
