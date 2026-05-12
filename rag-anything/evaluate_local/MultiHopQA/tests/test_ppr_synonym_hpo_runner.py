from pathlib import Path

from evaluate_local.MultiHopQA.optuna_ppr_synonym_hpo import (
    ANCHOR_CONFIG,
    HPOConfig,
    SEARCH_SPACE,
    _build_eval_command,
    _macro_metrics,
    _top_configs_for_full_confirmation,
    _threshold_label,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHELL_RUNNER = (
    PROJECT_ROOT
    / "evaluate_local"
    / "MultiHopQA"
    / "run_hipporag2_v4_ppr_synonym_hpo.sh"
)
PY_RUNNER = (
    PROJECT_ROOT
    / "evaluate_local"
    / "MultiHopQA"
    / "optuna_ppr_synonym_hpo.py"
)


def test_synonym_hpo_shell_runner_only_uses_existing_assets():
    text = SHELL_RUNNER.read_text(encoding="utf-8")

    assert "multihopqa_hr2_v0_ppr_hpo_semantic_prompt_syn_t" in text
    assert "SYNONYM_THRESHOLD=\"${SYNONYM_THRESHOLD:-0.8}\"" in text
    assert "CONCURRENCY=\"${CONCURRENCY:-100}\"" in text
    assert "N_TRIALS=\"${N_TRIALS:-40}\"" in text
    assert "build_index.py" not in text
    assert "download_hipporag2_datasets.py" not in text
    assert "scripts/manage_workspace_synonyms.py\" apply" in text


def test_synonym_hpo_shell_runner_all_stage_applies_synonyms_then_runs_all_stages():
    text = SHELL_RUNNER.read_text(encoding="utf-8")

    assert 'HPO_STAGE must be dev, verify, full, or all' in text
    assert 'if [[ "${HPO_STAGE}" == "all" ]]; then' in text
    assert "apply_synonym_edges_for_all" in text
    assert 'run_stage "dev"' in text
    assert 'run_stage "verify"' in text
    assert 'run_stage "full"' in text
    assert text.index("apply_synonym_edges_for_all") < text.index('run_stage "dev"')
    assert '--synonymy-threshold "${SYNONYM_THRESHOLD}"' in text
    assert "check_synonym_manifest_ready" in text


def test_synonym_hpo_shell_runner_does_not_forward_stage_name_as_extra_arg():
    text = SHELL_RUNNER.read_text(encoding="utf-8")
    run_stage_body = text.split("run_stage() {", 1)[1].split(
        "if [[ \"${HPO_STAGE}\" == \"all\" ]]",
        1,
    )[0]

    assert 'local stage="$1"' in run_stage_body
    assert "shift" in run_stage_body


def test_synonym_hpo_shell_runner_full_prefers_verify_top_configs():
    text = SHELL_RUNNER.read_text(encoding="utf-8")

    assert 'VERIFY_TOP_CONFIGS="${VERIFY_RESULTS_ROOT}/top_configs.tsv"' in text
    assert 'if [[ "${stage}" == "full" && -z "${CONFIGS_FILE:-}" && -f "${VERIFY_TOP_CONFIGS}" ]]; then' in text
    assert 'config_args=(--configs-file "${VERIFY_TOP_CONFIGS}")' in text


def test_synonym_hpo_search_space_matches_plan_and_excludes_synonym_weight_mode():
    assert SEARCH_SPACE == {
        "top_k": [5, 10, 20, 40],
        "ppr_qa_top_k": [3, 5, 8, 10],
        "ppr_top_k": [25, 50, 100],
        "passage_node_weight": [0, 0.02, 0.05, 0.1, 0.2],
        "ppr_damping": [0.35, 0.5, 0.65, 0.8],
        "hub_penalty_threshold": [0, 1, 10, 25, 50, 100],
    }
    assert "ppr_synonym_weight_mode" not in SEARCH_SPACE
    assert ANCHOR_CONFIG == HPOConfig(
        top_k=10,
        ppr_qa_top_k=5,
        ppr_top_k=50,
        passage_node_weight=0.05,
        ppr_damping=0.5,
        hub_penalty_threshold=50,
    )


def test_synonym_hpo_eval_command_pins_required_components(tmp_path):
    config = HPOConfig(
        top_k=20,
        ppr_qa_top_k=8,
        ppr_top_k=100,
        passage_node_weight=0.1,
        ppr_damping=0.65,
        hub_penalty_threshold=25,
    )

    cmd = _build_eval_command(
        python_executable="python",
        evaluate_script=Path("/repo/evaluate_local/MultiHopQA/evaluate_multihop.py"),
        dataset="hotpotqa",
        workspace_id="hotpotqa_hr2_v0",
        working_dir=Path("/work/hotpotqa"),
        data_dir=Path("/data/hr2"),
        output_dir=tmp_path / "out",
        config=config,
        n_samples=200,
        seed=42,
        concurrency=100,
        recall_k=(2, 5),
    )
    joined = " ".join(str(part) for part in cmd)

    assert "--qdrant-retrieval-mode hybrid" in joined
    assert "--keyword-fanout-mode joined" in joined
    assert "--no-exclude-synonym-edges" in joined
    assert "--ppr-synonym-weight-mode raw" in joined
    assert "--no-enable-kg-rerank" in joined
    assert "--no-ppr-enable-rerank" in joined
    assert "--ppr-post-rerank-fusion none" in joined
    assert "--qa-prompt-style semantic_cot" in joined
    assert "--answer-parse-mode answer_marker" in joined
    assert "--bypass-query-cache" in joined
    assert "--no-bypass-keywords-cache" in joined
    assert "--chunk-top-k 8" in joined
    assert "--ppr-qa-top-k 8" in joined


def test_macro_metrics_uses_macro_average_and_records_auxiliary_metrics():
    metrics = _macro_metrics(
        {
            "2wiki": {"f1": 0.7, "em": 0.6, "recall@2": 0.8, "recall@5": 0.9},
            "hotpotqa": {"f1": 0.5, "em": 0.4, "recall@2": 0.6, "recall@5": 0.7},
            "musique": {"f1": 0.3, "em": 0.2, "recall@2": 0.4, "recall@5": 0.5},
        }
    )

    assert metrics == {
        "macro_f1": 0.5,
        "macro_em": 0.4,
        "macro_recall@2": 0.6,
        "macro_recall@5": 0.7,
    }


def test_verify_top_configs_keeps_anchor_plus_top_three_non_anchor():
    configs = [
        ("trial_best", {"config": {"top_k": 20, "ppr_qa_top_k": 8, "ppr_top_k": 100, "passage_node_weight": 0.1, "ppr_damping": 0.65, "hub_penalty_threshold": 25}}),
        ("ppr_hybrid_syn_anchor", {"config": ANCHOR_CONFIG.to_params()}),
        ("trial_second", {"config": {"top_k": 40, "ppr_qa_top_k": 5, "ppr_top_k": 50, "passage_node_weight": 0.05, "ppr_damping": 0.5, "hub_penalty_threshold": 50}}),
        ("trial_third", {"config": {"top_k": 10, "ppr_qa_top_k": 3, "ppr_top_k": 25, "passage_node_weight": 0.02, "ppr_damping": 0.35, "hub_penalty_threshold": 10}}),
        ("trial_fourth", {"config": {"top_k": 5, "ppr_qa_top_k": 10, "ppr_top_k": 50, "passage_node_weight": 0.2, "ppr_damping": 0.8, "hub_penalty_threshold": 100}}),
    ]

    selected = _top_configs_for_full_confirmation(configs, top_n=3)

    assert [name for name, _ in selected] == [
        "ppr_hybrid_syn_anchor",
        "trial_best",
        "trial_second",
        "trial_third",
    ]


def test_threshold_label_is_path_safe():
    assert _threshold_label(0.8) == "0p8"
    assert _threshold_label(0.75) == "0p75"


def test_python_runner_uses_optuna_tpe_and_optional_median_pruner():
    text = PY_RUNNER.read_text(encoding="utf-8")

    assert "TPESampler" in text
    assert "n_startup_trials=12" in text
    assert "multivariate=True" in text
    assert "group=True" in text
    assert "constant_liar=optuna_jobs > 1" in text
    assert "MedianPruner" in text
    assert "n_warmup_steps=2" in text
    assert "n_min_trials=4" in text
    assert "enqueue_trial(ANCHOR_CONFIG.to_params())" in text
