from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "evaluate_local" / "MultiHopQA" / "run_hipporag2_v4_ppr_hpo.sh"


def _script_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_ppr_hpo_runner_exists_and_only_uses_existing_assets():
    text = _script_text()

    assert "multihopqa_hr2_v0_ppr_hpo_semantic_prompt_dev" in text
    assert "multihopqa_hr2_v0_ppr_hpo_semantic_prompt_full" in text
    assert "build_index.py" not in text
    assert "download_hipporag2_datasets.py" not in text
    assert "manage_workspace_synonyms.py" not in text
    assert 'file_prefix="2wikimultihopqa"' in text


def test_ppr_hpo_runner_pins_base_eval_controls():
    text = _script_text()

    for assignment in [
        'CONCURRENCY="${CONCURRENCY:-100}"',
        'DEV_N_SAMPLES="${DEV_N_SAMPLES:-200}"',
        'FULL_N_SAMPLES="${FULL_N_SAMPLES:-1000}"',
        'BASE_TOP_K="${BASE_TOP_K:-10}"',
        'BASE_PPR_QA_TOP_K="${BASE_PPR_QA_TOP_K:-5}"',
        'BASE_PPR_TOP_K="${BASE_PPR_TOP_K:-50}"',
        'BASE_PASSAGE_NODE_WEIGHT="${BASE_PASSAGE_NODE_WEIGHT:-0.05}"',
        'BASE_PPR_DAMPING="${BASE_PPR_DAMPING:-0.5}"',
        'BASE_HUB_PENALTY_THRESHOLD="${BASE_HUB_PENALTY_THRESHOLD:-50}"',
        'QDRANT_RETRIEVAL_MODE="${QDRANT_RETRIEVAL_MODE:-hybrid}"',
        'KEYWORD_FANOUT_MODE="${KEYWORD_FANOUT_MODE:-joined}"',
    ]:
        assert assignment in text

    for flag in [
        '--qa-prompt-style "semantic_cot"',
        '--answer-parse-mode "answer_marker"',
        "--bypass-query-cache",
        "--no-bypass-keywords-cache",
        "--exclude-synonym-edges",
        "--no-enable-kg-rerank",
        "--no-ppr-enable-rerank",
        '--ppr-post-rerank-fusion "none"',
        "--hub-penalty-threshold",
    ]:
        assert flag in text


def test_ppr_hpo_runner_declares_stage1_search_space():
    text = _script_text()

    for declaration in [
        "TOP_K_VALUES=(5 10 20 40)",
        "PPR_QA_TOP_K_VALUES=(3 5 8 10)",
        "PPR_TOP_K_VALUES=(25 50 100)",
        "PASSAGE_NODE_WEIGHT_VALUES=(0 0.02 0.05 0.1 0.2)",
        "PPR_DAMPING_VALUES=(0.35 0.5 0.65 0.8)",
        "HUB_PENALTY_THRESHOLD_VALUES=(0 1 10 50)",
    ]:
        assert declaration in text


def test_ppr_hpo_runner_keeps_hybrid_joined_no_synonym_anchor():
    text = _script_text()

    assert 'add_config "ppr_hybrid_anchor"' in text
    assert '"${KEYWORD_FANOUT_MODE}"' in text
    assert '"${QDRANT_RETRIEVAL_MODE}"' in text
    assert "per_keyword_rrf" not in text
    assert "--no-exclude-synonym-edges" not in text
    assert "check_synonym_manifest_ready" not in text
