from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "evaluate_local" / "MultiHopQA" / "run_hipporag2_v4_prompt_eval.sh"


def _script_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_v4_prompt_eval_runner_exists_and_only_uses_existing_assets():
    text = _script_text()

    assert "multihopqa_hr2_v0_v4_components_semantic_prompt" in text
    assert "build_index.py" not in text
    assert "download_hipporag2_datasets.py" not in text
    assert "manage_workspace_synonyms.py" not in text
    assert 'file_prefix="2wikimultihopqa"' in text
    assert "--qa-prompt-style \"semantic_cot\"" in text
    assert "--answer-parse-mode \"answer_marker\"" in text
    assert "--no-bypass-keywords-cache" in text


def test_v4_prompt_eval_runner_pins_expected_defaults():
    text = _script_text()

    for assignment in [
        'CONCURRENCY="${CONCURRENCY:-50}"',
        'TOP_K="${TOP_K:-10}"',
        'CHUNK_TOP_K="${CHUNK_TOP_K:-5}"',
        'NAIVE_TOP_K="${NAIVE_TOP_K:-10}"',
        'MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-45000}"',
        'PPR_DAMPING="${PPR_DAMPING:-0.5}"',
        'PPR_TOP_K="${PPR_TOP_K:-50}"',
        'PPR_QA_TOP_K="${PPR_QA_TOP_K:-5}"',
        'PASSAGE_NODE_WEIGHT="${PASSAGE_NODE_WEIGHT:-0.05}"',
        'RECOGNITION_TOP_K="${RECOGNITION_TOP_K:-20}"',
        'LINKING_TOP_K="${LINKING_TOP_K:-5}"',
    ]:
        assert assignment in text


def test_v4_prompt_eval_runner_has_four_ppr_component_experiments():
    text = _script_text()

    for name in [
        "ppr_default",
        "ppr_hybrid_no_rerank",
        "ppr_per_keyword_no_rerank",
        "ppr_default_with_synonym",
    ]:
        assert name in text
    for skipped in [
        "non_ppr_per_keyword",
        "ppr_rerank",
        "ppr_raw_rerank_rrf",
    ]:
        assert skipped not in text


def test_v4_prompt_eval_runner_ppr_flags_match_requested_components():
    text = _script_text()

    for call in [
        'run_ppr_experiment "ppr_default" "joined" "dense" "--exclude-synonym-edges"',
        'run_ppr_experiment "ppr_hybrid_no_rerank" "joined" "hybrid" "--exclude-synonym-edges"',
        'run_ppr_experiment "ppr_per_keyword_no_rerank" "per_keyword_rrf" "dense" "--exclude-synonym-edges"',
        'run_ppr_experiment "ppr_default_with_synonym" "joined" "dense" "--no-exclude-synonym-edges"',
    ]:
        assert call in text

    assert "--no-enable-kg-rerank" in text
    assert "--no-ppr-enable-rerank" in text
    assert '--ppr-post-rerank-fusion "none"' in text
    assert "check_synonym_manifest_ready" in text
    assert "synonym_linking_manifest.json" in text
