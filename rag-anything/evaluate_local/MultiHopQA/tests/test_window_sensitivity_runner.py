from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "evaluate_local" / "MultiHopQA" / "run_hipporag2_window_sensitivity.sh"


def _script_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_window_sensitivity_runner_is_marginal_only():
    text = _script_text()

    assert "multihopqa_hr2_v0_window_sensitivity" in text
    assert "build_index.py" not in text
    assert "manage_workspace_synonyms.py" not in text
    assert "--synonymy-threshold" not in text
    assert "BASELINE_RESULTS_ROOT" not in text
    assert "check_baseline_summaries_ready" not in text
    assert "baseline_non_ppr_all_on" not in text
    assert "baseline_ppr_all_on" not in text


def test_window_sensitivity_runner_checks_existing_workspace_and_synonyms():
    text = _script_text()

    assert "check_workspace_ready" in text
    assert "check_synonym_manifest_ready" in text
    assert 'SYNONYM_THRESHOLD="${SYNONYM_THRESHOLD:-0.8}"' in text
    assert "synonym_linking_manifest.json" in text
    assert 'file_prefix="2wikimultihopqa"' in text
    assert '"created_edges"' in text


def test_window_sensitivity_runner_has_expected_experiments():
    text = _script_text()

    for call in [
        'run_non_ppr_experiment "non_ppr_top20_chunk5" "20" "${BASE_CHUNK_TOP_K}" "${BASE_MIN_RERANK_SCORE}"',
        'run_non_ppr_experiment "non_ppr_top30_chunk5" "30" "${BASE_CHUNK_TOP_K}" "${BASE_MIN_RERANK_SCORE}"',
        'run_non_ppr_experiment "non_ppr_top10_chunk10" "${BASE_TOP_K}" "10" "${BASE_MIN_RERANK_SCORE}"',
        'run_non_ppr_experiment "non_ppr_top10_chunk5_min0p0" "${BASE_TOP_K}" "${BASE_CHUNK_TOP_K}" "0.0"',
        'run_ppr_experiment "ppr_top20_ppr50_qa5" "20" "${BASE_PPR_TOP_K}" "${BASE_PPR_QA_TOP_K}" "${BASE_MIN_RERANK_SCORE}"',
        'run_ppr_experiment "ppr_top30_ppr50_qa5" "30" "${BASE_PPR_TOP_K}" "${BASE_PPR_QA_TOP_K}" "${BASE_MIN_RERANK_SCORE}"',
        'run_ppr_experiment "ppr_top10_ppr100_qa5" "${BASE_TOP_K}" "100" "${BASE_PPR_QA_TOP_K}" "${BASE_MIN_RERANK_SCORE}"',
        'run_ppr_experiment "ppr_top10_ppr50_qa10" "${BASE_TOP_K}" "${BASE_PPR_TOP_K}" "10" "${BASE_MIN_RERANK_SCORE}"',
        'run_ppr_experiment "ppr_top10_ppr50_qa5_min0p0" "${BASE_TOP_K}" "${BASE_PPR_TOP_K}" "${BASE_PPR_QA_TOP_K}" "0.0"',
    ]:
        assert call in text


def test_window_sensitivity_runner_pins_common_defaults():
    text = _script_text()

    for assignment in [
        'CONCURRENCY="${CONCURRENCY:-50}"',
        'BASE_TOP_K="${BASE_TOP_K:-10}"',
        'BASE_CHUNK_TOP_K="${BASE_CHUNK_TOP_K:-5}"',
        'NAIVE_TOP_K="${NAIVE_TOP_K:-10}"',
        'BASE_MIN_RERANK_SCORE="${BASE_MIN_RERANK_SCORE:-0.3}"',
        'PPR_DAMPING="${PPR_DAMPING:-0.5}"',
        'BASE_PPR_TOP_K="${BASE_PPR_TOP_K:-50}"',
        'BASE_PPR_QA_TOP_K="${BASE_PPR_QA_TOP_K:-5}"',
        'PASSAGE_NODE_WEIGHT="${PASSAGE_NODE_WEIGHT:-0.05}"',
        'RECOGNITION_TOP_K="${RECOGNITION_TOP_K:-20}"',
        'LINKING_TOP_K="${LINKING_TOP_K:-5}"',
    ]:
        assert assignment in text


def test_window_sensitivity_runner_sets_rerank_threshold_per_experiment():
    text = _script_text()

    assert 'RAGANYTHING_MIN_RERANK_SCORE="${min_rerank_score}" \\' in text
    assert 'MIN_RERANK_SCORE="${min_rerank_score}" \\' in text
    assert "--naive-top-k" in text
    assert '"${NAIVE_TOP_K}"' in text


def test_window_sensitivity_runner_forces_synonym_semantics():
    text = _script_text()

    non_ppr_body = text.split("run_non_ppr_experiment() {", 1)[1].split("run_ppr_experiment() {", 1)[0]
    ppr_body = text.split("run_ppr_experiment() {", 1)[1].split("print_summary() {", 1)[0]

    assert "--exclude-synonym-edges" in non_ppr_body
    assert "--enable-kg-rerank" in non_ppr_body
    assert "--hybrid-enable-rerank" in non_ppr_body
    assert '--keyword-fanout-mode "per_keyword_rrf"' in non_ppr_body
    assert '--qdrant-retrieval-mode "hybrid"' in non_ppr_body
    assert '--answer-context-mode "chunk_only_prompt"' in non_ppr_body

    assert "--no-exclude-synonym-edges" in ppr_body
    assert "--ppr-enable-rerank" in ppr_body
    assert "--no-enable-kg-rerank" in ppr_body
    assert '--keyword-fanout-mode "per_keyword_rrf"' in ppr_body
    assert '--qdrant-retrieval-mode "hybrid"' in ppr_body
    assert '--answer-context-mode "chunk_only_prompt"' in ppr_body
