from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "evaluate_local" / "MultiHopQA" / "run_hipporag2_synonym_ablation.sh"


def _script_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_synonym_ablation_runner_exists_and_reuses_existing_workspaces():
    text = _script_text()

    assert "multihopqa_hr2_v0_syn0p80" in text
    assert "build_index.py" not in text
    assert "--workspace-id" in text
    assert '--workspace-id "${workspace_id}"' in text
    assert '--synonymy-threshold "${SYNONYM_THRESHOLD}"' in text
    assert 'file_prefix="2wikimultihopqa"' in text


def test_synonym_ablation_runner_has_expected_experiments():
    text = _script_text()

    for name in [
        "baseline_non_ppr_all_on",
        "non_ppr_no_chunk_only",
        "non_ppr_no_kg_rerank",
        "non_ppr_no_per_keyword",
        "non_ppr_no_retrieval_hybrid",
        "baseline_ppr_all_on",
        "ppr_no_rerank",
        "ppr_no_per_keyword",
        "ppr_no_retrieval_hybrid",
        "ppr_no_synonym_edges",
    ]:
        assert name in text


def test_synonym_ablation_runner_pins_multihop_windows_and_ppr_defaults():
    text = _script_text()

    for assignment in [
        'CONCURRENCY="${CONCURRENCY:-50}"',
        'TOP_K="${TOP_K:-10}"',
        'CHUNK_TOP_K="${CHUNK_TOP_K:-5}"',
        'NAIVE_TOP_K="${NAIVE_TOP_K:-10}"',
        'PPR_DAMPING="${PPR_DAMPING:-0.5}"',
        'PPR_TOP_K="${PPR_TOP_K:-50}"',
        'PPR_QA_TOP_K="${PPR_QA_TOP_K:-5}"',
        'PASSAGE_NODE_WEIGHT="${PASSAGE_NODE_WEIGHT:-0.05}"',
        'RECOGNITION_TOP_K="${RECOGNITION_TOP_K:-20}"',
        'LINKING_TOP_K="${LINKING_TOP_K:-5}"',
    ]:
        assert assignment in text


def test_synonym_ablation_runner_forces_expected_synonym_flags():
    text = _script_text()

    assert 'run_non_ppr_experiment "baseline_non_ppr_all_on"' in text
    assert 'run_ppr_experiment "baseline_ppr_all_on"' in text
    assert '--exclude-synonym-edges' in text
    assert '--no-exclude-synonym-edges' in text
    assert 'ppr_no_synonym_edges" "per_keyword_rrf" "hybrid" "--ppr-enable-rerank" "--exclude-synonym-edges"' in text


def test_synonym_ablation_runner_reads_evaluate_summary_results():
    text = _script_text()

    assert 'payload.get("results")' in text
