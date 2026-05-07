from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "evaluate_local" / "MultiHopQA" / "run_hipporag2_v4_ppr_synonym.sh"


def _script_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_v4_ppr_synonym_runner_exists_and_only_runs_synonym_default():
    text = _script_text()

    assert "multihopqa_hr2_v0_v4_components" in text
    assert 'EXPERIMENT="ppr_default_with_synonym"' in text
    assert "non_ppr_per_keyword" not in text
    assert "ppr_raw_rerank_rrf" not in text
    assert "build_index.py" not in text
    assert "download_hipporag2_datasets.py" not in text
    assert "manage_workspace_synonyms.py" not in text
    assert "--synonymy-threshold" not in text
    assert 'file_prefix="2wikimultihopqa"' in text


def test_v4_ppr_synonym_runner_pins_multihop_windows():
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
        'MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-45000}"',
        'RECALL_K="${RECALL_K:-2 5}"',
    ]:
        assert assignment in text


def test_v4_ppr_synonym_runner_requires_completed_threshold_0p8_manifest():
    text = _script_text()

    assert "check_synonym_manifest_ready" in text
    assert "synonym_linking_manifest.json" in text
    assert 'SYNONYM_THRESHOLD="${SYNONYM_THRESHOLD:-0.8}"' in text
    assert "Run run_hipporag2_synonym_ablation.sh first" in text
    assert '"created_edges"' in text


def test_v4_ppr_synonym_runner_uses_synonym_edges_and_default_ppr_settings():
    text = _script_text()

    assert '--modes "ppr"' in text
    assert '--keyword-fanout-mode "joined"' in text
    assert '--qdrant-retrieval-mode "dense"' in text
    assert "--no-ppr-enable-rerank" in text
    assert "--no-enable-kg-rerank" in text
    assert "--no-exclude-synonym-edges" in text
    assert '--ppr-post-rerank-fusion "none"' in text
    assert '--answer-context-mode "chunk_only_prompt"' in text
