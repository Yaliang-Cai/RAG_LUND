from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = (
    PROJECT_ROOT
    / "evaluate_local"
    / "MultiHopQA"
    / "run_hipporag2_ppr_hybrid_synonym_threshold.sh"
)


def _script_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_ppr_hybrid_synonym_threshold_runner_exists_and_reuses_existing_assets():
    text = _script_text()

    assert "multihopqa_hr2_v0_ppr_hybrid_syn_threshold" in text
    assert "build_index.py" not in text
    assert "download_hipporag2_datasets.py" not in text
    assert "manage_workspace_synonyms.py\" apply" in text
    assert 'file_prefix="2wikimultihopqa"' in text


def test_ppr_hybrid_synonym_threshold_runner_resolves_nested_or_flat_workspaces():
    text = _script_text()

    assert "resolve_working_dir()" in text
    assert 'local nested="${WORKSPACE_ROOT}/${dataset}/${workspace_id}"' in text
    assert 'local flat="${WORKSPACE_ROOT}/${dataset}"' in text
    assert 'WORKING_DIR="$(resolve_working_dir "${DATASET}" "${WORKSPACE_ID}")"' in text


def test_ppr_hybrid_synonym_threshold_runner_pins_threshold_and_eval_defaults():
    text = _script_text()

    for assignment in [
        'SYNONYM_THRESHOLDS="${SYNONYM_THRESHOLDS:-0.70 0.75 0.85 0.90}"',
        'CONCURRENCY="${CONCURRENCY:-200}"',
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

    assert 'threshold_label="${SYNONYM_THRESHOLD//./p}"' in text
    assert 'experiment="ppr_hybrid_syn_threshold_${threshold_label}"' in text
    assert 'local log_file="${SYNONYM_OPS_DIR}/${dataset}_syn${threshold_label}.log"' in text


def test_ppr_hybrid_synonym_threshold_runner_eval_flags_match_requested_semantics():
    text = _script_text()

    for flag in [
        '--modes "ppr"',
        '--qdrant-retrieval-mode "hybrid"',
        '--keyword-fanout-mode "joined"',
        "--no-ppr-enable-rerank",
        "--no-enable-kg-rerank",
        "--no-exclude-synonym-edges",
        '--answer-context-mode "chunk_only_prompt"',
        '--qa-prompt-style "semantic_cot"',
        '--answer-parse-mode "answer_marker"',
        "--bypass-query-cache",
        "--no-bypass-keywords-cache",
    ]:
        assert flag in text


def test_ppr_hybrid_synonym_threshold_runner_applies_and_checks_synonym_manifest():
    text = _script_text()

    assert '--workspace-id "${workspace_id}"' in text
    assert '--synonymy-threshold "${SYNONYM_THRESHOLD}"' in text
    assert "check_synonym_manifest_ready" in text
    assert "synonym_linking_manifest.json" in text
    assert 'payload.get("workspace_id")' in text
    assert 'payload.get("status") != "completed"' in text
    assert 'payload.get("synonymy_threshold")' in text
    assert 'payload.get("created_edges")' in text
