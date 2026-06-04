from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "evaluate_local" / "MultiHopQA" / "run_hipporag2_agentic_eval.sh"


def _script_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_agentic_eval_runner_exists_and_reuses_existing_workspaces():
    text = _script_text()

    assert "multihopqa_hr2_v0_agentic_eval" in text
    assert "build_index.py" not in text
    assert "download_hipporag2_datasets.py" not in text
    assert "manage_workspace_synonyms.py" not in text
    assert 'file_prefix="2wikimultihopqa"' in text


def test_agentic_eval_runner_has_expected_experiments_and_defaults():
    text = _script_text()

    assert '"agentic_chunk5:5:5"' in text
    assert '"agentic_chunk10:10:10"' in text
    for assignment in [
        'CONCURRENCY="${CONCURRENCY:-50}"',
        'TOP_K="${TOP_K:-10}"',
        'NAIVE_TOP_K="${NAIVE_TOP_K:-10}"',
        'RECALL_K="${RECALL_K:-2 5 10}"',
        'TEXT_REQUEST_TIMEOUT_SECONDS="${TEXT_REQUEST_TIMEOUT_SECONDS:-3600}"',
        'PPR_TOP_K="${PPR_TOP_K:-50}"',
        'PPR_DAMPING="${PPR_DAMPING:-0.5}"',
        'PASSAGE_NODE_WEIGHT="${PASSAGE_NODE_WEIGHT:-0.05}"',
        'RECOGNITION_TOP_K="${RECOGNITION_TOP_K:-20}"',
        'LINKING_TOP_K="${LINKING_TOP_K:-5}"',
        'MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-45000}"',
        'QDRANT_RETRIEVAL_MODE="${QDRANT_RETRIEVAL_MODE:-hybrid}"',
        'KEYWORD_FANOUT_MODE="${KEYWORD_FANOUT_MODE:-joined}"',
        'KG_CHUNK_SELECTION_SOURCE="${KG_CHUNK_SELECTION_SOURCE:-truncated}"',
        'QA_PROMPT_STYLE="${QA_PROMPT_STYLE:-semantic_cot}"',
        'ANSWER_PARSE_MODE="${ANSWER_PARSE_MODE:-answer_marker}"',
    ]:
        assert assignment in text


def test_agentic_eval_runner_eval_flags_match_requested_semantics():
    text = _script_text()

    for flag in [
        '--modes "agentic"',
        '--qdrant-retrieval-mode "${QDRANT_RETRIEVAL_MODE}"',
        '--keyword-fanout-mode "${KEYWORD_FANOUT_MODE}"',
        '--kg-chunk-selection-source "${KG_CHUNK_SELECTION_SOURCE}"',
        "--no-enable-kg-rerank",
        "--no-hybrid-enable-rerank",
        "--no-ppr-enable-rerank",
        '--qa-prompt-style "${QA_PROMPT_STYLE}"',
        '--answer-parse-mode "${ANSWER_PARSE_MODE}"',
        "--bypass-query-cache",
        "--no-bypass-keywords-cache",
    ]:
        assert flag in text

    assert 'RAGANYTHING_TEXT_REQUEST_TIMEOUT_SECONDS="${TEXT_REQUEST_TIMEOUT_SECONDS}"' in text

    assert "--exclude-synonym-edges" not in text
    assert "--no-exclude-synonym-edges" not in text


def test_agentic_eval_runner_checks_synonym_manifest_without_applying():
    text = _script_text()

    assert "check_synonym_manifest_ready" in text
    assert "synonym_linking_manifest.json" in text
    assert 'SYNONYM_THRESHOLD="${SYNONYM_THRESHOLD:-0.8}"' in text
    assert 'payload.get("workspace_id")' in text
    assert 'payload.get("status") != "completed"' in text
    assert 'payload.get("synonymy_threshold")' in text
