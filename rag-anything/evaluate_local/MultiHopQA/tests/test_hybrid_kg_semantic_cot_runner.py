from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = (
    PROJECT_ROOT
    / "evaluate_local"
    / "MultiHopQA"
    / "run_hipporag2_hybrid_kg_semantic_cot_eval.sh"
)


def _script_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_hybrid_kg_semantic_cot_runner_exists_and_reuses_existing_assets():
    text = _script_text()

    assert "multihopqa_hr2_v0_hybrid_kg_semantic_cot_eval" in text
    assert "build_index.py" not in text
    assert "download_hipporag2_datasets.py" not in text
    assert "manage_workspace_synonyms.py" not in text
    assert 'file_prefix="2wikimultihopqa"' in text
    assert "--qa-prompt-style \"kg_semantic_cot\"" in text
    assert "--answer-parse-mode \"answer_marker\"" in text
    assert "--no-bypass-keywords-cache" in text


def test_hybrid_kg_semantic_cot_runner_resolves_nested_or_flat_workspaces():
    text = _script_text()

    assert "resolve_working_dir()" in text
    assert 'local nested="${WORKSPACE_ROOT}/${dataset}/${workspace_id}"' in text
    assert 'local flat="${WORKSPACE_ROOT}/${dataset}"' in text
    assert 'WORKING_DIR="$(resolve_working_dir "${DATASET}" "${WORKSPACE_ID}")"' in text


def test_hybrid_kg_semantic_cot_runner_pins_hybrid_eval_defaults():
    text = _script_text()

    for assignment in [
        'CONCURRENCY="${CONCURRENCY:-50}"',
        'TOP_K="${TOP_K:-10}"',
        'CHUNK_TOP_K="${CHUNK_TOP_K:-5}"',
        'NAIVE_TOP_K="${NAIVE_TOP_K:-10}"',
        'MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-45000}"',
    ]:
        assert assignment in text

    for flag in [
        '--modes "hybrid"',
        '--qdrant-retrieval-mode "dense"',
        '--keyword-fanout-mode "joined"',
        '--kg-chunk-selection-source "truncated"',
        '--answer-context-mode "kg_prompt"',
        "--no-enable-kg-rerank",
        "--exclude-synonym-edges",
    ]:
        assert flag in text


def test_hybrid_kg_semantic_cot_runner_has_exact_requested_experiments():
    text = _script_text()

    assert '"hybrid_no_rerank_kg_semantic_cot"' in text
    assert '"hybrid_rerank_kg_semantic_cot"' in text
    assert "hybrid_no_rerank_lightrag_kg_prompt" not in text
    assert "semantic_cot" not in text.replace("kg_semantic_cot", "")

    assert (
        'run_hybrid_experiment "hybrid_no_rerank_kg_semantic_cot" '
        '"--no-hybrid-enable-rerank"'
    ) in text
    assert (
        'run_hybrid_experiment "hybrid_rerank_kg_semantic_cot" '
        '"--hybrid-enable-rerank"'
    ) in text
