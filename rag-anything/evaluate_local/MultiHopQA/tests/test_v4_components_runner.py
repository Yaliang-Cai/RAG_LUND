from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "evaluate_local" / "MultiHopQA" / "run_hipporag2_v4_components.sh"


def _script_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_v4_components_runner_exists_and_does_not_mutate_indexes_or_synonyms():
    text = _script_text()

    assert "multihopqa_hr2_v0_v4_components" in text
    assert "build_index.py" not in text
    assert "download_hipporag2_datasets.py" not in text
    assert "manage_workspace_synonyms.py" not in text
    assert "--synonymy-threshold" not in text
    assert "baseline_non_ppr" not in text
    assert "naive_dense" not in text
    assert "ppr_default" not in text
    assert "check_synonym_manifest_ready" not in text
    assert "synonym_linking_manifest.json" not in text
    assert 'file_prefix="2wikimultihopqa"' in text


def test_v4_components_runner_pins_multihop_windows():
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


def test_v4_components_runner_has_expected_nine_no_synonym_experiments():
    text = _script_text()

    for name in [
        "non_ppr_per_keyword",
        "non_ppr_kg_rerank",
        "non_ppr_retrieval_hybrid",
        "non_ppr_untruncated",
        "non_ppr_chunk_only",
        "ppr_per_keyword_no_rerank",
        "ppr_hybrid_no_rerank",
        "ppr_rerank",
        "ppr_raw_rerank_rrf",
    ]:
        assert name in text
    assert "ppr_default_with_synonym" not in text


def test_v4_components_runner_non_ppr_calls_match_docbench_v4_switches():
    text = _script_text()

    for call in [
        'run_non_ppr_experiment "non_ppr_per_keyword" "per_keyword_rrf" "dense" "--no-enable-kg-rerank" "kg_prompt" "truncated"',
        'run_non_ppr_experiment "non_ppr_kg_rerank" "joined" "dense" "--enable-kg-rerank" "kg_prompt" "truncated"',
        'run_non_ppr_experiment "non_ppr_retrieval_hybrid" "joined" "hybrid" "--no-enable-kg-rerank" "kg_prompt" "truncated"',
        'run_non_ppr_experiment "non_ppr_untruncated" "joined" "dense" "--no-enable-kg-rerank" "kg_prompt" "untruncated"',
        'run_non_ppr_experiment "non_ppr_chunk_only" "joined" "dense" "--no-enable-kg-rerank" "chunk_only_prompt" "truncated"',
    ]:
        assert call in text

    non_ppr_body = text.split("run_non_ppr_experiment() {", 1)[1].split(
        "run_ppr_experiment() {", 1
    )[0]
    assert "--hybrid-enable-rerank" in non_ppr_body
    assert "--exclude-synonym-edges" in non_ppr_body


def test_v4_components_runner_ppr_calls_match_docbench_v4_no_synonym_components():
    text = _script_text()

    for call in [
        'run_ppr_experiment "ppr_per_keyword_no_rerank" "per_keyword_rrf" "dense" "--no-ppr-enable-rerank" "--exclude-synonym-edges" "none"',
        'run_ppr_experiment "ppr_hybrid_no_rerank" "joined" "hybrid" "--no-ppr-enable-rerank" "--exclude-synonym-edges" "none"',
        'run_ppr_experiment "ppr_rerank" "joined" "dense" "--ppr-enable-rerank" "--exclude-synonym-edges" "none"',
        'run_ppr_experiment "ppr_raw_rerank_rrf" "joined" "dense" "--ppr-enable-rerank" "--exclude-synonym-edges" "raw_rrf"',
    ]:
        assert call in text

    ppr_body = text.split("run_ppr_experiment() {", 1)[1].split("print_summary() {", 1)[0]
    assert "--no-enable-kg-rerank" in ppr_body
    assert "--ppr-post-rerank-fusion" in text
    assert "--ppr-post-rerank-rrf-k" in text
