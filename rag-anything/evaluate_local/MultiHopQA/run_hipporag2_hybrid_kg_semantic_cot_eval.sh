#!/usr/bin/env bash
set -euo pipefail

# MultiHopQA hybrid KG-semantic-CoT answer prompt runner.
#
# This runner reuses existing V0 HippoRAG2 MultiHopQA workspaces. It does not
# download data, rebuild indexes, or rebuild synonym edges. Retrieval remains
# the default hybrid KG path; only the answer prompt style changes.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"
DATA_DIR="${DATA_DIR:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/hipporag2_data}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0}"
RESULTS_ROOT="${RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_hybrid_kg_semantic_cot_eval}"

INDEX_PROFILE="v0"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
export CHUNK_SIZE

RECALL_K="${RECALL_K:-2 5}"
CONCURRENCY="${CONCURRENCY:-50}"
TOP_K="${TOP_K:-10}"
CHUNK_TOP_K="${CHUNK_TOP_K:-5}"
NAIVE_TOP_K="${NAIVE_TOP_K:-10}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-45000}"
MIN_RERANK_SCORE="${MIN_RERANK_SCORE:-0.3}"

EVAL_RESUME="${EVAL_RESUME:-0}"

DATASETS=("hotpotqa" "musique" "2wiki")
if [[ $# -ge 1 ]]; then
    DATASETS=("$1")
fi

EXPERIMENTS=(
    "hybrid_no_rerank_kg_semantic_cot"
    "hybrid_rerank_kg_semantic_cot"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

check_data_ready() {
    [[ -d "${DATA_DIR}" ]] || die "Missing HippoRAG2 data dir: ${DATA_DIR}"
    for dataset in "${DATASETS[@]}"; do
        local file_prefix="${dataset}"
        if [[ "${dataset}" == "2wiki" ]]; then
            file_prefix="2wikimultihopqa"
        fi
        [[ -f "${DATA_DIR}/${file_prefix}.json" ]] || die "Missing query file: ${DATA_DIR}/${file_prefix}.json"
        [[ -f "${DATA_DIR}/${file_prefix}_corpus.json" ]] || die "Missing corpus file: ${DATA_DIR}/${file_prefix}_corpus.json"
    done
}

resolve_working_dir() {
    local dataset="$1"
    local workspace_id="$2"
    local nested="${WORKSPACE_ROOT}/${dataset}/${workspace_id}"
    local flat="${WORKSPACE_ROOT}/${dataset}"

    if [[ -f "${nested}/multihopqa_index_profile.json" ]]; then
        printf '%s\n' "${nested}"
    elif [[ -f "${flat}/multihopqa_index_profile.json" ]]; then
        printf '%s\n' "${flat}"
    else
        die "Missing workspace artifacts. Checked ${nested} and ${flat}."
    fi
}

check_workspace_ready() {
    local working_dir="$1"
    local dataset="$2"
    local workspace_id="$3"
    local profile_path="${working_dir}/multihopqa_index_profile.json"
    local manifest_path="${working_dir}/multihopqa_ingest_manifest.json"
    local source_map_path="${working_dir}/multihopqa_chunk_source_map.json"
    [[ -f "${profile_path}" ]] || die "Missing index profile: ${profile_path}"
    [[ -f "${manifest_path}" ]] || die "Missing ingest manifest: ${manifest_path}"
    [[ -f "${source_map_path}" ]] || die "Missing chunk source map: ${source_map_path}"

    python - "$profile_path" "$manifest_path" "$source_map_path" "$CHUNK_SIZE" "$dataset" "$workspace_id" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
source_map_path = Path(sys.argv[3])
expected_chunk_size = int(sys.argv[4])
expected_dataset = sys.argv[5]
expected_workspace = sys.argv[6]

profile = json.loads(profile_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
index_profile = profile.get("index_profile") or {}
ingest_stats = manifest.get("ingest_stats") or {}
source_payload = source_map.get("map") or {}

errors = []
if profile.get("workspace_id") != expected_workspace:
    errors.append(f"profile.workspace_id={profile.get('workspace_id')!r}")
if profile.get("dataset") != expected_dataset:
    errors.append(f"profile.dataset={profile.get('dataset')!r}")
if profile.get("ablation_profile") != "v0":
    errors.append(f"profile.ablation_profile={profile.get('ablation_profile')!r}")
if index_profile.get("profile_key") != "v0":
    errors.append(f"index_profile.profile_key={index_profile.get('profile_key')!r}")
if index_profile.get("enable_entity_disambiguation") is not False:
    errors.append(
        "index_profile.enable_entity_disambiguation="
        f"{index_profile.get('enable_entity_disambiguation')!r}"
    )
if index_profile.get("enable_synonym_linking") is not False:
    errors.append(
        f"index_profile.enable_synonym_linking={index_profile.get('enable_synonym_linking')!r}"
    )
if int(index_profile.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(
        f"index_profile.chunk_token_size={index_profile.get('chunk_token_size')!r}"
    )
if manifest.get("workspace_id") != expected_workspace:
    errors.append(f"manifest.workspace_id={manifest.get('workspace_id')!r}")
if manifest.get("dataset") != expected_dataset:
    errors.append(f"manifest.dataset={manifest.get('dataset')!r}")
if manifest.get("corpus_source") != "hipporag2":
    errors.append(f"manifest.corpus_source={manifest.get('corpus_source')!r}")
if manifest.get("n_samples") != 0 or manifest.get("seed") != 0:
    errors.append(
        f"manifest identity n_samples={manifest.get('n_samples')!r}, "
        f"seed={manifest.get('seed')!r}"
    )
if manifest.get("index_profile") != index_profile:
    errors.append("manifest.index_profile does not match profile.index_profile")
if int(manifest.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(f"manifest.chunk_token_size={manifest.get('chunk_token_size')!r}")
if int(ingest_stats.get("failed_now_batch_count") or 0) != 0:
    errors.append(
        f"ingest_stats.failed_now_batch_count={ingest_stats.get('failed_now_batch_count')!r}"
    )
if source_map.get("workspace_id") != expected_workspace:
    errors.append(f"source_map.workspace_id={source_map.get('workspace_id')!r}")
if source_map.get("dataset") != expected_dataset:
    errors.append(f"source_map.dataset={source_map.get('dataset')!r}")
if source_map.get("n_samples") != 0 or source_map.get("seed") != 0:
    errors.append(
        f"source_map identity n_samples={source_map.get('n_samples')!r}, "
        f"seed={source_map.get('seed')!r}"
    )
source_map_size = int(source_map.get("map_size") or 0)
if source_map_size <= 0:
    errors.append(f"source_map.map_size={source_map.get('map_size')!r}")
if source_map_size != len(source_payload):
    errors.append(
        f"source_map.map_size={source_map_size}, actual_map_size={len(source_payload)}"
    )
expected_chunk_total = int(manifest.get("expected_chunk_total") or 0)
if expected_chunk_total and expected_chunk_total != source_map_size:
    errors.append(
        f"manifest.expected_chunk_total={expected_chunk_total}, "
        f"source_map.map_size={source_map_size}"
    )

if errors:
    raise SystemExit("Workspace check failed: " + "; ".join(errors))
PY
}

run_hybrid_experiment() {
    local experiment="$1"
    local rerank_flag="$2"
    local dataset="$3"
    local working_dir="$4"
    local workspace_id="$5"
    local output_dir="${RESULTS_ROOT}/${dataset}/${experiment}"
    local eval_resume_arg=()
    if [[ "${EVAL_RESUME}" == "1" ]]; then
        eval_resume_arg=(--resume)
    fi

    mkdir -p "${output_dir}"
    log "[${dataset}] ${experiment}: hybrid, ${rerank_flag}, kg_semantic_cot"
    RAGANYTHING_MIN_RERANK_SCORE="${MIN_RERANK_SCORE}" \
    MIN_RERANK_SCORE="${MIN_RERANK_SCORE}" \
    python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/evaluate_multihop.py" \
        --dataset "${dataset}" \
        --workspace "${workspace_id}" \
        --working-dir "${working_dir}" \
        --hipporag2-data-dir "${DATA_DIR}" \
        --output-dir "${output_dir}" \
        --modes "hybrid" \
        --recall-k ${RECALL_K} \
        --concurrency "${CONCURRENCY}" \
        --top-k "${TOP_K}" \
        --chunk-top-k "${CHUNK_TOP_K}" \
        --naive-top-k "${NAIVE_TOP_K}" \
        --max-total-tokens "${MAX_TOTAL_TOKENS}" \
        --qdrant-retrieval-mode "dense" \
        --keyword-fanout-mode "joined" \
        --kg-chunk-selection-source "truncated" \
        --answer-context-mode "kg_prompt" \
        --no-enable-kg-rerank \
        "${rerank_flag}" \
        --exclude-synonym-edges \
        --qa-prompt-style "kg_semantic_cot" \
        --answer-parse-mode "answer_marker" \
        --bypass-query-cache \
        --no-bypass-keywords-cache \
        "${eval_resume_arg[@]}"
}

print_summary() {
    python - "$RESULTS_ROOT" "${DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
datasets = sys.argv[2:]
experiments = [
    "hybrid_no_rerank_kg_semantic_cot",
    "hybrid_rerank_kg_semantic_cot",
]

print("dataset\texperiment\tmode\tem\tf1\trecall@2\trecall@5")
for dataset in datasets:
    for experiment in experiments:
        path = root / dataset / experiment / f"{dataset}_summary.json"
        if not path.exists():
            print(f"{dataset}\t{experiment}\tMISSING\t\t\t\t")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for mode, metrics in (payload.get("results") or {}).items():
            print(
                f"{dataset}\t{experiment}\t{mode}\t"
                f"{metrics.get('em', '')}\t{metrics.get('f1', '')}\t"
                f"{metrics.get('recall@2', '')}\t{metrics.get('recall@5', '')}"
            )
PY
}

check_data_ready
mkdir -p "${RESULTS_ROOT}"

log "================================================================"
log "MultiHopQA hybrid KG-semantic-CoT prompt eval"
log "Workspace root:     ${WORKSPACE_ROOT}"
log "Results root:       ${RESULTS_ROOT}"
log "Profile:            ${INDEX_PROFILE}"
log "CHUNK_SIZE:         ${CHUNK_SIZE}"
log "Concurrency:        ${CONCURRENCY}"
log "Experiments:        ${EXPERIMENTS[*]}"
log "================================================================"

for DATASET in "${DATASETS[@]}"; do
    WORKSPACE_ID="${DATASET}_hr2_v0"
    WORKING_DIR="$(resolve_working_dir "${DATASET}" "${WORKSPACE_ID}")"

    log "================================================================"
    log "Dataset:      ${DATASET}"
    log "Workspace ID: ${WORKSPACE_ID}"
    log "Working dir:  ${WORKING_DIR}"
    log "================================================================"

    check_workspace_ready "${WORKING_DIR}" "${DATASET}" "${WORKSPACE_ID}"
    run_hybrid_experiment "hybrid_no_rerank_kg_semantic_cot" "--no-hybrid-enable-rerank" "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}"
    run_hybrid_experiment "hybrid_rerank_kg_semantic_cot" "--hybrid-enable-rerank" "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}"
done

log "================================================================"
log "Results summary"
log "================================================================"
print_summary
log "Done."
