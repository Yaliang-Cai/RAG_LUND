#!/usr/bin/env bash
set -euo pipefail

# MultiHopQA V4 PPR component runner with semantic passage CoT QA prompt.
#
# This runner reuses existing V0 HippoRAG2 MultiHopQA workspaces. It does not
# download data, build indexes, or rebuild SYNONYM edges. The synonym component
# only runs after an existing threshold-0.8 synonym manifest is present.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"
DATA_DIR="${DATA_DIR:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/hipporag2_data}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0}"
RESULTS_ROOT="${RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_v4_components_semantic_prompt}"
SYNONYM_THRESHOLD="${SYNONYM_THRESHOLD:-0.8}"

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

PPR_DAMPING="${PPR_DAMPING:-0.5}"
PPR_TOP_K="${PPR_TOP_K:-50}"
PPR_QA_TOP_K="${PPR_QA_TOP_K:-5}"
PASSAGE_NODE_WEIGHT="${PASSAGE_NODE_WEIGHT:-0.05}"
RECOGNITION_TOP_K="${RECOGNITION_TOP_K:-20}"
LINKING_TOP_K="${LINKING_TOP_K:-5}"
PPR_POST_RERANK_RRF_K="${PPR_POST_RERANK_RRF_K:-60}"
PPR_SYNONYM_WEIGHT_MODE="${PPR_SYNONYM_WEIGHT_MODE:-raw}"

EVAL_RESUME="${EVAL_RESUME:-0}"

DATASETS=("hotpotqa" "musique" "2wiki")
if [[ $# -ge 1 ]]; then
    DATASETS=("$1")
fi

EXPERIMENTS=(
    "ppr_default"
    "ppr_hybrid_no_rerank"
    "ppr_per_keyword_no_rerank"
    "ppr_default_with_synonym"
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

errors = []
if profile.get("workspace") != expected_workspace:
    errors.append(f"profile.workspace={profile.get('workspace')!r}")
if profile.get("dataset") != expected_dataset:
    errors.append(f"profile.dataset={profile.get('dataset')!r}")
if profile.get("ablation_profile") != "v0":
    errors.append(f"ablation_profile={profile.get('ablation_profile')!r}")
if profile.get("corpus_source") != "hipporag2":
    errors.append(f"corpus_source={profile.get('corpus_source')!r}")
if int(profile.get("n_samples") or 0) != 0:
    errors.append(f"n_samples={profile.get('n_samples')!r}")
if int(profile.get("seed") or -1) != 0:
    errors.append(f"seed={profile.get('seed')!r}")
if profile.get("enable_synonym_linking") is not False:
    errors.append(f"enable_synonym_linking={profile.get('enable_synonym_linking')!r}")
if int(profile.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(
        f"chunk_token_size={profile.get('chunk_token_size')!r}, expected={expected_chunk_size}"
    )
if manifest.get("status") != "completed":
    errors.append(f"manifest.status={manifest.get('status')!r}")
if manifest.get("workspace_id") != expected_workspace:
    errors.append(f"manifest.workspace_id={manifest.get('workspace_id')!r}")
if source_map.get("workspace_id") != expected_workspace:
    errors.append(f"source_map.workspace_id={source_map.get('workspace_id')!r}")
if source_map.get("dataset") != expected_dataset:
    errors.append(f"source_map.dataset={source_map.get('dataset')!r}")
if int(source_map.get("n_samples") or 0) != 0:
    errors.append(f"source_map.n_samples={source_map.get('n_samples')!r}")
if int(source_map.get("seed") or -1) != 0:
    errors.append(f"source_map.seed={source_map.get('seed')!r}")
if int(source_map.get("map_size") or 0) <= 0:
    errors.append(f"source_map.map_size={source_map.get('map_size')!r}")

if errors:
    raise SystemExit("Workspace check failed: " + "; ".join(errors))
PY
}

check_synonym_manifest_ready() {
    local working_dir="$1"
    local workspace_id="$2"
    local nested_manifest_path="${working_dir}/${workspace_id}/synonym_linking_manifest.json"
    local flat_manifest_path="${working_dir}/synonym_linking_manifest.json"
    local manifest_path=""
    if [[ -f "${nested_manifest_path}" ]]; then
        manifest_path="${nested_manifest_path}"
    elif [[ -f "${flat_manifest_path}" ]]; then
        manifest_path="${flat_manifest_path}"
    else
        die "Missing synonym manifest. Checked ${nested_manifest_path} and ${flat_manifest_path}."
    fi

    python - "$manifest_path" "$workspace_id" "$SYNONYM_THRESHOLD" <<'PY'
import json
import math
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_workspace = sys.argv[2]
expected_threshold = float(sys.argv[3])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))

errors = []
if payload.get("workspace_id") != expected_workspace:
    errors.append(f"workspace_id={payload.get('workspace_id')!r}")
if payload.get("status") != "completed":
    errors.append(f"status={payload.get('status')!r}")
try:
    threshold = float(payload.get("synonymy_threshold"))
except (TypeError, ValueError):
    threshold = math.nan
if not math.isclose(threshold, expected_threshold, rel_tol=0.0, abs_tol=1e-9):
    errors.append(f"synonymy_threshold={payload.get('synonymy_threshold')!r}")
if int(payload.get("created_edges") or 0) <= 0:
    errors.append(f"created_edges={payload.get('created_edges')!r}")

if errors:
    raise SystemExit("Synonym manifest check failed: " + "; ".join(errors))
PY
}

run_ppr_experiment() {
    local experiment="$1"
    local keyword_fanout_mode="$2"
    local qdrant_retrieval_mode="$3"
    local synonym_flag="$4"
    local dataset="$5"
    local working_dir="$6"
    local workspace_id="$7"
    local output_dir="${RESULTS_ROOT}/${dataset}/${experiment}"
    local eval_resume_arg=()
    if [[ "${EVAL_RESUME}" == "1" ]]; then
        eval_resume_arg=(--resume)
    fi

    mkdir -p "${output_dir}"
    log "[${dataset}] ${experiment}: ppr, keyword_fanout=${keyword_fanout_mode}, qdrant=${qdrant_retrieval_mode}, ${synonym_flag}"
    RAGANYTHING_MIN_RERANK_SCORE="${MIN_RERANK_SCORE}" \
    MIN_RERANK_SCORE="${MIN_RERANK_SCORE}" \
    python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/evaluate_multihop.py" \
        --dataset "${dataset}" \
        --workspace "${workspace_id}" \
        --working-dir "${working_dir}" \
        --hipporag2-data-dir "${DATA_DIR}" \
        --output-dir "${output_dir}" \
        --modes "ppr" \
        --recall-k ${RECALL_K} \
        --concurrency "${CONCURRENCY}" \
        --top-k "${TOP_K}" \
        --chunk-top-k "${CHUNK_TOP_K}" \
        --naive-top-k "${NAIVE_TOP_K}" \
        --max-total-tokens "${MAX_TOTAL_TOKENS}" \
        --ppr-damping "${PPR_DAMPING}" \
        --ppr-top-k "${PPR_TOP_K}" \
        --ppr-qa-top-k "${PPR_QA_TOP_K}" \
        --passage-node-weight "${PASSAGE_NODE_WEIGHT}" \
        --recognition-top-k "${RECOGNITION_TOP_K}" \
        --linking-top-k "${LINKING_TOP_K}" \
        --ppr-post-rerank-fusion "none" \
        --ppr-post-rerank-rrf-k "${PPR_POST_RERANK_RRF_K}" \
        --ppr-synonym-weight-mode "${PPR_SYNONYM_WEIGHT_MODE}" \
        --no-enable-kg-rerank \
        --no-ppr-enable-rerank \
        "${synonym_flag}" \
        --keyword-fanout-mode "${keyword_fanout_mode}" \
        --qdrant-retrieval-mode "${qdrant_retrieval_mode}" \
        --answer-context-mode "chunk_only_prompt" \
        --qa-prompt-style "semantic_cot" \
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
    "ppr_default",
    "ppr_hybrid_no_rerank",
    "ppr_per_keyword_no_rerank",
    "ppr_default_with_synonym",
]

print("dataset\texperiment\tmode\tem\tf1\trecall@2\trecall@5")
for dataset in datasets:
    for experiment in experiments:
        path = root / dataset / experiment / f"{dataset}_summary.json"
        if not path.exists():
            print(f"{dataset}\t{experiment}\tMISSING\t\t\t\t")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results") or {}
        if not results:
            print(f"{dataset}\t{experiment}\tNO_MODES\t\t\t\t")
            continue
        for mode, metrics in results.items():
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
log "MultiHopQA V4 PPR components with semantic passage CoT QA prompt"
log "Workspace root:     ${WORKSPACE_ROOT}"
log "Results root:       ${RESULTS_ROOT}"
log "Synonym threshold:  ${SYNONYM_THRESHOLD}"
log "Profile:            ${INDEX_PROFILE}"
log "CHUNK_SIZE:         ${CHUNK_SIZE}"
log "Concurrency:        ${CONCURRENCY}"
log "Experiments:        ${EXPERIMENTS[*]}"
log "================================================================"

for DATASET in "${DATASETS[@]}"; do
    WORKSPACE_ID="${DATASET}_hr2_v0"
    WORKING_DIR="${WORKSPACE_ROOT}/${DATASET}"

    log "================================================================"
    log "Dataset:      ${DATASET}"
    log "Workspace ID: ${WORKSPACE_ID}"
    log "Working dir:  ${WORKING_DIR}"
    log "================================================================"

    check_workspace_ready "${WORKING_DIR}" "${DATASET}" "${WORKSPACE_ID}"

    run_ppr_experiment "ppr_default" "joined" "dense" "--exclude-synonym-edges" "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}"
    run_ppr_experiment "ppr_hybrid_no_rerank" "joined" "hybrid" "--exclude-synonym-edges" "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}"
    run_ppr_experiment "ppr_per_keyword_no_rerank" "per_keyword_rrf" "dense" "--exclude-synonym-edges" "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}"
    check_synonym_manifest_ready "${WORKING_DIR}" "${WORKSPACE_ID}"
    run_ppr_experiment "ppr_default_with_synonym" "joined" "dense" "--no-exclude-synonym-edges" "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}"
done

log "================================================================"
log "Results summary"
log "================================================================"
print_summary
log "Done."
