#!/usr/bin/env bash
set -euo pipefail

# MultiHopQA PPR hybrid-retrieval synonym threshold runner.
#
# Reuses existing V0 HippoRAG2 MultiHopQA workspaces. For each requested
# threshold, this runner rebuilds only SYNONYM edges, verifies the synonym
# manifest, then evaluates the same PPR + Qdrant hybrid retrieval configuration
# with the semantic passage CoT QA prompt.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"
DATA_DIR="${DATA_DIR:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/hipporag2_data}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0}"
RESULTS_ROOT="${RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_ppr_hybrid_syn_threshold}"
SYNONYM_OPS_DIR="${RESULTS_ROOT}/_synonym_ops"
SYNONYM_THRESHOLDS="${SYNONYM_THRESHOLDS:-0.70 0.75 0.85 0.90}"

INDEX_PROFILE="v0"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
export CHUNK_SIZE

RECALL_K="${RECALL_K:-2 5}"
CONCURRENCY="${CONCURRENCY:-200}"
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

profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
index_profile = profile_payload.get("index_profile") or {}
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
source_mapping = source_map.get("map") or {}

errors = []
if profile_payload.get("workspace_id") != expected_workspace:
    errors.append(f"profile.workspace_id={profile_payload.get('workspace_id')!r}")
if profile_payload.get("dataset") != expected_dataset:
    errors.append(f"profile.dataset={profile_payload.get('dataset')!r}")
if profile_payload.get("ablation_profile") != "v0":
    errors.append(f"profile.ablation_profile={profile_payload.get('ablation_profile')!r}")
if profile_payload.get("ablation_group") != "DB-only":
    errors.append(f"profile.ablation_group={profile_payload.get('ablation_group')!r}")
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
if manifest.get("ablation_profile") != "v0":
    errors.append(f"manifest.ablation_profile={manifest.get('ablation_profile')!r}")
if manifest.get("ablation_group") != "DB-only":
    errors.append(f"manifest.ablation_group={manifest.get('ablation_group')!r}")
if manifest.get("index_profile") != index_profile:
    errors.append("manifest.index_profile does not match profile.index_profile")
if int(manifest.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(f"manifest.chunk_token_size={manifest.get('chunk_token_size')!r}")
ingest_stats = manifest.get("ingest_stats") or {}
batch_count = int(manifest.get("batch_count") or 0)
successful = int(ingest_stats.get("successful_before_batch_count") or 0) + int(
    ingest_stats.get("successful_now_batch_count") or 0
)
failed = int(ingest_stats.get("failed_now_batch_count") or 0)
if failed:
    errors.append(f"ingest_stats.failed_now_batch_count={failed}")
if batch_count and successful != batch_count:
    errors.append(f"successful_batch_count={successful}, batch_count={batch_count}")
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
if source_map_size != len(source_mapping):
    errors.append(
        f"source_map.map_size={source_map_size}, actual_map_size={len(source_mapping)}"
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

synonym_manifest_path() {
    local working_dir="$1"
    local workspace_id="$2"
    local nested="${working_dir}/${workspace_id}/synonym_linking_manifest.json"
    local flat="${working_dir}/synonym_linking_manifest.json"

    if [[ -f "${nested}" ]]; then
        printf '%s\n' "${nested}"
    elif [[ -f "${flat}" ]]; then
        printf '%s\n' "${flat}"
    else
        die "Missing synonym manifest. Checked ${nested} and ${flat}."
    fi
}

check_synonym_manifest_ready() {
    local working_dir="$1"
    local workspace_id="$2"
    local threshold="$3"
    local manifest_path
    manifest_path="$(synonym_manifest_path "${working_dir}" "${workspace_id}")"

    python - "$manifest_path" "$workspace_id" "$threshold" <<'PY'
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

apply_synonym_edges() {
    local dataset="$1"
    local working_dir="$2"
    local workspace_id="$3"
    local threshold_label="$4"
    local log_file="${SYNONYM_OPS_DIR}/${dataset}_syn${threshold_label}.log"

    mkdir -p "${SYNONYM_OPS_DIR}"
    log "[${dataset}] Applying SYNONYM edges at threshold ${SYNONYM_THRESHOLD}"
    if ! python "${RAGANYTHING_ROOT}/scripts/manage_workspace_synonyms.py" apply \
        --workspace-path "${working_dir}" \
        --workspace-id "${workspace_id}" \
        --synonymy-threshold "${SYNONYM_THRESHOLD}" \
        >"${log_file}" 2>&1; then
        tail -80 "${log_file}" >&2 || true
        die "[${dataset}] SYNONYM edge apply failed. Full log: ${log_file}"
    fi
    check_synonym_manifest_ready "${working_dir}" "${workspace_id}" "${SYNONYM_THRESHOLD}"
    log "[${dataset}] SYNONYM apply log: ${log_file}"
}

run_ppr_hybrid_synonym_eval() {
    local dataset="$1"
    local working_dir="$2"
    local workspace_id="$3"
    local threshold_label="$4"
    local experiment="ppr_hybrid_syn_threshold_${threshold_label}"
    local output_dir="${RESULTS_ROOT}/${dataset}/${experiment}"
    local eval_resume_arg=()
    if [[ "${EVAL_RESUME}" == "1" ]]; then
        eval_resume_arg=(--resume)
    fi

    mkdir -p "${output_dir}"
    log "[${dataset}] ${experiment}: ppr + Qdrant hybrid + SYNONYM edges + semantic_cot"
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
        --no-exclude-synonym-edges \
        --keyword-fanout-mode "joined" \
        --qdrant-retrieval-mode "hybrid" \
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

print("dataset\texperiment\tmode\tem\tf1\trecall@2\trecall@5")
for dataset in datasets:
    dataset_root = root / dataset
    experiments = sorted(dataset_root.glob("ppr_hybrid_syn_threshold_*"))
    if not experiments:
        print(f"{dataset}\tMISSING\t\t\t\t\t")
        continue
    for experiment_dir in experiments:
        path = experiment_dir / f"{dataset}_summary.json"
        if not path.exists():
            print(f"{dataset}\t{experiment_dir.name}\tMISSING\t\t\t\t")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for mode, metrics in (payload.get("results") or {}).items():
            print(
                f"{dataset}\t{experiment_dir.name}\t{mode}\t"
                f"{metrics.get('em', '')}\t{metrics.get('f1', '')}\t"
                f"{metrics.get('recall@2', '')}\t{metrics.get('recall@5', '')}"
            )
PY
}

check_data_ready
mkdir -p "${RESULTS_ROOT}" "${SYNONYM_OPS_DIR}"

log "================================================================"
log "MultiHopQA PPR hybrid-retrieval synonym threshold sweep"
log "Workspace root:     ${WORKSPACE_ROOT}"
log "Results root:       ${RESULTS_ROOT}"
log "Synonym ops dir:    ${SYNONYM_OPS_DIR}"
log "Synonym thresholds: ${SYNONYM_THRESHOLDS}"
log "Profile:            ${INDEX_PROFILE}"
log "CHUNK_SIZE:         ${CHUNK_SIZE}"
log "Concurrency:        ${CONCURRENCY}"
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

    for SYNONYM_THRESHOLD in ${SYNONYM_THRESHOLDS}; do
        threshold_label="${SYNONYM_THRESHOLD//./p}"
        log "[${DATASET}] Threshold: ${SYNONYM_THRESHOLD}"
        apply_synonym_edges "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}" "${threshold_label}"
        run_ppr_hybrid_synonym_eval "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}" "${threshold_label}"
    done
done

log "================================================================"
log "Results summary"
log "================================================================"
print_summary
log "Done."
