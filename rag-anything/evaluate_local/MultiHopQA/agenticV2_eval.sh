#!/usr/bin/env bash
set -euo pipefail

# MultiHopQA agentic v2 retrieval eval.
#
# This runner evaluates the already-built V0 MultiHopQA workspaces. It assumes
# the workspaces already contain completed 0.8 SYNONYM manifests because the
# agentic_v2 PPR/full_v2 profiles use synonym edges through profile-level path
# overrides.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"
DATA_DIR="${DATA_DIR:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/hipporag2_data}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0}"
RESULTS_ROOT="${RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_agentic_v2_eval}"

INDEX_PROFILE="v0"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
export CHUNK_SIZE

RECALL_K="${RECALL_K:-2 5 10}"
CONCURRENCY="${CONCURRENCY:-50}"
TOP_K="${TOP_K:-10}"
NAIVE_TOP_K="${NAIVE_TOP_K:-10}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-45000}"
TEXT_REQUEST_TIMEOUT_SECONDS="${TEXT_REQUEST_TIMEOUT_SECONDS:-3600}"

PPR_DAMPING="${PPR_DAMPING:-0.5}"
PPR_TOP_K="${PPR_TOP_K:-50}"
PASSAGE_NODE_WEIGHT="${PASSAGE_NODE_WEIGHT:-0.05}"
RECOGNITION_TOP_K="${RECOGNITION_TOP_K:-20}"
LINKING_TOP_K="${LINKING_TOP_K:-5}"
PPR_SYNONYM_WEIGHT_MODE="${PPR_SYNONYM_WEIGHT_MODE:-raw}"
SYNONYM_THRESHOLD="${SYNONYM_THRESHOLD:-0.8}"

QDRANT_RETRIEVAL_MODE="${QDRANT_RETRIEVAL_MODE:-hybrid}"
KEYWORD_FANOUT_MODE="${KEYWORD_FANOUT_MODE:-joined}"
KG_CHUNK_SELECTION_SOURCE="${KG_CHUNK_SELECTION_SOURCE:-truncated}"
QA_PROMPT_STYLE="${QA_PROMPT_STYLE:-semantic_cot}"
ANSWER_PARSE_MODE="${ANSWER_PARSE_MODE:-answer_marker}"

EVAL_RESUME="${EVAL_RESUME:-0}"

DATASETS=("hotpotqa" "musique" "2wiki")
if [[ $# -ge 1 ]]; then
    DATASETS=("$1")
fi

EXPERIMENTS=(
    "agentic_v2_chunk5:5:5"
    "agentic_v2_chunk10:10:10"
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
        return 0
    fi
    if [[ -f "${flat}/multihopqa_index_profile.json" ]]; then
        printf '%s\n' "${flat}"
        return 0
    fi
    die "Missing workspace artifacts for ${dataset}: checked ${nested} and ${flat}"
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

    # This validates the original v0 build profile. Later SYNONYM apply status
    # is validated separately through synonym_linking_manifest.json.
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

payload = json.loads(profile_path.read_text(encoding="utf-8"))
profile = payload.get("index_profile") or {}
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
source_mapping = source_map.get("map") or {}

errors = []
if payload.get("ablation_profile") != "v0":
    errors.append(f"ablation_profile={payload.get('ablation_profile')!r}")
if profile.get("profile_key") != "v0":
    errors.append(f"profile_key={profile.get('profile_key')!r}")
if profile.get("enable_entity_disambiguation") is not False:
    errors.append("entity disambiguation is enabled")
if profile.get("enable_synonym_linking") is not False:
    errors.append("initial synonym linking is enabled")
if int(profile.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(
        f"chunk_token_size={profile.get('chunk_token_size')!r}, expected={expected_chunk_size}"
    )
if manifest.get("workspace_id") != expected_workspace:
    errors.append(f"manifest.workspace_id={manifest.get('workspace_id')!r}")
if manifest.get("dataset") != expected_dataset:
    errors.append(f"manifest.dataset={manifest.get('dataset')!r}")
if manifest.get("ablation_profile") != "v0":
    errors.append(f"manifest.ablation_profile={manifest.get('ablation_profile')!r}")
if int(manifest.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(
        f"manifest.chunk_token_size={manifest.get('chunk_token_size')!r}, expected={expected_chunk_size}"
    )
if source_map.get("workspace_id") != expected_workspace:
    errors.append(f"source_map.workspace_id={source_map.get('workspace_id')!r}")
if source_map.get("dataset") != expected_dataset:
    errors.append(f"source_map.dataset={source_map.get('dataset')!r}")
if int(source_map.get("map_size") or 0) != len(source_mapping):
    errors.append(
        f"source_map.map_size={source_map.get('map_size')!r}, actual={len(source_mapping)}"
    )
if not source_mapping:
    errors.append("source map is empty")

ingest_stats = manifest.get("ingest_stats") or {}
failed = int(ingest_stats.get("failed_now_batch_count") or 0)
if failed:
    errors.append(f"ingest failed_now_batch_count={failed}")

if errors:
    raise SystemExit("Workspace readiness check failed: " + "; ".join(errors))
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
        die "Missing synonym manifest. Checked ${nested} and ${flat}. Apply threshold ${SYNONYM_THRESHOLD} first."
    fi
}

check_synonym_manifest_ready() {
    local working_dir="$1"
    local workspace_id="$2"
    local manifest_path
    manifest_path="$(synonym_manifest_path "${working_dir}" "${workspace_id}")"

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
    errors.append(f"synonymy_threshold={payload.get('synonymy_threshold')!r}, expected={expected_threshold}")
if errors:
    raise SystemExit("Synonym manifest check failed: " + "; ".join(errors))
PY
}

run_agentic_v2_eval() {
    local dataset="$1"
    local working_dir="$2"
    local workspace_id="$3"
    local experiment="$4"
    local chunk_top_k="$5"
    local ppr_qa_top_k="$6"

    local output_dir="${RESULTS_ROOT}/${dataset}/${experiment}"
    local eval_resume_arg=()
    if [[ "${EVAL_RESUME}" == "1" ]]; then
        eval_resume_arg=(--resume)
    fi

    mkdir -p "${output_dir}"
    log "[${dataset}] ${experiment}: chunk_top_k=${chunk_top_k}, ppr_qa_top_k=${ppr_qa_top_k}"
    RAGANYTHING_TEXT_REQUEST_TIMEOUT_SECONDS="${TEXT_REQUEST_TIMEOUT_SECONDS}" \
    python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/evaluate_multihop.py" \
        --dataset "${dataset}" \
        --workspace "${workspace_id}" \
        --working-dir "${working_dir}" \
        --hipporag2-data-dir "${DATA_DIR}" \
        --output-dir "${output_dir}" \
        --modes "agentic_v2" \
        --recall-k ${RECALL_K} \
        --concurrency "${CONCURRENCY}" \
        --top-k "${TOP_K}" \
        --chunk-top-k "${chunk_top_k}" \
        --naive-top-k "${NAIVE_TOP_K}" \
        --max-total-tokens "${MAX_TOTAL_TOKENS}" \
        --qdrant-retrieval-mode "${QDRANT_RETRIEVAL_MODE}" \
        --keyword-fanout-mode "${KEYWORD_FANOUT_MODE}" \
        --kg-chunk-selection-source "${KG_CHUNK_SELECTION_SOURCE}" \
        --no-enable-kg-rerank \
        --no-hybrid-enable-rerank \
        --no-ppr-enable-rerank \
        --ppr-damping "${PPR_DAMPING}" \
        --ppr-top-k "${PPR_TOP_K}" \
        --ppr-qa-top-k "${ppr_qa_top_k}" \
        --passage-node-weight "${PASSAGE_NODE_WEIGHT}" \
        --recognition-top-k "${RECOGNITION_TOP_K}" \
        --linking-top-k "${LINKING_TOP_K}" \
        --ppr-synonym-weight-mode "${PPR_SYNONYM_WEIGHT_MODE}" \
        --answer-context-mode "chunk_only_prompt" \
        --qa-prompt-style "${QA_PROMPT_STYLE}" \
        --answer-parse-mode "${ANSWER_PARSE_MODE}" \
        --bypass-query-cache \
        --no-bypass-keywords-cache \
        "${eval_resume_arg[@]}"
}

check_data_ready
mkdir -p "${RESULTS_ROOT}"

log "================================================================"
log "MultiHopQA agentic v2 retrieval eval"
log "Workspace root:    ${WORKSPACE_ROOT}"
log "Results root:      ${RESULTS_ROOT}"
log "Synonym threshold: ${SYNONYM_THRESHOLD}"
log "Profile:           ${INDEX_PROFILE}"
log "CHUNK_SIZE:        ${CHUNK_SIZE}"
log "Concurrency:       ${CONCURRENCY}"
log "Text timeout:      ${TEXT_REQUEST_TIMEOUT_SECONDS}s"
log "Experiments:       ${EXPERIMENTS[*]}"
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
    check_synonym_manifest_ready "${WORKING_DIR}" "${WORKSPACE_ID}"

    for spec in "${EXPERIMENTS[@]}"; do
        IFS=: read -r EXPERIMENT CHUNK_TOP_K PPR_QA_TOP_K <<<"${spec}"
        run_agentic_v2_eval "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}" "${EXPERIMENT}" "${CHUNK_TOP_K}" "${PPR_QA_TOP_K}"
    done
done

log "Done. Results under: ${RESULTS_ROOT}"
