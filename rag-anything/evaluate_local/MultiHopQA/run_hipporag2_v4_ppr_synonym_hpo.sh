#!/usr/bin/env bash
set -euo pipefail

# MultiHopQA PPR HPO with hybrid retrieval and prebuilt synonym edges.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"

SYNONYM_THRESHOLD="${SYNONYM_THRESHOLD:-0.8}"
HPO_STAGE="${HPO_STAGE:-dev}"
APPLY_SYNONYM_EDGES="${APPLY_SYNONYM_EDGES:-0}"
N_TRIALS="${N_TRIALS:-40}"
CONCURRENCY="${CONCURRENCY:-100}"
OPTUNA_JOBS="${OPTUNA_JOBS:-1}"
PRUNER="${PRUNER:-none}"
VERIFY_TOP_N="${VERIFY_TOP_N:-5}"
FULL_TOP_N="${FULL_TOP_N:-3}"

DEV_N_SAMPLES="${DEV_N_SAMPLES:-200}"
VERIFY_N_SAMPLES="${VERIFY_N_SAMPLES:-300}"
FULL_N_SAMPLES="${FULL_N_SAMPLES:-1000}"
SEED="${SEED:-42}"
VERIFY_SEED="${VERIFY_SEED:-43}"

DATA_DIR="${DATA_DIR:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/hipporag2_data}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0}"
DATASETS="${DATASETS:-hotpotqa musique 2wiki}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"

threshold_label="$(
python - "$SYNONYM_THRESHOLD" <<'PY'
import sys
print(f"{float(sys.argv[1]):.12g}".replace("-", "m").replace(".", "p"))
PY
)"

DEV_RESULTS_ROOT="${DEV_RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_ppr_hpo_semantic_prompt_syn_t${threshold_label}_dev}"
VERIFY_RESULTS_ROOT="${VERIFY_RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_ppr_hpo_semantic_prompt_syn_t${threshold_label}_verify}"
FULL_RESULTS_ROOT="${FULL_RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_ppr_hpo_semantic_prompt_syn_t${threshold_label}_full}"
VERIFY_TOP_CONFIGS="${VERIFY_RESULTS_ROOT}/top_configs.tsv"
STUDY_DB="${STUDY_DB:-${DEV_RESULTS_ROOT}/study.db}"
SYNONYM_OPS_DIR="${SYNONYM_OPS_DIR:-${DEV_RESULTS_ROOT}/_synonym_ops}"

read -r -a DATASET_ARRAY <<< "${DATASETS}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

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

check_synonym_manifest_ready() {
    local working_dir="$1"
    local workspace_id="$2"
    local threshold="$3"
    local nested="${working_dir}/${workspace_id}/synonym_linking_manifest.json"
    local flat="${working_dir}/synonym_linking_manifest.json"
    local manifest_path=""

    if [[ -f "${nested}" ]]; then
        manifest_path="${nested}"
    elif [[ -f "${flat}" ]]; then
        manifest_path="${flat}"
    else
        die "Missing synonym manifest. Checked ${nested} and ${flat}."
    fi

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
    local workspace_id="${dataset}_hr2_v0"
    local working_dir=""
    local log_file="${SYNONYM_OPS_DIR}/${dataset}_syn${threshold_label}.log"

    working_dir="$(resolve_working_dir "${dataset}" "${workspace_id}")"
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

apply_synonym_edges_for_all() {
    log "Applying SYNONYM edges for datasets: ${DATASETS}"
    for dataset in "${DATASET_ARRAY[@]}"; do
        apply_synonym_edges "${dataset}"
    done
}

stage_defaults() {
    local stage="$1"
    case "${stage}" in
        dev)
            STAGE_RESULTS_ROOT="${DEV_RESULTS_ROOT}"
            STAGE_N_SAMPLES="${DEV_N_SAMPLES}"
            STAGE_SEED="${SEED}"
            STAGE_TOP_N="${VERIFY_TOP_N}"
            ;;
        verify)
            STAGE_RESULTS_ROOT="${VERIFY_RESULTS_ROOT}"
            STAGE_N_SAMPLES="${VERIFY_N_SAMPLES}"
            STAGE_SEED="${VERIFY_SEED}"
            STAGE_TOP_N="${VERIFY_TOP_N}"
            ;;
        full)
            STAGE_RESULTS_ROOT="${FULL_RESULTS_ROOT}"
            STAGE_N_SAMPLES="${FULL_N_SAMPLES}"
            STAGE_SEED="${SEED}"
            STAGE_TOP_N="${FULL_TOP_N}"
            ;;
        *)
            die "HPO_STAGE must be dev, verify, full, or all"
            ;;
    esac
}

run_stage() {
    local stage="$1"
    shift
    local results_root=""
    local n_samples=""
    local run_seed=""
    local top_n=""
    local config_args=()
    local dataset_args=()

    stage_defaults "${stage}"
    results_root="${STAGE_RESULTS_ROOT}"
    n_samples="${STAGE_N_SAMPLES}"
    run_seed="${STAGE_SEED}"
    top_n="${STAGE_TOP_N}"

    if [[ "${HPO_STAGE}" != "all" ]]; then
        results_root="${RESULTS_ROOT:-${results_root}}"
        n_samples="${N_SAMPLES:-${n_samples}}"
        run_seed="${RUN_SEED:-${run_seed}}"
        top_n="${TOP_N:-${top_n}}"
    fi

    if [[ "${stage}" == "full" && -z "${CONFIGS_FILE:-}" && -f "${VERIFY_TOP_CONFIGS}" ]]; then
        config_args=(--configs-file "${VERIFY_TOP_CONFIGS}")
    elif [[ -n "${CONFIGS_FILE:-}" ]]; then
        config_args=(--configs-file "${CONFIGS_FILE}")
    fi

    dataset_args=(--datasets "${DATASET_ARRAY[@]}")

    echo "================================================================"
    echo "MultiHopQA PPR HPO: hybrid retrieval + synonym edges"
    echo "Stage:              ${stage}"
    echo "Synonym threshold:  ${SYNONYM_THRESHOLD}"
    echo "Results root:       ${results_root}"
    echo "Study DB:           ${STUDY_DB}"
    echo "N samples:          ${n_samples}"
    echo "Seed:               ${run_seed}"
    echo "Datasets:           ${DATASETS}"
    echo "Concurrency:        ${CONCURRENCY}"
    echo "N trials:           ${N_TRIALS}"
    echo "Top N:              ${top_n}"
    echo "================================================================"

    python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/optuna_ppr_synonym_hpo.py" \
        --stage "${stage}" \
        --repo-root "${RAGANYTHING_ROOT}" \
        --data-dir "${DATA_DIR}" \
        --workspace-root "${WORKSPACE_ROOT}" \
        --results-root "${results_root}" \
        --study-db "${STUDY_DB}" \
        --synonym-threshold "${SYNONYM_THRESHOLD}" \
        --n-samples "${n_samples}" \
        --seed "${run_seed}" \
        --n-trials "${N_TRIALS}" \
        --top-n "${top_n}" \
        --concurrency "${CONCURRENCY}" \
        --optuna-jobs "${OPTUNA_JOBS}" \
        --pruner "${PRUNER}" \
        "${dataset_args[@]}" \
        "${config_args[@]}" \
        "$@"
}

if [[ "${HPO_STAGE}" == "all" ]]; then
    apply_synonym_edges_for_all
    run_stage "dev" "$@"
    run_stage "verify" "$@"
    run_stage "full" "$@"
elif [[ "${HPO_STAGE}" == "dev" || "${HPO_STAGE}" == "verify" || "${HPO_STAGE}" == "full" ]]; then
    if [[ "${APPLY_SYNONYM_EDGES}" == "1" ]]; then
        apply_synonym_edges_for_all
    fi
    run_stage "${HPO_STAGE}" "$@"
else
    die "HPO_STAGE must be dev, verify, full, or all"
fi
