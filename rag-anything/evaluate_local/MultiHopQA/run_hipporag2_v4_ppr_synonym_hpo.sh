#!/usr/bin/env bash
set -euo pipefail

# MultiHopQA PPR HPO with hybrid retrieval and prebuilt synonym edges.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"

SYNONYM_THRESHOLD="${SYNONYM_THRESHOLD:-0.8}"
HPO_STAGE="${HPO_STAGE:-dev}"
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

case "${HPO_STAGE}" in
    dev)
        RESULTS_ROOT="${RESULTS_ROOT:-${DEV_RESULTS_ROOT}}"
        N_SAMPLES="${N_SAMPLES:-${DEV_N_SAMPLES}}"
        RUN_SEED="${RUN_SEED:-${SEED}}"
        TOP_N="${TOP_N:-${VERIFY_TOP_N}}"
        ;;
    verify)
        RESULTS_ROOT="${RESULTS_ROOT:-${VERIFY_RESULTS_ROOT}}"
        N_SAMPLES="${N_SAMPLES:-${VERIFY_N_SAMPLES}}"
        RUN_SEED="${RUN_SEED:-${VERIFY_SEED}}"
        TOP_N="${TOP_N:-${VERIFY_TOP_N}}"
        ;;
    full)
        RESULTS_ROOT="${RESULTS_ROOT:-${FULL_RESULTS_ROOT}}"
        N_SAMPLES="${N_SAMPLES:-${FULL_N_SAMPLES}}"
        RUN_SEED="${RUN_SEED:-${SEED}}"
        TOP_N="${TOP_N:-${FULL_TOP_N}}"
        ;;
    *)
        echo "ERROR: HPO_STAGE must be dev, verify, or full" >&2
        exit 1
        ;;
esac

echo "================================================================"
echo "MultiHopQA PPR HPO: hybrid retrieval + synonym edges"
echo "Stage:              ${HPO_STAGE}"
echo "Synonym threshold:  ${SYNONYM_THRESHOLD}"
echo "Results root:       ${RESULTS_ROOT}"
echo "Study DB:           ${STUDY_DB}"
echo "N samples:          ${N_SAMPLES}"
echo "Seed:               ${RUN_SEED}"
echo "Concurrency:        ${CONCURRENCY}"
echo "N trials:           ${N_TRIALS}"
echo "Top N:              ${TOP_N}"
echo "================================================================"

CONFIG_ARGS=()
if [[ "${HPO_STAGE}" == "full" && -z "${CONFIGS_FILE:-}" && -f "${VERIFY_TOP_CONFIGS}" ]]; then
    CONFIGS_FILE="${VERIFY_TOP_CONFIGS}"
fi
if [[ -n "${CONFIGS_FILE:-}" ]]; then
    CONFIG_ARGS=(--configs-file "${CONFIGS_FILE}")
fi

python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/optuna_ppr_synonym_hpo.py" \
    --stage "${HPO_STAGE}" \
    --repo-root "${RAGANYTHING_ROOT}" \
    --data-dir "${DATA_DIR}" \
    --workspace-root "${WORKSPACE_ROOT}" \
    --results-root "${RESULTS_ROOT}" \
    --study-db "${STUDY_DB}" \
    --synonym-threshold "${SYNONYM_THRESHOLD}" \
    --n-samples "${N_SAMPLES}" \
    --seed "${RUN_SEED}" \
    --n-trials "${N_TRIALS}" \
    --top-n "${TOP_N}" \
    --concurrency "${CONCURRENCY}" \
    --optuna-jobs "${OPTUNA_JOBS}" \
    --pruner "${PRUNER}" \
    "${CONFIG_ARGS[@]}" \
    "$@"
