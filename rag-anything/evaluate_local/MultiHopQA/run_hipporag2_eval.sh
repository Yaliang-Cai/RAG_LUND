#!/usr/bin/env bash
set -euo pipefail

# HippoRAG2-aligned MultiHopQA runner.
# Current index profile is V0: no entity disambiguation, no synonym linking.
#
# The initial V0 build contains no SYNONYM edges. Later experiments may add
# SYNONYM edges directly to this same workspace, following the retrieval v5
# workflow. The runner does not pass exclude_synonym_edges: LightRAG's mode-aware
# default excludes synonym edges for hybrid and allows them for PPR.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"
DATA_DIR="${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/hipporag2_data"

INDEX_PROFILE="v0"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
export CHUNK_SIZE

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/data/workspaces/multihopqa_hr2_v0}"
RESULTS_ROOT="${RESULTS_ROOT:-${RAGANYTHING_ROOT}/results/multihopqa_hr2_v0}"

# PPR is intentionally deferred while PPR ablations are still being selected.
# Re-enable later with: MODES="naive hybrid ppr"
MODES="${MODES:-naive hybrid}"
RECALL_K="${RECALL_K:-2 5}"

CONCURRENCY="${CONCURRENCY:-32}"
TOP_K="${TOP_K:-10}"
CHUNK_TOP_K="${CHUNK_TOP_K:-5}"
NAIVE_TOP_K="${NAIVE_TOP_K:-10}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-45000}"

PPR_DAMPING="${PPR_DAMPING:-0.5}"
PPR_TOP_K="${PPR_TOP_K:-50}"
PPR_QA_TOP_K="${PPR_QA_TOP_K:-5}"
PASSAGE_NODE_WEIGHT="${PASSAGE_NODE_WEIGHT:-0.05}"
LINKING_TOP_K="${LINKING_TOP_K:-5}"
RECOGNITION_TOP_K="${RECOGNITION_TOP_K:-20}"

INGEST_BATCH_SIZE="${INGEST_BATCH_SIZE:-256}"
BATCH_DOC_CONCURRENCY="${BATCH_DOC_CONCURRENCY:-2}"
LLM_MAX_ASYNC="${LLM_MAX_ASYNC:-48}"

BUILD_RESUME="${BUILD_RESUME:-1}"
EVAL_RESUME="${EVAL_RESUME:-0}"
SKIP_INDEX="${SKIP_INDEX:-0}"

DATASETS=("hotpotqa" "musique" "2wiki")
if [[ $# -ge 1 ]]; then
    DATASETS=("$1")
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

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

payload = json.loads(profile_path.read_text(encoding="utf-8"))
profile = payload.get("index_profile") or {}
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
source_mapping = source_map.get("map") or {}

errors = []
if payload.get("ablation_profile") != "v0":
    errors.append(f"ablation_profile={payload.get('ablation_profile')!r}")
if payload.get("ablation_group") != "DB-only":
    errors.append(f"ablation_group={payload.get('ablation_group')!r}")
if profile.get("profile_key") != "v0":
    errors.append(f"profile_key={profile.get('profile_key')!r}")
if profile.get("enable_entity_disambiguation") is not False:
    errors.append(
        f"enable_entity_disambiguation={profile.get('enable_entity_disambiguation')!r}"
    )
if profile.get("enable_synonym_linking") is not False:
    errors.append(f"enable_synonym_linking={profile.get('enable_synonym_linking')!r}")
if int(profile.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(
        f"chunk_token_size={profile.get('chunk_token_size')!r}, expected={expected_chunk_size}"
    )

if manifest.get("workspace_id") != expected_workspace:
    errors.append(f"manifest.workspace_id={manifest.get('workspace_id')!r}")
if manifest.get("dataset") != expected_dataset:
    errors.append(f"manifest.dataset={manifest.get('dataset')!r}")
if manifest.get("corpus_source") != "hipporag2":
    errors.append(f"manifest.corpus_source={manifest.get('corpus_source')!r}")
if manifest.get("n_samples") != 0 or manifest.get("seed") != 0:
    errors.append(
        f"manifest identity n_samples={manifest.get('n_samples')!r}, seed={manifest.get('seed')!r}"
    )
if manifest.get("ablation_profile") != "v0":
    errors.append(f"manifest.ablation_profile={manifest.get('ablation_profile')!r}")
if manifest.get("ablation_group") != "DB-only":
    errors.append(f"manifest.ablation_group={manifest.get('ablation_group')!r}")
if manifest.get("index_profile") != profile:
    errors.append("manifest.index_profile does not match multihopqa_index_profile.json")
if int(manifest.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(
        f"manifest.chunk_token_size={manifest.get('chunk_token_size')!r}, expected={expected_chunk_size}"
    )

if source_map.get("workspace_id") != expected_workspace:
    errors.append(f"source_map.workspace_id={source_map.get('workspace_id')!r}")
if source_map.get("dataset") != expected_dataset:
    errors.append(f"source_map.dataset={source_map.get('dataset')!r}")
if source_map.get("n_samples") != 0 or source_map.get("seed") != 0:
    errors.append(
        f"source_map identity n_samples={source_map.get('n_samples')!r}, seed={source_map.get('seed')!r}"
    )
if int(source_map.get("map_size") or 0) != len(source_mapping):
    errors.append(
        f"source_map.map_size={source_map.get('map_size')!r}, actual={len(source_mapping)}"
    )
if not source_mapping:
    errors.append("source_map is empty")
if int(manifest.get("expected_chunk_total") or 0) != len(source_mapping):
    errors.append(
        f"manifest.expected_chunk_total={manifest.get('expected_chunk_total')!r}, "
        f"source_map_size={len(source_mapping)}"
    )

ingest_stats = manifest.get("ingest_stats") or {}
batch_count = int(manifest.get("batch_count") or 0)
successful = int(ingest_stats.get("successful_before_batch_count") or 0) + int(
    ingest_stats.get("successful_now_batch_count") or 0
)
failed = int(ingest_stats.get("failed_now_batch_count") or 0)
if failed:
    errors.append(f"ingest failed_now_batch_count={failed}")
if batch_count and successful != batch_count:
    errors.append(f"successful batches={successful}, batch_count={batch_count}")

if errors:
    raise SystemExit(
        "Workspace readiness check failed. Use a clean V0/CHUNK_SIZE=4096 workspace. "
        + "; ".join(errors)
    )
PY
}

log "=== Step 0: Download HippoRAG2 datasets ==="
python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/download_hipporag2_datasets.py" \
    --output-dir "${DATA_DIR}"

for DATASET in "${DATASETS[@]}"; do
    WORKSPACE_ID="${DATASET}_hr2_v0"
    WORKING_DIR="${WORKSPACE_ROOT}/${DATASET}"
    RESULTS_DIR="${RESULTS_ROOT}/${DATASET}"

    log "================================================================"
    log "Dataset: ${DATASET}"
    log "Workspace ID: ${WORKSPACE_ID}"
    log "Working dir:  ${WORKING_DIR}"
    log "Results:      ${RESULTS_DIR}"
    log "Profile:      ${INDEX_PROFILE}"
    log "CHUNK_SIZE:   ${CHUNK_SIZE}"
    log "Modes:        ${MODES}"
    log "================================================================"

    mkdir -p "${WORKING_DIR}" "${RESULTS_DIR}"

    if [[ "${SKIP_INDEX}" == "1" ]]; then
        log "[${DATASET}] SKIP_INDEX=1; skipping build_index"
    else
        build_resume_arg=()
        if [[ "${BUILD_RESUME}" == "1" ]]; then
            build_resume_arg=(--resume)
        fi

        log "[${DATASET}] Step 1: Building V0 index"
        python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/build_index.py" \
            --dataset "${DATASET}" \
            --workspace "${WORKSPACE_ID}" \
            --working-dir "${WORKING_DIR}" \
            --hipporag2-data-dir "${DATA_DIR}" \
            --index-profile "${INDEX_PROFILE}" \
            --ingest-batch-size "${INGEST_BATCH_SIZE}" \
            --batch-doc-concurrency "${BATCH_DOC_CONCURRENCY}" \
            --llm-model-max-async "${LLM_MAX_ASYNC}" \
            "${build_resume_arg[@]}"
        log "[${DATASET}] Index build complete."
    fi

    check_workspace_ready "${WORKING_DIR}" "${DATASET}" "${WORKSPACE_ID}"

    eval_resume_arg=()
    if [[ "${EVAL_RESUME}" == "1" ]]; then
        eval_resume_arg=(--resume)
    fi

    log "[${DATASET}] Step 2: Evaluating modes: ${MODES}"
    python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/evaluate_multihop.py" \
        --dataset "${DATASET}" \
        --workspace "${WORKSPACE_ID}" \
        --working-dir "${WORKING_DIR}" \
        --hipporag2-data-dir "${DATA_DIR}" \
        --output-dir "${RESULTS_DIR}" \
        --modes ${MODES} \
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
        --linking-top-k "${LINKING_TOP_K}" \
        --recognition-top-k "${RECOGNITION_TOP_K}" \
        --no-enable-kg-rerank \
        --no-ppr-enable-rerank \
        --bypass-query-cache \
        "${eval_resume_arg[@]}"
    log "[${DATASET}] Evaluation complete: ${RESULTS_DIR}/${DATASET}_summary.json"
done

log "================================================================"
log "Results summary"
log "================================================================"
python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/print_results_table.py" \
    --results-root "${RESULTS_ROOT}" \
    --datasets "${DATASETS[@]}" \
    --modes ${MODES} \
    --recall-k ${RECALL_K}

log "Done."
