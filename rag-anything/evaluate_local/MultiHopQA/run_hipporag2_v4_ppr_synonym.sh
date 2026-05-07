#!/usr/bin/env bash
set -euo pipefail

# MultiHopQA V4 ppr_default_with_synonym runner.
#
# This runner reuses existing V0 MultiHopQA workspaces and existing threshold
# 0.8 SYNONYM edges. It does not download data, build indexes, rebuild SYNONYM
# edges, or run the no-synonym V4 component experiments.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"
DATA_DIR="${DATA_DIR:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/hipporag2_data}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0}"
RESULTS_ROOT="${RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_v4_components}"
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
EXPERIMENT="ppr_default_with_synonym"

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

check_synonym_manifest_ready() {
    local working_dir="$1"
    local workspace_id="$2"
    local manifest_path="${working_dir}/synonym_linking_manifest.json"
    [[ -f "${manifest_path}" ]] || die "Missing synonym manifest: ${manifest_path}. Run run_hipporag2_synonym_ablation.sh first."

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
    raise SystemExit(
        "Synonym manifest check failed. Expected completed threshold-0.8 edges: "
        + "; ".join(errors)
    )
PY
}

run_with_synonym() {
    local dataset="$1"
    local working_dir="$2"
    local workspace_id="$3"
    local output_dir="${RESULTS_ROOT}/${dataset}/${EXPERIMENT}"
    local eval_resume_arg=()
    if [[ "${EVAL_RESUME}" == "1" ]]; then
        eval_resume_arg=(--resume)
    fi

    mkdir -p "${output_dir}"
    log "[${dataset}] ${EXPERIMENT}: ppr default with SYNONYM edges"
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
        --qdrant-retrieval-mode "dense" \
        --answer-context-mode "chunk_only_prompt" \
        --bypass-query-cache \
        "${eval_resume_arg[@]}"
}

print_summary() {
    python - "$RESULTS_ROOT" "$EXPERIMENT" "${DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
experiment = sys.argv[2]
datasets = sys.argv[3:]

print("dataset\texperiment\tmode\tem\tf1\trecall@2\trecall@5")
for dataset in datasets:
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
log "MultiHopQA V4 PPR with SYNONYM edges"
log "Workspace root:     ${WORKSPACE_ROOT}"
log "Results root:       ${RESULTS_ROOT}"
log "Synonym threshold:  ${SYNONYM_THRESHOLD}"
log "Profile:            ${INDEX_PROFILE}"
log "CHUNK_SIZE:         ${CHUNK_SIZE}"
log "Concurrency:        ${CONCURRENCY}"
log "Experiment:         ${EXPERIMENT}"
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
    check_synonym_manifest_ready "${WORKING_DIR}" "${WORKSPACE_ID}"
    run_with_synonym "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}"
done

log "================================================================"
log "Results summary"
log "================================================================"
print_summary
log "Done."
