#!/usr/bin/env bash
set -euo pipefail

# MultiHopQA PPR hyperparameter runner with semantic passage CoT QA prompt.
#
# This runner reuses existing V0 HippoRAG2 MultiHopQA workspaces. It does not
# download data, build indexes, rebuild SYNONYM edges, or enable synonym edges.
# Default run is Dataiku-style Stage 1 one-factor screening on a fixed dev
# query subset. Stage 2/3 candidates can be supplied through CONFIG_FILE.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"
DATA_DIR="${DATA_DIR:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/hipporag2_data}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/workspaces/multihopqa_hr2_v0}"
DEV_RESULTS_ROOT="${DEV_RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_ppr_hpo_semantic_prompt_dev}"
FULL_RESULTS_ROOT="${FULL_RESULTS_ROOT:-${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/results/multihopqa_hr2_v0_ppr_hpo_semantic_prompt_full}"

HPO_STAGE="${HPO_STAGE:-dev}"
CONFIG_SET="${CONFIG_SET:-stage1}"
CONFIG_FILE="${CONFIG_FILE:-}"
SEED="${SEED:-42}"
DEV_N_SAMPLES="${DEV_N_SAMPLES:-200}"
FULL_N_SAMPLES="${FULL_N_SAMPLES:-1000}"

CHUNK_SIZE="${CHUNK_SIZE:-4096}"
export CHUNK_SIZE

RECALL_K="${RECALL_K:-2 5}"
CONCURRENCY="${CONCURRENCY:-100}"
NAIVE_TOP_K="${NAIVE_TOP_K:-10}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-45000}"
MIN_RERANK_SCORE="${MIN_RERANK_SCORE:-0.3}"

QDRANT_RETRIEVAL_MODE="${QDRANT_RETRIEVAL_MODE:-hybrid}"
KEYWORD_FANOUT_MODE="${KEYWORD_FANOUT_MODE:-joined}"
BASE_TOP_K="${BASE_TOP_K:-10}"
BASE_PPR_QA_TOP_K="${BASE_PPR_QA_TOP_K:-5}"
BASE_PPR_TOP_K="${BASE_PPR_TOP_K:-50}"
BASE_PASSAGE_NODE_WEIGHT="${BASE_PASSAGE_NODE_WEIGHT:-0.05}"
BASE_PPR_DAMPING="${BASE_PPR_DAMPING:-0.5}"
BASE_HUB_PENALTY_THRESHOLD="${BASE_HUB_PENALTY_THRESHOLD:-50}"

RECOGNITION_TOP_K="${RECOGNITION_TOP_K:-20}"
LINKING_TOP_K="${LINKING_TOP_K:-5}"
PPR_POST_RERANK_RRF_K="${PPR_POST_RERANK_RRF_K:-60}"
PPR_SYNONYM_WEIGHT_MODE="${PPR_SYNONYM_WEIGHT_MODE:-raw}"
EVAL_RESUME="${EVAL_RESUME:-0}"

TOP_K_VALUES=(5 10 20 40)
PPR_QA_TOP_K_VALUES=(3 5 8 10)
PPR_TOP_K_VALUES=(25 50 100)
PASSAGE_NODE_WEIGHT_VALUES=(0 0.02 0.05 0.1 0.2)
PPR_DAMPING_VALUES=(0.35 0.5 0.65 0.8)
HUB_PENALTY_THRESHOLD_VALUES=(0 1 10 50)

DATASETS=("hotpotqa" "musique" "2wiki")
if [[ $# -ge 1 ]]; then
    DATASETS=("$1")
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

case "${HPO_STAGE}" in
    dev)
        RESULTS_ROOT="${RESULTS_ROOT:-${DEV_RESULTS_ROOT}}"
        N_SAMPLES="${N_SAMPLES:-${DEV_N_SAMPLES}}"
        ;;
    full)
        RESULTS_ROOT="${RESULTS_ROOT:-${FULL_RESULTS_ROOT}}"
        N_SAMPLES="${N_SAMPLES:-${FULL_N_SAMPLES}}"
        ;;
    *)
        die "Unknown HPO_STAGE=${HPO_STAGE}; expected dev or full"
        ;;
esac

if [[ "${QDRANT_RETRIEVAL_MODE}" != "hybrid" ]]; then
    die "This HPO runner is scoped to qdrant_retrieval_mode=hybrid"
fi
if [[ "${KEYWORD_FANOUT_MODE}" != "joined" ]]; then
    die "This HPO runner is scoped to keyword_fanout_mode=joined"
fi

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
manifest_index_profile = manifest.get("index_profile") or {}
ingest_stats = manifest.get("ingest_stats") or {}
source_payload = source_map.get("map") or {}
errors = []

def require_int(payload, key, label):
    if key not in payload or payload.get(key) is None:
        errors.append(f"{label}.{key}=missing")
        return None
    try:
        return int(payload.get(key))
    except (TypeError, ValueError):
        errors.append(f"{label}.{key}={payload.get(key)!r}")
        return None

if profile.get("schema_version") != "multihopqa_index_profile_v1":
    errors.append(f"profile.schema_version={profile.get('schema_version')!r}")
if profile.get("workspace_id") != expected_workspace:
    errors.append(f"profile.workspace_id={profile.get('workspace_id')!r}")
if profile.get("dataset") != expected_dataset:
    errors.append(f"profile.dataset={profile.get('dataset')!r}")
if profile.get("ablation_profile") != "v0":
    errors.append(f"profile.ablation_profile={profile.get('ablation_profile')!r}")
profile_n_samples = require_int(profile, "n_samples", "profile")
profile_seed = require_int(profile, "seed", "profile")
if profile_n_samples is not None and profile_n_samples != 0:
    errors.append(f"profile.n_samples={profile.get('n_samples')!r}")
if profile_seed is not None and profile_seed != 0:
    errors.append(f"profile.seed={profile.get('seed')!r}")
if index_profile.get("profile_key") != "v0":
    errors.append(f"index_profile.profile_key={index_profile.get('profile_key')!r}")
if int(index_profile.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(
        f"index_profile.chunk_token_size={index_profile.get('chunk_token_size')!r}, "
        f"expected={expected_chunk_size}"
    )
if index_profile.get("enable_synonym_linking") is not False:
    errors.append(
        f"index_profile.enable_synonym_linking={index_profile.get('enable_synonym_linking')!r}"
    )

if manifest.get("schema_version") != "multihopqa_ingest_manifest_v1":
    errors.append(f"manifest.schema_version={manifest.get('schema_version')!r}")
if manifest.get("workspace_id") != expected_workspace:
    errors.append(f"manifest.workspace_id={manifest.get('workspace_id')!r}")
if manifest.get("dataset") != expected_dataset:
    errors.append(f"manifest.dataset={manifest.get('dataset')!r}")
if manifest.get("corpus_source") != "hipporag2":
    errors.append(f"manifest.corpus_source={manifest.get('corpus_source')!r}")
manifest_n_samples = require_int(manifest, "n_samples", "manifest")
manifest_seed = require_int(manifest, "seed", "manifest")
if manifest_n_samples is not None and manifest_n_samples != 0:
    errors.append(f"manifest.n_samples={manifest.get('n_samples')!r}")
if manifest_seed is not None and manifest_seed != 0:
    errors.append(f"manifest.seed={manifest.get('seed')!r}")
if int(manifest.get("chunk_token_size") or 0) != expected_chunk_size:
    errors.append(
        f"manifest.chunk_token_size={manifest.get('chunk_token_size')!r}, "
        f"expected={expected_chunk_size}"
    )
if manifest_index_profile != index_profile:
    errors.append("manifest.index_profile does not match profile.index_profile")
if int(ingest_stats.get("failed_now_batch_count") or 0) != 0:
    errors.append(
        f"ingest_stats.failed_now_batch_count={ingest_stats.get('failed_now_batch_count')!r}"
    )
batch_count = int(manifest.get("batch_count") or 0)
successful_count = int(ingest_stats.get("successful_before_batch_count") or 0) + int(
    ingest_stats.get("successful_now_batch_count") or 0
)
if batch_count > 0 and successful_count != batch_count:
    errors.append(f"successful_batch_count={successful_count}, batch_count={batch_count}")

if source_map.get("workspace_id") != expected_workspace:
    errors.append(f"source_map.workspace_id={source_map.get('workspace_id')!r}")
if source_map.get("dataset") != expected_dataset:
    errors.append(f"source_map.dataset={source_map.get('dataset')!r}")
source_map_n_samples = require_int(source_map, "n_samples", "source_map")
source_map_seed = require_int(source_map, "seed", "source_map")
if source_map_n_samples is not None and source_map_n_samples != 0:
    errors.append(f"source_map.n_samples={source_map.get('n_samples')!r}")
if source_map_seed is not None and source_map_seed != 0:
    errors.append(f"source_map.seed={source_map.get('seed')!r}")
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
        f"manifest.expected_chunk_total={expected_chunk_total}, source_map.map_size={source_map_size}"
    )

if errors:
    raise SystemExit("Workspace check failed: " + "; ".join(errors))
PY
}

CONFIG_NAMES=()
CONFIG_TOP_K=()
CONFIG_PPR_QA_TOP_K=()
CONFIG_PPR_TOP_K=()
CONFIG_PASSAGE_NODE_WEIGHT=()
CONFIG_PPR_DAMPING=()
CONFIG_HUB_PENALTY_THRESHOLD=()

add_config() {
    local name="$1"
    local top_k="$2"
    local ppr_qa_top_k="$3"
    local ppr_top_k="$4"
    local passage_node_weight="$5"
    local ppr_damping="$6"
    local hub_penalty_threshold="$7"

    CONFIG_NAMES+=("${name}")
    CONFIG_TOP_K+=("${top_k}")
    CONFIG_PPR_QA_TOP_K+=("${ppr_qa_top_k}")
    CONFIG_PPR_TOP_K+=("${ppr_top_k}")
    CONFIG_PASSAGE_NODE_WEIGHT+=("${passage_node_weight}")
    CONFIG_PPR_DAMPING+=("${ppr_damping}")
    CONFIG_HUB_PENALTY_THRESHOLD+=("${hub_penalty_threshold}")
}

value_label() {
    local value="$1"
    value="${value//./p}"
    printf '%s\n' "${value}"
}

add_anchor_config() {
    add_config "ppr_hybrid_anchor" \
        "${BASE_TOP_K}" \
        "${BASE_PPR_QA_TOP_K}" \
        "${BASE_PPR_TOP_K}" \
        "${BASE_PASSAGE_NODE_WEIGHT}" \
        "${BASE_PPR_DAMPING}" \
        "${BASE_HUB_PENALTY_THRESHOLD}"
}

add_stage1_configs() {
    local value=""
    add_anchor_config

    for value in "${TOP_K_VALUES[@]}"; do
        [[ "${value}" == "${BASE_TOP_K}" ]] && continue
        add_config "top_k_${value}" \
            "${value}" "${BASE_PPR_QA_TOP_K}" "${BASE_PPR_TOP_K}" \
            "${BASE_PASSAGE_NODE_WEIGHT}" "${BASE_PPR_DAMPING}" "${BASE_HUB_PENALTY_THRESHOLD}"
    done
    for value in "${PPR_QA_TOP_K_VALUES[@]}"; do
        [[ "${value}" == "${BASE_PPR_QA_TOP_K}" ]] && continue
        add_config "ppr_qa_top_k_${value}" \
            "${BASE_TOP_K}" "${value}" "${BASE_PPR_TOP_K}" \
            "${BASE_PASSAGE_NODE_WEIGHT}" "${BASE_PPR_DAMPING}" "${BASE_HUB_PENALTY_THRESHOLD}"
    done
    for value in "${PPR_TOP_K_VALUES[@]}"; do
        [[ "${value}" == "${BASE_PPR_TOP_K}" ]] && continue
        add_config "ppr_top_k_${value}" \
            "${BASE_TOP_K}" "${BASE_PPR_QA_TOP_K}" "${value}" \
            "${BASE_PASSAGE_NODE_WEIGHT}" "${BASE_PPR_DAMPING}" "${BASE_HUB_PENALTY_THRESHOLD}"
    done
    for value in "${PASSAGE_NODE_WEIGHT_VALUES[@]}"; do
        [[ "${value}" == "${BASE_PASSAGE_NODE_WEIGHT}" ]] && continue
        add_config "passage_node_weight_$(value_label "${value}")" \
            "${BASE_TOP_K}" "${BASE_PPR_QA_TOP_K}" "${BASE_PPR_TOP_K}" \
            "${value}" "${BASE_PPR_DAMPING}" "${BASE_HUB_PENALTY_THRESHOLD}"
    done
    for value in "${PPR_DAMPING_VALUES[@]}"; do
        [[ "${value}" == "${BASE_PPR_DAMPING}" ]] && continue
        add_config "ppr_damping_$(value_label "${value}")" \
            "${BASE_TOP_K}" "${BASE_PPR_QA_TOP_K}" "${BASE_PPR_TOP_K}" \
            "${BASE_PASSAGE_NODE_WEIGHT}" "${value}" "${BASE_HUB_PENALTY_THRESHOLD}"
    done
    for value in "${HUB_PENALTY_THRESHOLD_VALUES[@]}"; do
        [[ "${value}" == "${BASE_HUB_PENALTY_THRESHOLD}" ]] && continue
        add_config "hub_penalty_threshold_${value}" \
            "${BASE_TOP_K}" "${BASE_PPR_QA_TOP_K}" "${BASE_PPR_TOP_K}" \
            "${BASE_PASSAGE_NODE_WEIGHT}" "${BASE_PPR_DAMPING}" "${value}"
    done
}

add_custom_configs() {
    [[ -n "${CONFIG_FILE}" ]] || die "CONFIG_SET=custom requires CONFIG_FILE"
    [[ -f "${CONFIG_FILE}" ]] || die "Missing CONFIG_FILE: ${CONFIG_FILE}"

    add_anchor_config
    while IFS=$'\t ' read -r name top_k ppr_qa_top_k ppr_top_k passage_node_weight ppr_damping hub_penalty_threshold extra; do
        [[ -z "${name:-}" ]] && continue
        [[ "${name:0:1}" == "#" ]] && continue
        [[ -z "${extra:-}" ]] || die "Too many columns in CONFIG_FILE row for ${name}"
        add_config "${name}" "${top_k}" "${ppr_qa_top_k}" "${ppr_top_k}" \
            "${passage_node_weight}" "${ppr_damping}" "${hub_penalty_threshold}"
    done < "${CONFIG_FILE}"
}

build_configs() {
    case "${CONFIG_SET}" in
        stage1)
            if [[ "${HPO_STAGE}" != "dev" ]]; then
                die "CONFIG_SET=stage1 is intended for HPO_STAGE=dev"
            fi
            add_stage1_configs
            ;;
        anchor)
            add_anchor_config
            ;;
        custom)
            add_custom_configs
            ;;
        *)
            die "Unknown CONFIG_SET=${CONFIG_SET}; expected stage1, anchor, or custom"
            ;;
    esac
}

run_ppr_config() {
    local idx="$1"
    local dataset="$2"
    local working_dir="$3"
    local workspace_id="$4"
    local name="${CONFIG_NAMES[$idx]}"
    local top_k="${CONFIG_TOP_K[$idx]}"
    local ppr_qa_top_k="${CONFIG_PPR_QA_TOP_K[$idx]}"
    local ppr_top_k="${CONFIG_PPR_TOP_K[$idx]}"
    local passage_node_weight="${CONFIG_PASSAGE_NODE_WEIGHT[$idx]}"
    local ppr_damping="${CONFIG_PPR_DAMPING[$idx]}"
    local hub_penalty_threshold="${CONFIG_HUB_PENALTY_THRESHOLD[$idx]}"
    local output_dir="${RESULTS_ROOT}/${dataset}/${name}"
    local eval_resume_arg=()

    if [[ "${EVAL_RESUME}" == "1" ]]; then
        eval_resume_arg=(--resume)
    fi

    mkdir -p "${output_dir}"
    log "[${dataset}] ${name}: top_k=${top_k}, ppr_qa_top_k=${ppr_qa_top_k}, ppr_top_k=${ppr_top_k}, passage_node_weight=${passage_node_weight}, ppr_damping=${ppr_damping}, hub_penalty_threshold=${hub_penalty_threshold}"
    RAGANYTHING_MIN_RERANK_SCORE="${MIN_RERANK_SCORE}" \
    MIN_RERANK_SCORE="${MIN_RERANK_SCORE}" \
    python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/evaluate_multihop.py" \
        --dataset "${dataset}" \
        --workspace "${workspace_id}" \
        --working-dir "${working_dir}" \
        --hipporag2-data-dir "${DATA_DIR}" \
        --n-samples "${N_SAMPLES}" \
        --seed "${SEED}" \
        --output-dir "${output_dir}" \
        --modes "ppr" \
        --recall-k ${RECALL_K} \
        --concurrency "${CONCURRENCY}" \
        --top-k "${top_k}" \
        --chunk-top-k "${ppr_qa_top_k}" \
        --naive-top-k "${NAIVE_TOP_K}" \
        --max-total-tokens "${MAX_TOTAL_TOKENS}" \
        --ppr-damping "${ppr_damping}" \
        --ppr-top-k "${ppr_top_k}" \
        --ppr-qa-top-k "${ppr_qa_top_k}" \
        --hub-penalty-threshold "${hub_penalty_threshold}" \
        --passage-node-weight "${passage_node_weight}" \
        --recognition-top-k "${RECOGNITION_TOP_K}" \
        --linking-top-k "${LINKING_TOP_K}" \
        --ppr-post-rerank-fusion "none" \
        --ppr-post-rerank-rrf-k "${PPR_POST_RERANK_RRF_K}" \
        --ppr-synonym-weight-mode "${PPR_SYNONYM_WEIGHT_MODE}" \
        --no-enable-kg-rerank \
        --no-ppr-enable-rerank \
        --exclude-synonym-edges \
        --keyword-fanout-mode "${KEYWORD_FANOUT_MODE}" \
        --qdrant-retrieval-mode "${QDRANT_RETRIEVAL_MODE}" \
        --answer-context-mode "chunk_only_prompt" \
        --qa-prompt-style "semantic_cot" \
        --answer-parse-mode "answer_marker" \
        --bypass-query-cache \
        --no-bypass-keywords-cache \
        "${eval_resume_arg[@]}"
}

print_summary() {
    python - "$RESULTS_ROOT" "${CONFIG_NAMES[@]}" -- "${DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

sep = sys.argv.index("--")
root = Path(sys.argv[1])
configs = sys.argv[2:sep]
datasets = sys.argv[sep + 1:]

print("dataset\tconfig\tmode\tem\tf1\trecall@2\trecall@5")
for dataset in datasets:
    for config in configs:
        path = root / dataset / config / f"{dataset}_summary.json"
        if not path.exists():
            print(f"{dataset}\t{config}\tMISSING\t\t\t\t")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results") or {}
        if not results:
            print(f"{dataset}\t{config}\tNO_MODES\t\t\t\t")
            continue
        for mode, metrics in results.items():
            print(
                f"{dataset}\t{config}\t{mode}\t"
                f"{metrics.get('em', '')}\t{metrics.get('f1', '')}\t"
                f"{metrics.get('recall@2', '')}\t{metrics.get('recall@5', '')}"
            )
PY
}

check_data_ready
build_configs
mkdir -p "${RESULTS_ROOT}"

log "================================================================"
log "MultiHopQA PPR HPO with semantic passage CoT QA prompt"
log "Workspace root:       ${WORKSPACE_ROOT}"
log "Results root:         ${RESULTS_ROOT}"
log "HPO stage:            ${HPO_STAGE}"
log "Config set:           ${CONFIG_SET}"
log "N samples:            ${N_SAMPLES}"
log "Seed:                 ${SEED}"
log "CHUNK_SIZE:           ${CHUNK_SIZE}"
log "Concurrency:          ${CONCURRENCY}"
log "Qdrant mode:          ${QDRANT_RETRIEVAL_MODE}"
log "Keyword fanout:       ${KEYWORD_FANOUT_MODE}"
log "Configs:              ${CONFIG_NAMES[*]}"
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

    for i in "${!CONFIG_NAMES[@]}"; do
        run_ppr_config "${i}" "${DATASET}" "${WORKING_DIR}" "${WORKSPACE_ID}"
    done
done

log "================================================================"
log "Results summary"
log "================================================================"
print_summary
log "Done."
