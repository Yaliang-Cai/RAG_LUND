#!/usr/bin/env bash
# =============================================================================
# HippoRAG2-aligned MultiHopQA evaluation runner
#
# Reproduces the exact experimental setup of HippoRAG2 (osunlp/HippoRAG_v2):
#   HotpotQA  : 1 000 queries, 9 221 corpus paragraphs
#   MuSiQue   : 1 000 queries, 6 119 corpus paragraphs
#   2Wiki     : 1 000 queries, 11 656 corpus paragraphs
#
# Chunk size  : 1 200 tokens  (DEFAULT_CHUNK_TOKEN_SIZE in raganything/constants.py)
# Metrics     : EM, F1, Recall@2, Recall@5  (passage-level via source_map)
#
# -----------------------------------------------------------------------------
# 快速上手
# -----------------------------------------------------------------------------
#
#   第一步：修改下方 ① Paths 中的 WORKSPACE_ROOT 和 RESULTS_ROOT，其余保持默认。
#
#   全流程（下载数据集 → 建索引 → 跑 naive/hybrid/ppr/auto 四个 mode）：
#     bash run_hipporag2_eval.sh
#
#   只跑单个数据集：
#     bash run_hipporag2_eval.sh hotpotqa
#     bash run_hipporag2_eval.sh musique
#     bash run_hipporag2_eval.sh 2wiki
#
#   索引已建好，跳过 build_index 直接重跑评测（切换 mode 不需要重新索引）：
#     SKIP_INDEX=1 bash run_hipporag2_eval.sh hotpotqa
#
#   修改要测试的 mode 列表（见下方 ② MODES），然后带 SKIP_INDEX=1 重跑即可：
#     SKIP_INDEX=1 bash run_hipporag2_eval.sh   # 读取最新 MODES 变量，索引不变
#
# -----------------------------------------------------------------------------
# 查看汇总结果表格
# -----------------------------------------------------------------------------
#   评测结束后脚本会自动调用 print_results_table.py 打印结果。
#   也可以单独运行：
#
#     python evaluate_local/MultiHopQA/print_results_table.py \
#         --results-root <RESULTS_ROOT>
#
#   可选参数：
#     --datasets  hotpotqa musique 2wiki   # 只打印指定数据集
#     --modes     naive hybrid ppr auto    # 只打印指定 mode
#     --recall-k  2 5                      # 指定 Recall@K 的 K 值
#
#   输出示例：
#     Dataset      Mode          EM       F1     R@2     R@5
#     --------------------------------------------------------
#     hotpotqa     naive     0.3210   0.4120  0.5200  0.6800
#     hotpotqa     ppr       0.4120   0.5080  0.6800  0.8100
#
# -----------------------------------------------------------------------------
# 重要说明
# -----------------------------------------------------------------------------
#   - build_index.py 只需运行一次，生成的 LightRAG 索引与 query mode 无关。
#     naive / hybrid / ppr / auto 四个 mode 读取同一份索引，无需重建。
#   - evaluate_multihop.py 的 --modes 参数接受多个值，会依次跑完后统一输出。
#   - Recall@K 使用 passage-level source_map 计算，与 HippoRAG2 对齐。
#     source_map 在 build_index 阶段自动生成，存储在 WORKING_DIR 下。
#   - 结果 JSONL 支持 --resume 断点续跑，不会重复计算已完成的 query。
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# ① Paths — 运行前修改 WORKSPACE_ROOT 和 RESULTS_ROOT，其余保持默认即可
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # RAG_LUND/
RAGANYTHING_ROOT="${REPO_ROOT}/rag-anything"
DATA_DIR="${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/hipporag2_data"  # HippoRAG2 JSON 文件，自动下载到此处
WORKSPACE_ROOT="/data/workspaces/multihopqa_hr2"   # LightRAG 索引存储位置（需要修改）
RESULTS_ROOT="${RAGANYTHING_ROOT}/results/multihopqa_hr2"   # 结果输出位置（可修改）

# ---------------------------------------------------------------------------
# ② Evaluation settings — 修改 MODES 后用 SKIP_INDEX=1 重跑即可切换 mode
# ---------------------------------------------------------------------------
# 支持的 mode（见 evaluate_multihop.py VALID_MODES）：
#   naive      — 纯向量检索，无 KG
#   hybrid     — 实体 + chunk 混合检索（默认模式）
#   ppr        — Personalized PageRank（对齐 HippoRAG2 核心算法）
#   auto       — 自动分类路由（simple/medium/complex 三轨）
#   full       — 所有路径 RRF 融合（最强但最慢）
MODES="naive hybrid ppr auto"          # 要测试的 mode，空格分隔；加 "full" 可测 RRF-all

RECALL_K="2 5"                         # Recall@K 的 K 值；与 GFM-RAG/HippoRAG2 论文对齐
CONCURRENCY=16                         # 并发 query 数，根据 GPU 显存调整
TOP_K=10                               # 每次检索返回的实体/关系数
CHUNK_TOP_K=5                          # rerank 后送入 LLM 的 chunk 数
MAX_TOTAL_TOKENS=45000                 # LLM 上下文 token 上限

# PPR 参数 — 与 HippoRAG2 默认值对齐（raganything/constants.py）
PPR_DAMPING=0.5                        # DEFAULT_PPR_DAMPING
PPR_TOP_K=50                           # DEFAULT_PPR_TOP_K
PPR_QA_TOP_K=5                         # DEFAULT_PPR_QA_TOP_K：PPR 后送 LLM 的 chunk 数
PASSAGE_NODE_WEIGHT=0.05               # DEFAULT_PASSAGE_NODE_WEIGHT（HippoRAG2 DPR 权重）
LINKING_TOP_K=5                        # DEFAULT_LINKING_TOP_K（recognition memory 实体种子数）
RECOGNITION_TOP_K=20                   # DEFAULT_RECOGNITION_TOP_K（global PPR relation 候选数）

# Ingest settings
INGEST_BATCH_SIZE=256
BATCH_DOC_CONCURRENCY=2
LLM_MAX_ASYNC=48

# ---------------------------------------------------------------------------
# ③ Dataset list
# ---------------------------------------------------------------------------
DATASETS=("hotpotqa" "musique" "2wiki")
if [[ $# -ge 1 ]]; then
    DATASETS=("$1")
fi

SKIP_INDEX="${SKIP_INDEX:-0}"

# ---------------------------------------------------------------------------
# ④ Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Step 0: Download HippoRAG2 datasets (idempotent — skipped if files exist)
# ---------------------------------------------------------------------------
log "=== Step 0: Download HippoRAG2 datasets ==="
python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/download_hipporag2_datasets.py" \
    --output-dir "${DATA_DIR}"

# ---------------------------------------------------------------------------
# Steps 1+2: For each dataset — build index, then evaluate all modes at once
# ---------------------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
    WORKSPACE_ID="${DATASET}_hr2"
    WORKING_DIR="${WORKSPACE_ROOT}/${DATASET}"
    RESULTS_DIR="${RESULTS_ROOT}/${DATASET}"

    log "================================================================"
    log "Dataset: ${DATASET}"
    log "Workspace: ${WORKING_DIR}"
    log "Results:   ${RESULTS_DIR}"
    log "================================================================"

    mkdir -p "${WORKING_DIR}" "${RESULTS_DIR}"

    # ── Step 1: Build index ──────────────────────────────────────────────
    if [[ "${SKIP_INDEX}" == "1" ]]; then
        log "[${DATASET}] SKIP_INDEX=1 — skipping build_index"
    else
        log "[${DATASET}] Step 1: Building index (corpus from HippoRAG2 files)"
        python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/build_index.py" \
            --dataset          "${DATASET}" \
            --workspace        "${WORKSPACE_ID}" \
            --working-dir      "${WORKING_DIR}" \
            --hipporag2-data-dir "${DATA_DIR}" \
            --ingest-batch-size  "${INGEST_BATCH_SIZE}" \
            --batch-doc-concurrency "${BATCH_DOC_CONCURRENCY}" \
            --llm-model-max-async   "${LLM_MAX_ASYNC}" \
            --resume
        log "[${DATASET}] Index build complete."
    fi

    # ── Step 2: Evaluate — all modes in ONE run ──────────────────────────
    # After build_index is done once, re-running this script with SKIP_INDEX=1
    # lets you test additional modes or re-score without re-indexing.
    log "[${DATASET}] Step 2: Evaluating modes: ${MODES}"
    python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/evaluate_multihop.py" \
        --dataset          "${DATASET}" \
        --workspace        "${WORKSPACE_ID}" \
        --working-dir      "${WORKING_DIR}" \
        --hipporag2-data-dir "${DATA_DIR}" \
        --output-dir       "${RESULTS_DIR}" \
        --modes            ${MODES} \
        --recall-k         ${RECALL_K} \
        --concurrency      "${CONCURRENCY}" \
        --top-k            "${TOP_K}" \
        --chunk-top-k      "${CHUNK_TOP_K}" \
        --max-total-tokens "${MAX_TOTAL_TOKENS}" \
        --ppr-damping      "${PPR_DAMPING}" \
        --ppr-top-k        "${PPR_TOP_K}" \
        --ppr-qa-top-k     "${PPR_QA_TOP_K}" \
        --passage-node-weight "${PASSAGE_NODE_WEIGHT}" \
        --linking-top-k    "${LINKING_TOP_K}" \
        --recognition-top-k "${RECOGNITION_TOP_K}" \
        --answer-context-mode chunk_only_prompt \
        --no-enable-kg-rerank \
        --no-ppr-enable-rerank \
        --bypass-query-cache \
        --resume
    log "[${DATASET}] Evaluation complete. Results: ${RESULTS_DIR}/${DATASET}_summary.json"
done

# ---------------------------------------------------------------------------
# Step 3: Print consolidated results table
# ---------------------------------------------------------------------------
log "================================================================"
log "Results summary"
log "================================================================"
python "${RAGANYTHING_ROOT}/evaluate_local/MultiHopQA/print_results_table.py" \
    --results-root "${RESULTS_ROOT}" \
    --datasets     "${DATASETS[@]}" \
    --modes        ${MODES} \
    --recall-k     ${RECALL_K}

log "Done."
