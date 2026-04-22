#!/usr/bin/env bash
# Multi-Hop QA Evaluation Pipeline — Smoke Test
#
# Run this before launching the full 500-question evaluation to verify
# the pipeline works end-to-end.
#
# Usage:
#   export WORKSPACE_ID=my_hotpotqa_workspace
#   export WORKING_DIR=/data/y50056788/.../rag_workspaces/my_hotpotqa_workspace
#   bash evaluate_local/MultiHopQA/smoke_test.sh
#
# Or override inline:
#   WORKSPACE_ID=foo WORKING_DIR=/path/to/ws bash evaluate_local/MultiHopQA/smoke_test.sh

set -euo pipefail

WORKSPACE_ID="${WORKSPACE_ID:-}"
WORKING_DIR="${WORKING_DIR:-}"
SMOKE_OUTPUT="/tmp/multihop_smoke_$(date +%Y%m%d_%H%M%S)"
DATASET="hotpotqa"
MODE="hybrid"
N=5
SEED=42

# ---------------------------------------------------------------------------
# Validate required env vars
# ---------------------------------------------------------------------------
if [[ -z "$WORKSPACE_ID" ]]; then
    echo "[smoke] ERROR: WORKSPACE_ID is not set."
    echo "        Export it before running: export WORKSPACE_ID=<your_workspace_id>"
    exit 1
fi
if [[ -z "$WORKING_DIR" ]]; then
    echo "[smoke] ERROR: WORKING_DIR is not set."
    echo "        Export it before running: export WORKING_DIR=<path_to_workspace_dir>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "================================================================"
echo "[smoke] Multi-Hop QA Pipeline Smoke Test"
echo "[smoke] Workspace:   $WORKSPACE_ID"
echo "[smoke] Working dir: $WORKING_DIR"
echo "[smoke] Output dir:  $SMOKE_OUTPUT"
echo "[smoke] Dataset:     $DATASET  mode=$MODE  n=$N  seed=$SEED"
echo "================================================================"

mkdir -p "$SMOKE_OUTPUT"

# ---------------------------------------------------------------------------
# Step 1: Fresh run (5 questions, hybrid mode)
# ---------------------------------------------------------------------------
echo ""
echo "[smoke] Step 1: Fresh 5-question run..."

cd "$PROJECT_ROOT"
python evaluate_local/MultiHopQA/evaluate_multihop.py \
    --dataset "$DATASET" \
    --workspace "$WORKSPACE_ID" \
    --working-dir "$WORKING_DIR" \
    --output-dir "$SMOKE_OUTPUT" \
    --modes "$MODE" \
    --n-samples "$N" \
    --seed "$SEED"

# Verify JSONL has exactly N lines
JSONL="$SMOKE_OUTPUT/${DATASET}_${MODE}_results.jsonl"
NLINES=$(wc -l < "$JSONL")
echo ""
echo "[smoke] JSONL line count: $NLINES (expected $N)"
if [[ "$NLINES" -ne "$N" ]]; then
    echo "[smoke] FAIL: Expected $N lines, got $NLINES"
    exit 1
fi

# Verify summary JSON exists and has expected keys
SUMMARY="$SMOKE_OUTPUT/${DATASET}_summary.json"
if [[ ! -f "$SUMMARY" ]]; then
    echo "[smoke] FAIL: summary JSON not found at $SUMMARY"
    exit 1
fi
python -c "
import json, sys
with open('$SUMMARY') as f:
    s = json.load(f)
assert 'results' in s, 'missing results key'
assert '$MODE' in s['results'], 'missing mode key'
m = s['results']['$MODE']
assert 'em' in m, 'missing em'
assert 'f1' in m, 'missing f1'
print(f'[smoke] Summary OK — EM={m[\"em\"]:.4f}  F1={m[\"f1\"]:.4f}')
"

# Spot-check: answers should be short (< 200 chars each)
echo ""
echo "[smoke] Answer spot-check (first 3 predictions):"
python -c "
import json
with open('$JSONL') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        r = json.loads(line)
        pred = r['pred']
        flag = 'OK' if len(pred) < 200 else 'LONG'
        print(f'  [{flag}] Q: {r[\"question\"][:60]}...')
        print(f'        Gold: {r[\"gold\"]}')
        print(f'        Pred: {pred[:120]}')
        print()
"

echo "[smoke] Step 1: PASSED"

# ---------------------------------------------------------------------------
# Step 2: Resume test (re-run with --resume, should skip all 5 and reuse results)
# ---------------------------------------------------------------------------
echo ""
echo "[smoke] Step 2: Resume test (should skip all $N already-answered questions)..."

RESUME_LOG="$SMOKE_OUTPUT/resume_test.log"
python evaluate_local/MultiHopQA/evaluate_multihop.py \
    --dataset "$DATASET" \
    --workspace "$WORKSPACE_ID" \
    --working-dir "$WORKING_DIR" \
    --output-dir "$SMOKE_OUTPUT" \
    --modes "$MODE" \
    --n-samples "$N" \
    --seed "$SEED" \
    --resume 2>&1 | tee "$RESUME_LOG"

# JSONL should still have exactly N lines (no duplicates)
NLINES_AFTER=$(wc -l < "$JSONL")
echo ""
echo "[smoke] JSONL line count after resume: $NLINES_AFTER (expected $N)"
if [[ "$NLINES_AFTER" -ne "$N" ]]; then
    echo "[smoke] FAIL: Resume added duplicate lines. Expected $N, got $NLINES_AFTER"
    exit 1
fi

# Check unique IDs == N
python -c "
import json
ids = set()
with open('$JSONL') as f:
    for line in f:
        line = line.strip()
        if line:
            ids.add(json.loads(line)['id'])
assert len(ids) == $N, f'Expected $N unique IDs, got {len(ids)}'
print(f'[smoke] Unique IDs: {len(ids)} — no duplicates')
"

echo "[smoke] Step 2: PASSED"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[smoke] ALL STEPS PASSED"
echo "[smoke] Output dir: $SMOKE_OUTPUT"
echo "[smoke] Ready to launch full evaluation with run_multihop_evals.py"
echo "================================================================"
