#!/bin/bash
set -euo pipefail

echo "=================================================="
echo "Starting Qwen3-VL-30B-A3B-Instruct-FP8 service..."
echo "Port: 8001 | CUDA_VISIBLE_DEVICES: 1"
echo "=================================================="

source /data/y50056788/Yaliang/projects/lightrag/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=1

python -m vllm.entrypoints.openai.api_server \
    --model /data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --served-model-name "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8" \
    --trust-remote-code \
    --port 8001 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --enable-chunked-prefill \
    --enforce-eager
