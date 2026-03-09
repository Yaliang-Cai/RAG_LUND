#!/bin/bash

echo "=================================================="
echo "Starting Qwen3.5-35B-A3B-FP8 service..."
echo "Port: 8001 | CUDA_VISIBLE_DEVICES: 0,1"
echo "Thinking: disabled"
echo "=================================================="

source /data/y50056788/Yaliang/models/.venv_qwen35/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
    --model /data/y50056788/Yaliang/models/Qwen3.5-35B-A3B-FP8 \
    --served-model-name "Qwen/Qwen3.5-35B-A3B-FP8" \
    --trust-remote-code \
    --port 8001 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 49152 \
    --dtype bfloat16 \
    --quantization fp8 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enforce-eager