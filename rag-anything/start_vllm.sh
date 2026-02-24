#!/bin/bash
# 强制使用第二张显卡 (GPU 1)
export CUDA_VISIBLE_DEVICES=1

# 启动 vLLM
# 注意：--model 的路径必须和下载路径完全一致
python -m vllm.entrypoints.openai.api_server \
    --model /data/h50056787/models/Qwen2-VL-7B-Instruct \
    --served-model-name "Qwen/Qwen2-VL-7B-Instruct" \
    --trust-remote-code \
    --port 8001 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image": 10}'