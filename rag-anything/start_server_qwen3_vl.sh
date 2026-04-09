#!/bin/bash

# 1. 打印提示信息
echo "=================================================="
echo "正在启动 Qwen3-VL-30B-A3B-Instruct-FP8 模型服务..."
echo "端口: 8001 | GPU: 1 (请确保此卡空闲)"
echo "=================================================="

# 2. 激活虚拟环境
source /data/y50056788/Yaliang/projects/lightrag/.venv/bin/activate

# 3. 设置只使用 GPU 1 (防止和你的 Python 脚本抢 GPU 0)
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export OMP_NUM_THREADS=1

# 4. 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model /data/y50056788/Yaliang/models//Qwen3-VL-30B-A3B-Instruct-FP8 \
    --served-model-name "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8" \
    --trust-remote-code \
    --port 8001 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 65536 \
    --dtype bfloat16 \
    --quantization fp8 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 32