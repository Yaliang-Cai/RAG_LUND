#!/bin/bash

# 1. 打印提示信息
echo "=================================================="
echo "正在启动 InternVL2-26B-AWQ 模型服务..."
echo "端口: 8001 | GPU: 1 (请确保此卡空闲)"
echo "=================================================="

# 2. 激活虚拟环境
source /data/y50056788/Yaliang/projects/lightrag/.venv/bin/activate

# 3. 设置只使用 GPU 1 (防止和你的 Python 脚本抢 GPU 0)
export CUDA_VISIBLE_DEVICES=1

# 4. 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model /data/h50056787/models/InternVL2-26B-AWQ \
    --served-model-name "OpenGVLab/InternVL2-26B-AWQ" \
    --trust-remote-code \
    --port 8001 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --dtype float16 \
    --quantization awq \
    --enforce-eager \