#!/bin/bash

# 1. 打印提示信息
echo "=================================================="
echo "正在启动 Qwen2-VL 模型服务..."
echo "端口: 8001 | GPU: 1 (请确保此卡空闲)"
echo "=================================================="

# 2. 激活虚拟环境
source /data/y50056788/Yaliang/projects/lightrag/.venv/bin/activate

# 3. 设置只使用 GPU 1 (防止和你的 Python 脚本抢 GPU 0)
export CUDA_VISIBLE_DEVICES=1

# 4. 启动 vLLM 服务
# 注意：这里加载的是【大模型 Qwen】，不是 BGE 嵌入模型
python -m vllm.entrypoints.openai.api_server \
    --model /data/h50056787/models/Qwen2-VL-7B-Instruct \
    --served-model-name "Qwen/Qwen2-VL-7B-Instruct" \
    --trust-remote-code \
    --port 8001 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 10}'