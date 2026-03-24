#!/bin/bash

################################################################################
# 离线部署一键脚本 (auto_deploy.sh)
#
# 用途: 在离线机上自动部署 RAG-Anything + LightRAG
# 用法: bash auto_deploy.sh [部署目录]
#       如不指定部署目录，默认为当前脚本所在目录的 parent
#
# 脚本功能：
#   1. 环境检测（Python, GPU, 磁盘）
#   2. 离线安装 Python 依赖
#   3. 自动检测模型路径
#   4. 生成本机专用 .env
#   5. 初始化数据目录
#   6. 运行验证测试
#   7. (可选) 启动 vLLM
################################################################################

set -e

# ============================================================================
# 颜色和日志函数
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

LOG_FILE="deploy.log"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[✗ ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# ============================================================================
# 环境变量和路径检测
# ============================================================================

# 确定部署根目录
DEPLOY_ROOT="${1:-.}"
DEPLOY_ROOT=$(cd "$DEPLOY_ROOT" && pwd)

# 检查必要目录结构
if [ ! -d "$DEPLOY_ROOT/wheels" ] || [ ! -d "$DEPLOY_ROOT/models" ]; then
    log_error "缺失必要目录！确保以下目录存在："
    log_error "  $DEPLOY_ROOT/wheels/"
    log_error "  $DEPLOY_ROOT/models/"
    exit 1
fi

CODE_DIR="$DEPLOY_ROOT/code"
MODELS_DIR="$DEPLOY_ROOT/models"
WHEELS_DIR="$DEPLOY_ROOT/wheels"
DATA_DIR="$DEPLOY_ROOT/data"

# ============================================================================
# 第 1 步：环境检测
# ============================================================================

log_info "====== 第 1 步：环境检测 ======"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 未找到。请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_success "Python 版本: $PYTHON_VERSION"

# 检查 pip
if ! command -v pip3 &> /dev/null; then
    log_error "pip3 未找到。请先安装 pip"
    exit 1
fi

log_success "pip 已找到"

# 检查 GPU (可选)
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    log_success "GPU 检测: $GPU_INFO"
    DEVICE="cuda:0"
else
    log_warn "未检测到 NVIDIA GPU，将使用 CPU（推荐仅用于测试）"
    DEVICE="cpu"
fi

# 检查磁盘空间
DISK_AVAILABLE=$(df "$DEPLOY_ROOT" | awk 'NR==2 {print int($4/1024)}') # MB
DISK_REQUIRED=5000 # 5GB 最小要求

log_info "磁盘可用空间: ${DISK_AVAILABLE} MB"
if [ "$DISK_AVAILABLE" -lt "$DISK_REQUIRED" ]; then
    log_warn "磁盘空间可能不足（需要至少 ${DISK_REQUIRED} MB，现有 ${DISK_AVAILABLE} MB）"
fi

# ============================================================================
# 第 2 步：离线安装 Python 依赖
# ============================================================================

log_info ""
log_info "====== 第 2 步：安装 Python 依赖（离线模式）======"

# 检查 wheels 目录是否有包
WHEELS_COUNT=$(ls "$WHEELS_DIR"/*.whl 2>/dev/null | wc -l)
if [ "$WHEELS_COUNT" -lt 10 ]; then
    log_error "wheels 目录中的包数过少（仅 $WHEELS_COUNT 个）"
    log_error "请确保从联网机完整下载了所有依赖包"
    exit 1
fi

log_info "检测到 $WHEELS_COUNT 个 Python 包"

# 检查 requirements.txt
if [ ! -f "$CODE_DIR/requirements.txt" ]; then
    log_error "未找到 $CODE_DIR/requirements.txt"
    exit 1
fi

log_info "开始离线安装（这可能需要几分钟）..."
if pip install --no-index --find-links "$WHEELS_DIR" -r "$CODE_DIR/requirements.txt" >> "$LOG_FILE" 2>&1; then
    log_success "依赖包安装完成"
else
    log_error "依赖包安装失败，详见 $LOG_FILE"
    exit 1
fi

# ============================================================================
# 第 3 步：自动检测模型路径
# ============================================================================

log_info ""
log_info "====== 第 3 步：自动检测模型路径 ======"

# 遍历常见模型名称
EMBEDDING_PATH=""
RERANKER_PATH=""
VISION_PATH=""

declare -a EMBEDDING_NAMES=("bge-m3" "all-MiniLM-L6-v2" "bge-base-zh-v1.5")
declare -a RERANKER_NAMES=("bge-reranker-v2-m3" "bge-reranker-v2-m3-multilingual")
declare -a VISION_NAMES=("Qwen2-VL-7B-Instruct" "InternVL2-2B" "Qwen/Qwen2-VL-7B-Instruct")

log_info "扫描嵌入模型..."
for name in "${EMBEDDING_NAMES[@]}"; do
    if [ -d "$MODELS_DIR/embedding/$name" ]; then
        EMBEDDING_PATH="$MODELS_DIR/embedding/$name"
        log_success "找到嵌入模型: $name"
        break
    fi
done

if [ -z "$EMBEDDING_PATH" ]; then
    log_warn "未找到嵌入模型，将使用 HuggingFace Hub（需要联网）"
    EMBEDDING_PATH="sentence-transformers/all-MiniLM-L6-v2"
fi

log_info "扫描重排模型..."
for name in "${RERANKER_NAMES[@]}"; do
    if [ -d "$MODELS_DIR/reranker/$name" ] || [ -d "$MODELS_DIR/embedding/$name" ]; then
        RERANKER_PATH=$(find "$MODELS_DIR" -maxdepth 3 -type d -name "$name" | head -1)
        if [ -n "$RERANKER_PATH" ]; then
            log_success "找到重排模型: $name"
            break
        fi
    fi
done

if [ -z "$RERANKER_PATH" ]; then
    log_warn "未找到重排模型，将禁用重排功能"
    ENABLE_RERANK="false"
fi

log_info "扫描 VLM 模型..."
for name in "${VISION_NAMES[@]}"; do
    if [ -d "$MODELS_DIR/llm/$name" ] || [ -d "$MODELS_DIR/$name" ]; then
        VISION_PATH=$(find "$MODELS_DIR" -maxdepth 3 -type d -name "${name##*/}" | head -1)
        if [ -n "$VISION_PATH" ]; then
            log_success "找到 VLM 模型: ${name##*/}"
            break
        fi
    fi
done

if [ -z "$VISION_PATH" ]; then
    log_warn "未找到 VLM 模型，将禁用图像处理"
fi

# ============================================================================
# 第 4 步：生成本机专用 .env
# ============================================================================

log_info ""
log_info "====== 第 4 步：生成 .env 配置 ======"

# 创建数据目录
mkdir -p "$DATA_DIR"/{rag_workspace,output,logs,cache}

# 生成 .env 文件
ENV_FILE="$DEPLOY_ROOT/.env"
cat > "$ENV_FILE" << EOF
# ================================================================
# 自动生成的离线部署配置
# 生成时间: $(date)
# 部署根目录: $DEPLOY_ROOT
# ================================================================

# --- 模型路径（自动检测） ---
RAGANYTHING_EMBEDDING_MODEL_PATH=$EMBEDDING_PATH
$([ -n "$RERANKER_PATH" ] && echo "RAGANYTHING_RERANK_MODEL_PATH=$RERANKER_PATH" || echo "# RAGANYTHING_RERANK_MODEL_PATH=未找到")
$([ -n "$VISION_PATH" ] && echo "RAGANYTHING_VISION_MODEL_PATH=$VISION_PATH" || echo "# RAGANYTHING_VISION_MODEL_PATH=未找到")

# --- 工作目录 ---
WORKING_DIR=$DATA_DIR/rag_workspace
OUTPUT_DIR=$DATA_DIR/output
LOG_DIR=$DATA_DIR/logs

# --- LLM 服务（本地 vLLM） ---
VLLM_API_BASE=http://localhost:8001/v1
VLLM_API_KEY=EMPTY
$([ -n "$VISION_PATH" ] && echo "LLM_MODEL_NAME=${VISION_PATH##*/}" || echo "# LLM_MODEL_NAME=")

# --- 系统资源 ---
DEVICE=$DEVICE
VLLM_GPU_MEMORY_UTILIZATION=0.7

# --- 查询参数 ---
TOP_K=20
CHUNK_TOP_K=10
ENABLE_RERANK=$([ -n "$RERANKER_PATH" ] && echo "true" || echo "false")
ENABLE_VLM_ENHANCED=$([ -n "$VISION_PATH" ] && echo "true" || echo "false")

# --- 性能调优（离线机） ---
LLM_MODEL_MAX_ASYNC=4
MAX_PARALLEL_INSERT=2
EMBEDDING_BATCH_NUM=16
EMBEDDING_FUNC_MAX_ASYNC=4

# --- 文档解析 ---
PARSER=mineru
PARSE_METHOD=auto
CHUNKING_STRATEGY=token
CHUNK_TOKEN_SIZE=1200

# ================================================================
# 注意：如果模型路径自动检测失败，请手动编辑此文件
# ================================================================
EOF

log_success "生成 .env 配置: $ENV_FILE"

# 显示 .env 内容摘要
log_info "配置摘要："
log_info "  嵌入模型: $EMBEDDING_PATH"
[ -n "$RERANKER_PATH" ] && log_info "  重排模型: $RERANKER_PATH" || log_info "  重排模型: （未启用）"
[ -n "$VISION_PATH" ] && log_info "  VLM 模型: $VISION_PATH" || log_info "  VLM 模型: （未启用）"
log_info "  计算设备: $DEVICE"
log_info "  工作目录: $DATA_DIR"

# ============================================================================
# 第 5 步：验证安装
# ============================================================================

log_info ""
log_info "====== 第 5 步：验证安装 ======"

export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# 测试 PyTorch
log_info "测试 PyTorch..."
if python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}')"; then
    log_success "PyTorch 导入成功"
else
    log_warn "PyTorch 导入失败"
fi

# 测试 LightRAG
log_info "测试 LightRAG..."
if python3 -c "from lightrag import LightRAG; print('LightRAG 已导入')"; then
    log_success "LightRAG 导入成功"
else
    log_error "LightRAG 导入失败"
fi

# 测试 RAG-Anything
log_info "测试 RAG-Anything..."
if python3 -c "from raganything import RAGAnything; print('RAG-Anything 已导入')"; then
    log_success "RAG-Anything 导入成功"
else
    log_error "RAG-Anything 导入失败"
fi

# 测试 Embedding 模型加载（如果是本地路径）
if [[ "$EMBEDDING_PATH" == /* ]]; then
    log_info "加载嵌入模型（首次可能较慢）..."
    if timeout 120 python3 << PYEOF
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer('$EMBEDDING_PATH')
    emb = model.encode(['test'])
    print(f'✓ Embedding 模型加载成功，维度: {len(emb[0])}')
except Exception as e:
    print(f'✗ Embedding 模型加载失败: {e}')
    exit(1)
PYEOF
    then
        log_success "嵌入模型加载成功"
    else
        log_warn "嵌入模型加载失败或超时"
    fi
fi

# ============================================================================
# 第 6 步：生成启动脚本
# ============================================================================

log_info ""
log_info "====== 第 6 步：生成启动脚本 ======"

# 生成 vLLM 启动脚本（如果有 VLM 模型）
if [ -n "$VISION_PATH" ]; then
    START_VLLM_SCRIPT="$DEPLOY_ROOT/start_vllm.sh"
    cat > "$START_VLLM_SCRIPT" << 'VLLMEOF'
#!/bin/bash
set -e

VISION_MODEL_PATH="$(cd "$(dirname "$0")" && pwd)/models/llm/$(ls models/llm | head -1)"

echo "启动 vLLM 服务..."
echo "模型: $VISION_MODEL_PATH"
echo "监听: http://localhost:8001/v1"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model "$VISION_MODEL_PATH" \
  --tensor-parallel-size 1 \
  --port 8001 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096
VLLMEOF

    chmod +x "$START_VLLM_SCRIPT"
    log_success "生成 vLLM 启动脚本: $START_VLLM_SCRIPT"
fi

# 生成主启动脚本
START_SCRIPT="$DEPLOY_ROOT/start_rag.sh"
cat > "$START_SCRIPT" << RAGEOF
#!/bin/bash
set -e

DEPLOY_ROOT="$DEPLOY_ROOT"
CODE_DIR="\$DEPLOY_ROOT/code"

export PYTHONPATH="\$CODE_DIR:\$PYTHONPATH"

# 加载 .env
[ -f "\$DEPLOY_ROOT/.env" ] && source "\$DEPLOY_ROOT/.env"

cd "\$CODE_DIR"

echo "======================================"
echo "RAG-Anything 离线系统已启动"
echo "======================================"
echo "索引文档: python -m raganything.services.local_rag -p <folder>"
echo "启动 API: uvicorn server.app:app --host 0.0.0.0 --port 9621"
echo "查询示例: python -c 'from raganything import RAGAnything; ...'"
echo ""

# 默认启动 API 服务
uvicorn server.app:app --host 0.0.0.0 --port 9621
RAGEOF

chmod +x "$START_SCRIPT"
log_success "生成 RAG 启动脚本: $START_SCRIPT"

# ============================================================================
# 第 7 步：总结
# ============================================================================

log_info ""
log_success "====== 部署完成！======"
log_info ""
log_info "部署位置: $DEPLOY_ROOT"
log_info "数据目录: $DATA_DIR"
log_info "配置文件: $ENV_FILE"
log_info "日志文件: $LOG_FILE"
log_info ""

echo -e "${GREEN}后续步骤：${NC}"
echo ""

if [ -n "$VISION_PATH" ]; then
    echo "1️⃣  (可选) 启动 vLLM 服务"
    echo "   bash $START_VLLM_SCRIPT"
    echo ""
fi

echo "2️⃣  进入代码目录"
echo "   cd $CODE_DIR"
echo ""

echo "3️⃣  索引文档"
echo "   python -m raganything.services.local_rag -p ./documents -i my_graph"
echo ""

echo "4️⃣  启动 Web API（可选）"
echo "   bash $START_SCRIPT"
echo ""

echo "5️⃣  查询示例"
echo "   python -m raganything.services.local_rag -q '问题' -i my_graph"
echo ""

log_info ""
log_warn "重要提示："
log_warn "- 如果模型路径检测不正确，请手动编辑 $ENV_FILE"
log_warn "- 首次运行时可能会下载缓存，请耐心等待"
log_warn "- 所有日志已记录到 $LOG_FILE"

# 询问是否启动 vLLM
if [ -n "$VISION_PATH" ]; then
    echo ""
    read -p "是否现在启动 vLLM 服务？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash "$START_VLLM_SCRIPT"
    fi
fi
