#!/bin/bash

################################################################################
# 离线部署一键脚本 (auto_deploy.sh)
#
# 用途: 在离线机上自动部署 RAG-Anything + LightRAG (neo4j-milvus branch)
# 用法: bash auto_deploy.sh [部署目录]
#
# 脚本功能：
#   1. 环境检测（Python, GPU, 磁盘）
#   2. 离线安装 Python 依赖
#   3. 初始化数据目录
#   4. 运行验证测试
#   5. (可选) 生成 .env 配置
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
if [ ! -d "$DEPLOY_ROOT/wheels" ] || [ ! -d "$DEPLOY_ROOT/code" ]; then
    log_error "缺失必要目录！确保以下目录存在："
    log_error "  $DEPLOY_ROOT/wheels/"
    log_error "  $DEPLOY_ROOT/code/"
    exit 1
fi

CODE_DIR="$DEPLOY_ROOT/code"
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
else
    log_warn "未检测到 NVIDIA GPU，将使用 CPU"
fi

# 检查磁盘空间
DISK_AVAILABLE=$(df "$DEPLOY_ROOT" | awk 'NR==2 {print int($4/1024)}') # MB
log_info "磁盘可用空间: ${DISK_AVAILABLE} MB"

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
# 第 3 步：初始化数据目录
# ============================================================================

log_info ""
log_info "====== 第 3 步：初始化数据目录 ======"

mkdir -p "$DATA_DIR"/{rag_workspace,output,logs,cache}
log_success "数据目录已创建: $DATA_DIR"

# ============================================================================
# 第 4 步：验证安装
# ============================================================================

log_info ""
log_info "====== 第 4 步：验证安装 ======"

export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# 测试 LightRAG
log_info "测试 LightRAG..."
if python3 -c "from lightrag import LightRAG; print('✓ LightRAG 已导入')"; then
    log_success "LightRAG 导入成功"
else
    log_error "LightRAG 导入失败"
    exit 1
fi

# 测试 RAG-Anything
log_info "测试 RAG-Anything..."
if python3 -c "from raganything import RAGAnything; print('✓ RAG-Anything 已导入')"; then
    log_success "RAG-Anything 导入成功"
else
    log_error "RAG-Anything 导入失败"
    exit 1
fi

# 测试 V2/V3 模块
log_info "测试 V2/V3 模块..."
if python3 -c "from lightrag.synonym_linking import build_synonym_edges; from lightrag.ppr import personalized_pagerank; print('✓ V2/V3 模块已导入')"; then
    log_success "V2/V3 模块导入成功"
else
    log_warn "V2/V3 模块导入可能失败（不影响基础功能）"
fi

# ============================================================================
# 第 5 步：生成启动脚本
# ============================================================================

log_info ""
log_info "====== 第 5 步：生成启动脚本 ======"

# 生成主启动脚本
START_SCRIPT="$DEPLOY_ROOT/start_rag.sh"
cat > "$START_SCRIPT" << 'RAGEOF'
#!/bin/bash
set -e

DEPLOY_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$DEPLOY_ROOT/code"

export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

cd "$CODE_DIR"

echo "======================================"
echo "RAG-Anything 离线系统已启动"
echo "======================================"
echo ""
echo "访问 Web UI: http://localhost:9621"
echo "API 文档: http://localhost:9621/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动 FastAPI 服务
uvicorn server.app:app --host 0.0.0.0 --port 9621
RAGEOF

chmod +x "$START_SCRIPT"
log_success "生成启动脚本: $START_SCRIPT"

# ============================================================================
# 第 6 步：提示配置说明
# ============================================================================

log_info ""
log_success "====== 部署完成！======"
log_info ""
log_info "部署位置: $DEPLOY_ROOT"
log_info "代码目录: $CODE_DIR"
log_info "数据目录: $DATA_DIR"
log_info "日志文件: $LOG_FILE"
log_info ""

echo -e "${GREEN}后续步骤：${NC}"
echo ""
echo "1️⃣  查看/编辑 constants.py 中的配置"
echo "   $CODE_DIR/rag-anything/raganything/constants.py"
echo ""
echo "2️⃣  (可选) 编辑 .env 配置 V2/V3 参数"
echo "   $DEPLOY_ROOT/.env"
echo ""
echo "3️⃣  进入代码目录"
echo "   cd $CODE_DIR"
echo ""
echo "4️⃣  索引文档"
echo "   python -m raganything.services.local_rag -p ./documents -i my_graph"
echo ""
echo "5️⃣  启动 Web API"
echo "   bash $START_SCRIPT"
echo ""

log_info ""
log_warn "重要提示："
log_warn "- V2/V3 配置参数定义在 $CODE_DIR/rag-anything/raganything/constants.py"
log_warn "- 可选：在 .env 中覆盖 V2/V3 的默认值"
log_warn "- 所有日志已记录到 $LOG_FILE"
