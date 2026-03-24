# 🚀 离线部署快速指南（同伴版）

你收到的是一个**完全离线的 RAG-Anything 系统包**。只需 **3 个命令**，即可在你的机器上运行。

---

## 📋 前置要求

- Python 3.8+ 已安装
- NVIDIA GPU（推荐）或 CPU（会很慢）
- 至少 10 GB 磁盘空间
- Linux / macOS / Windows (WSL2)

---

## ⚡ 一键部署（推荐）

### 步骤 1️⃣：解压离线包

```bash
# 获得的压缩文件
tar -xzf offline_rag_env.tar.gz
cd offline_env_package
```

### 步骤 2️⃣：运行自动部署脚本

```bash
bash auto_deploy.sh
```

**脚本会自动：**
- ✅ 检测你的 Python/GPU 环境
- ✅ 安装所有依赖包（离线）
- ✅ 检测本地模型路径
- ✅ 生成 `.env` 配置文件
- ✅ 验证安装是否成功
- ✅ 提示启动方式

**耗时：** 5-15 分钟（取决于网络和硬件）

### 步骤 3️⃣：选择启动方式

#### 方式 A：启动 Web API（推荐）

```bash
bash start_rag.sh
```

然后访问 http://localhost:9621 在浏览器中使用。

#### 方式 B：启动 vLLM 服务（如有 VLM 模型）

在一个 terminal 窗口：
```bash
bash start_vllm.sh
```

等待输出 `Uvicorn running on http://0.0.0.0:8001`。

#### 方式 C：命令行使用

```bash
cd code

# 索引文档
python -m raganything.services.local_rag \
  -p ./my_documents \
  -i my_knowledge_graph

# 查询
python -m raganything.services.local_rag \
  -q "你的问题" \
  -i my_knowledge_graph
```

---

## 🔧 如果部署失败

### 问题 1：缺少模型

```
log_warn "未找到 VLM 模型"
```

**解决：** 手动编辑 `data/.env`，找到模型实际位置：

```bash
# 查找模型
find . -name "bge-m3" -o -name "Qwen*" -type d

# 编辑 .env
nano .env

# 修改为实际路径，例如：
RAGANYTHING_EMBEDDING_MODEL_PATH=/full/path/to/bge-m3
```

### 问题 2：GPU 内存不足

```
RuntimeError: CUDA out of memory
```

**解决：** 编辑 `start_vllm.sh`：

```bash
# 找到这一行
--gpu-memory-utilization 0.7

# 改为更小的值（如 0.5）
--gpu-memory-utilization 0.5
```

### 问题 3：Python 包导入失败

```
ModuleNotFoundError: No module named 'torch'
```

**解决：** 重新运行安装

```bash
pip install --no-index --find-links ./wheels -r code/requirements.txt
```

### 问题 4：vLLM 无法连接

```
Connection refused: 8001
```

**解决：**
1. 确保 `start_vllm.sh` 正在运行
2. 检查防火墙是否允许 8001 端口
3. 修改 `.env` 的 `VLLM_API_BASE`：

```bash
VLLM_API_BASE=http://127.0.0.1:8001/v1  # 改为 127.0.0.1
```

---

## 📁 文件结构说明

```
offline_env_package/
├── wheels/                    ← Python 依赖包（pip install 用）
├── models/                    ← 模型文件（Embedding, VLM 等）
│   ├── embedding/
│   ├── reranker/
│   └── llm/
├── code/                      ← RAG-Anything 源代码
├── data/                      ← 运行时数据（索引、缓存等）
│   ├── rag_workspace/         ← 知识图谱数据（会自动生成）
│   ├── output/                ← 文档解析输出
│   └── logs/                  ← 日志文件
├── auto_deploy.sh             ← 一键部署脚本 ⭐
├── start_rag.sh               ← 启动 Web API
├── start_vllm.sh              ← 启动 vLLM（可选）
├── .env                       ← 配置文件（自动生成）
├── deploy.log                 ← 部署日志
└── README_同伴使用.md          ← 本文件
```

---

## 🎯 常用命令

### 索引文档

```bash
cd code

# 索引单个文件夹
python -m raganything.services.local_rag \
  -p ./documents \
  -i my_graph_name

# 索引后查询
python -m raganything.services.local_rag \
  -q "什么是知识图谱？" \
  -i my_graph_name
```

### 使用 Python API

```python
import asyncio
from raganything.services.local_rag import LocalRagService
from raganything.services.config import LocalRagSettings

async def main():
    # 自动从 .env 读取配置
    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings=settings)

    # 查询
    result = await service.aquery(
        query="问题",
        graph_name="my_graph_name",
        mode="hybrid"
    )
    print(result["response"])

asyncio.run(main())
```

### 启动 Web UI

```bash
bash start_rag.sh
# 访问 http://localhost:9621
```

---

## ⚙️ 性能优化

### 如果索引很慢

编辑 `.env`，降低并发数：

```bash
LLM_MODEL_MAX_ASYNC=2          # 从 4 改为 2
MAX_PARALLEL_INSERT=1          # 从 2 改为 1
```

### 如果查询很慢

```bash
# 改用 "local" 模式（不查图）
python -m raganything.services.local_rag \
  -q "问题" \
  -i my_graph \
  --mode local

# 或禁用重排
ENABLE_RERANK=false
```

### 如果内存不足

```bash
# 禁用 VLM 增强查询
ENABLE_VLM_ENHANCED=false

# 减少 chunk 数量
CHUNK_TOP_K=5
```

---

## 📞 需要帮助？

1. **检查日志**：
   ```bash
   cat deploy.log
   ```

2. **查看配置**：
   ```bash
   cat .env
   ```

3. **联系准备者**（你的同伴）：
   - 提供 `deploy.log` 的错误信息
   - 说明你的硬件配置（GPU 型号、内存等）

---

## ✅ 验证部署成功

部署完成后，你会看到：

```
✓ Python 版本: 3.10.x
✓ pip 已找到
✓ GPU 检测: NVIDIA RTX 4090 ...
✓ 找到嵌入模型: bge-m3
✓ PyTorch 导入成功
✓ LightRAG 导入成功
✓ RAG-Anything 导入成功
✓ Embedding 模型加载成功

====== 部署完成！======
```

接下来按照"后续步骤"进行即可！

---

## 🎓 快速入门示例

### 示例 1：索引一个 PDF 文件夹

```bash
mkdir test_docs
# 将 PDF 文件放入 test_docs/

python -m raganything.services.local_rag \
  -p test_docs \
  -i my_first_graph
```

### 示例 2：创建一个简单的查询脚本

创建 `query.py`：

```python
import asyncio
from raganything.services.local_rag import LocalRagService

async def main():
    service = LocalRagService.from_env()
    result = await service.aquery(
        query="请总结文档的主要内容",
        graph_name="my_first_graph"
    )
    print(result["response"])

asyncio.run(main())
```

运行：
```bash
python query.py
```

---

## 🔒 安全提示

- ⚠️ **不要** 修改 `.env` 中的 LLM 服务地址到公网服务（会泄露数据）
- ⚠️ **不要** 在 vLLM terminal 中输入敏感信息
- ⚠️ **定期备份** `data/rag_workspace/` 中的知识图谱数据

---

## 📦 包含的组件

| 组件 | 版本 | 用途 |
|------|------|------|
| LightRAG | 最新 | 知识图谱 + 向量检索 |
| RAG-Anything | 最新 | 多模态文档处理 |
| PyTorch | 2.0+ | 深度学习框架 |
| vLLM | 最新 | 本地 LLM 推理 |
| Sentence-Transformers | 最新 | Embedding 模型 |

---

## 💡 一句话总结

```bash
# 就这样
bash auto_deploy.sh
bash start_rag.sh
# 访问 http://localhost:9621
```

完成！🎉

---

**准备者注**：这个包是在 `neo4j-milvus` 分支打包的，包含最新的 V2/V3 升级（同义词链接 + PPR 多跳）。

有问题？检查 `deploy.log` 或联系准备者。
