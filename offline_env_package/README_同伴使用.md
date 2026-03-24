# 🚀 离线部署快速指南（同伴版）

这是 **neo4j-milvus branch** 的离线部署包。

## ⚡ 一键部署

### 步骤 1️⃣：解压离线包

```bash
tar -xzf offline_rag_env.tar.gz
cd offline_env_package
```

### 步骤 2️⃣：运行自动部署脚本

```bash
bash auto_deploy.sh
```

脚本会自动：
- ✅ 检测 Python 和 GPU 环境
- ✅ 离线安装所有依赖包
- ✅ 初始化数据目录
- ✅ 验证 LightRAG 和 RAG-Anything 导入
- ✅ 验证 V2/V3 模块导入

### 步骤 3️⃣：启动系统

```bash
bash start_rag.sh
```

访问 http://localhost:9621

---

## 📋 V2/V3 配置说明

本 branch 新增了两个核心功能：

### V2：同义词链接 (Synonym Linking)

```python
# constants.py 中的默认值（可通过 .env 覆盖）
ENABLE_SYNONYM_LINKING = False              # 启用/禁用
SYNONYMY_THRESHOLD = 0.8                    # 相似度阈值
SYNONYMY_TOPK = 100                         # KNN top-K
SYNONYMY_MIN_ENTITY_LEN = 2                 # 最小实体长度
```

启用 V2：
```bash
# 方式 1：修改 constants.py
nano code/rag-anything/raganything/constants.py
# 找到 DEFAULT_ENABLE_SYNONYM_LINKING = False，改为 True

# 方式 2：设置环境变量或 .env
export ENABLE_SYNONYM_LINKING=true
```

### V3：PPR 多跳推理 (PPR Multi-hop)

```python
# 在查询时启用（QueryParam）
enable_multi_hop=True
multi_hop_depth=2
ppr_damping=0.5
ppr_top_k=50
passage_node_weight=0.05
```

使用 V3：
```python
from lightrag.base import QueryParam

result = await rag.aquery(
    query="问题",
    param=QueryParam(
        mode="mix",
        enable_multi_hop=True,      # 启用 V3
        multi_hop_depth=2,
        ppr_top_k=50,
    )
)
```

或命令行：
```bash
# 通过 QueryParam 在查询时传入
python -c "
from lightrag.base import QueryParam
from raganything import RAGAnything
import asyncio

async def main():
    rag = RAGAnything(...)
    result = await rag.aquery(
        'question',
        param=QueryParam(enable_multi_hop=True)
    )

asyncio.run(main())
"
```

---

## 📁 目录结构

```
offline_env_package/
├── wheels/                    ← Python 依赖包
├── code/                      ← 源代码（neo4j-milvus branch）
│   ├── lightrag/
│   │   ├── lightrag.py
│   │   ├── operate.py
│   │   ├── synonym_linking.py     ← V2 新增
│   │   └── ppr.py                 ← V3 新增
│   └── rag-anything/
│       └── raganything/
│           └── constants.py       ← 配置（V2/V3 参数在这里）
├── data/                      ← 运行时数据
├── .env.offline.example       ← V2/V3 参数覆盖模板
├── auto_deploy.sh             ← 一键部署脚本
├── start_rag.sh               ← 启动脚本
└── deploy.log                 ← 部署日志
```

---

## 🔧 配置 V2/V3 参数

### 方式 1：编辑 constants.py（推荐）

```bash
cd code/rag-anything/raganything/

# 编辑配置
nano constants.py

# 找到以下行（约第 139-158 行）
DEFAULT_ENABLE_SYNONYM_LINKING = False
DEFAULT_SYNONYMY_THRESHOLD = 0.8
DEFAULT_SYNONYMY_TOPK = 100
DEFAULT_SYNONYMY_MIN_ENTITY_LEN = 2
```

### 方式 2：创建 .env 文件覆盖（可选）

```bash
# 复制模板
cp .env.offline.example .env

# 编辑
nano .env

# 只需要修改要改的参数，其余保持默认
```

.env 文件示例：
```bash
# V2: 同义词链接
ENABLE_SYNONYM_LINKING=true
SYNONYMY_THRESHOLD=0.8
SYNONYMY_TOPK=100

# V3: PPR 多跳
# (V3 通过 QueryParam 在查询时传入，不在 constants.py)
```

---

## 🎯 使用示例

### 示例 1：启用 V2 并索引文档

```bash
cd code

# 修改 constants.py 启用 V2
sed -i 's/DEFAULT_ENABLE_SYNONYM_LINKING = False/DEFAULT_ENABLE_SYNONYM_LINKING = True/' \
    rag-anything/raganything/constants.py

# 索引文档（ingestion 阶段会自动建立 SYNONYM 边）
python -m raganything.services.local_rag \
    -p ./documents \
    -i my_graph
```

### 示例 2：查询时启用 V3

```python
import asyncio
from lightrag.base import QueryParam
from raganything.services.local_rag import LocalRagService

async def main():
    service = LocalRagService.from_env()

    # V3: 启用 PPR 多跳推理
    result = await service.aquery(
        query="问题",
        graph_name="my_graph",
        param=QueryParam(
            mode="hybrid",
            enable_multi_hop=True,       # V3 启用
            multi_hop_depth=2,
            ppr_top_k=50,
        )
    )

    print(result["response"])

asyncio.run(main())
```

### 示例 3：V2 + V3 协同

```bash
# 1. 修改 constants.py 启用 V2
ENABLE_SYNONYM_LINKING=true

# 2. 索引（V2 会建立同义词边）
python -m raganything.services.local_rag -p ./docs -i my_graph

# 3. 查询（V3 会利用 SYNONYM 边做多跳推理）
python -c "
from lightrag.base import QueryParam
from raganything.services.local_rag import LocalRagService
import asyncio

async def main():
    service = LocalRagService.from_env()
    result = await service.aquery(
        'question',
        'my_graph',
        param=QueryParam(enable_multi_hop=True)  # V3 启用
    )
    print(result['response'])

asyncio.run(main())
"
```

---

## ✅ 验证部署成功

部署完成后，你会看到：

```
✓ Python 版本: 3.10.x
✓ pip 已找到
✓ 依赖包安装完成
✓ LightRAG 导入成功
✓ RAG-Anything 导入成功
✓ V2/V3 模块导入成功

====== 部署完成！======
```

然后运行：
```bash
bash start_rag.sh
# 访问 http://localhost:9621
```

---

## 🔍 关键参数速查

| 功能 | 位置 | 参数 | 默认值 |
|------|------|------|--------|
| V2 启用 | constants.py | DEFAULT_ENABLE_SYNONYM_LINKING | False |
| V2 阈值 | constants.py | DEFAULT_SYNONYMY_THRESHOLD | 0.8 |
| V2 TopK | constants.py | DEFAULT_SYNONYMY_TOPK | 100 |
| V3 启用 | QueryParam | enable_multi_hop | False |
| V3 深度 | QueryParam | multi_hop_depth | 2 |
| V3 TopK | QueryParam | ppr_top_k | 50 |

---

## 📞 问题排查

### Q: 模块导入失败

```bash
# 检查日志
cat deploy.log

# 重新安装依赖
pip install --no-index --find-links ./wheels -r code/requirements.txt
```

### Q: 如何修改 V2/V3 参数

```bash
# 编辑 constants.py
cd code/rag-anything/raganything/
nano constants.py

# 搜索相关参数并修改
```

### Q: V3 (PPR) 如何使用

V3 不在 constants.py 中，而是在查询时通过 `QueryParam` 传入：

```python
# 方式 1: Python API
param = QueryParam(enable_multi_hop=True, ppr_top_k=50)
result = await rag.aquery(query, param=param)

# 方式 2: 继承后修改查询参数
# (具体取决于 raganything 的查询接口)
```

---

## 💡 性能优化建议

如果 indexing 很慢：
```bash
# 编辑 constants.py，降低并发
DEFAULT_LLM_MODEL_MAX_ASYNC=2
MAX_PARALLEL_INSERT=1
```

如果查询很慢：
```bash
# 在 QueryParam 中调整
enable_multi_hop=False          # 关闭 V3 加速
CHUNK_TOP_K=5                   # 减少 chunk 数量
```

---

简言之：

✅ 部署：`bash auto_deploy.sh && bash start_rag.sh`

✅ 配置 V2：编辑 `constants.py` 中的 `DEFAULT_ENABLE_SYNONYM_LINKING`

✅ 使用 V3：查询时在 `QueryParam` 中设 `enable_multi_hop=True`

就这么简单！🎉
