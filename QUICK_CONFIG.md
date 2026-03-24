# 快速配置示例

## 场景 1：使用 Neo4j + Milvus Lite（推荐离线）

### 步骤 1：启动 Neo4j

```bash
# Docker 启动
docker run -d \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# 验证：访问 http://localhost:7474 看到 Neo4j Browser
```

### 步骤 2：编写初始化代码（你的代码中）

```python
from lightrag import LightRAG
from lightrag.kg import Neo4JStorage, MilvusVectorDBStorage

# 创建 RAG 实例
rag = LightRAG(
    working_dir="./rag_storage",

    # 使用 Neo4j 作为图存储
    graph_storage_cls=Neo4JStorage,
    graph_storage_kwargs={
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password",
    },

    # 使用 Milvus Lite 作为向量存储
    vector_storage_cls=MilvusVectorDBStorage,
    vector_storage_kwargs={
        "milvus_db_uri": "sqlite:///./milvus.db"  # 本地单文件
    }
)
```

### 步骤 3：启用 V2（可选）

编辑 `rag-anything/raganything/constants.py`：

```python
# 约第 197 行
DEFAULT_ENABLE_SYNONYM_LINKING = True   # 从 False 改为 True
```

或通过环境变量：

```bash
export ENABLE_SYNONYM_LINKING=true
```

### 步骤 4：索引文档

```python
import asyncio

async def main():
    # 索引文档（V2 会自动在 indexing 阶段建立 SYNONYM 边）
    await rag.ainsert_file(
        file_path="./document.pdf",
        doc_id="doc1"
    )

asyncio.run(main())
```

### 步骤 5：查询（启用 V3）

```python
from lightrag.base import QueryParam

async def main():
    # V3 多跳推理
    result = await rag.aquery(
        query="你的问题是什么？",
        param=QueryParam(
            mode="hybrid",
            enable_multi_hop=True,      # ⭐ V3 启用
            multi_hop_depth=2,          # PPR 深度
            ppr_top_k=50,               # PPR 返回的 chunk 数
        )
    )
    print(result["response"])

asyncio.run(main())
```

---

## 场景 2：只用 NetworkX + Milvus（简单，数据不持久）

```python
from lightrag import LightRAG
from lightrag.kg import MilvusVectorDBStorage

rag = LightRAG(
    working_dir="./rag_storage",
    # graph_storage_cls 不指定，默认用 NetworkX（内存）

    vector_storage_cls=MilvusVectorDBStorage,
    vector_storage_kwargs={
        "milvus_db_uri": "sqlite:///./milvus.db"
    }
)
```

---

## 场景 3：使用 LocalRagService（推荐用于部署）

如果用 `raganything/services/local_rag.py`：

```python
from raganything.services.local_rag import LocalRagService
from raganything.services.config import LocalRagSettings

async def main():
    # 从 .env 自动读取配置
    settings = LocalRagSettings.from_env()
    service = LocalRagService(settings=settings)

    # 索引
    await service.ainsert_from_folder(
        folder_path="./documents",
        graph_name="my_graph"
    )

    # 查询（V3）
    from lightrag.base import QueryParam
    result = await service.aquery(
        query="问题",
        graph_name="my_graph",
        param=QueryParam(enable_multi_hop=True)
    )

asyncio.run(main())
```

---

## 环境变量优先级

如果 constants.py 中设置不方便，可用环境变量覆盖：

```bash
# V2 参数
export ENABLE_SYNONYM_LINKING=true
export SYNONYMY_THRESHOLD=0.8
export SYNONYMY_TOPK=100

# Neo4j（如果用）
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

# 其他
export DEVICE=cuda:0
export WORKING_DIR=./rag_storage
```

然后在代码中读取：

```python
import os
from lightrag import LightRAG
from lightrag.kg import Neo4JStorage

rag = LightRAG(
    working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
    graph_storage_cls=Neo4JStorage,
    graph_storage_kwargs={
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "username": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
    }
)
```

---

## QueryParam 完整参数

在 `aquery()` 时传入：

```python
from lightrag.base import QueryParam

param = QueryParam(
    mode="hybrid",              # "naive" | "local" | "global" | "hybrid"
    top_k=20,                   # 实体数量
    chunk_top_k=10,             # chunk 数量

    # V3 参数
    enable_multi_hop=True,      # 启用 PPR 多跳
    multi_hop_depth=2,          # 图搜索深度
    ppr_top_k=50,               # PPR 返回的 chunk 数
)

result = await rag.aquery(query="...", param=param)
```

---

## 常见问题

### Q: V2 启用后索引会变慢吗？

A: 会。V2 在 indexing 时需要对每个实体做 embedding + KNN 查询。可以通过 constants.py 调整：

```python
DEFAULT_LLM_MODEL_MAX_ASYNC = 4    # 降低并发
DEFAULT_EMBEDDING_BATCH_NUM = 16   # 降低 batch 大小
```

### Q: Neo4j 和 Milvus 的数据存在哪里？

A:
- **Neo4j**：存在 Neo4j 服务的数据目录（Docker 内或本地）
- **Milvus Lite**：`./milvus.db` 单文件（可以随意移动）

### Q: 如何切换后端（比如从 NetworkX 改成 Neo4j）？

A: 修改 `graph_storage_cls` 参数即可：

```python
# 之前（NetworkX，内存）
rag = LightRAG(working_dir="./rag_storage")

# 之后（Neo4j，持久化）
from lightrag.kg import Neo4JStorage
rag = LightRAG(
    working_dir="./rag_storage",
    graph_storage_cls=Neo4JStorage,
    graph_storage_kwargs={...}
)
```

**注意**：索引数据不会自动迁移，需要重新索引。

### Q: V2 和 V3 可以分别禁用吗？

A: 可以。

- **禁用 V2**：`DEFAULT_ENABLE_SYNONYM_LINKING = False`
- **禁用 V3**：查询时不传 `enable_multi_hop` 或设为 `False`

---

## 完整工作流示例

```python
import asyncio
from lightrag import LightRAG
from lightrag.kg import Neo4JStorage, MilvusVectorDBStorage
from lightrag.base import QueryParam

async def main():
    # 1. 初始化（Neo4j + Milvus + V2）
    rag = LightRAG(
        working_dir="./rag_storage",
        graph_storage_cls=Neo4JStorage,
        graph_storage_kwargs={
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
        },
        vector_storage_cls=MilvusVectorDBStorage,
        vector_storage_kwargs={"milvus_db_uri": "sqlite:///./milvus.db"}
    )

    # 2. 索引文档（V2 自动建立 SYNONYM 边）
    print("Indexing...")
    await rag.ainsert_file("document.pdf", doc_id="doc1")

    # 3. 查询（V3 多跳推理）
    print("Querying...")
    result = await rag.aquery(
        query="这个文档讲了什么？",
        param=QueryParam(
            mode="hybrid",
            enable_multi_hop=True,
            multi_hop_depth=2,
        )
    )

    print("回答：", result["response"])

if __name__ == "__main__":
    asyncio.run(main())
```

---

就这么简单。选一个场景，复制对应的代码，改改路径和参数就行。
