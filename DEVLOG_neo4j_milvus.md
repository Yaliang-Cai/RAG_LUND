# 开发日志：`neo4j-milvus` 分支

> 基线提交：`257f887 keep only path-based extract filtering`（main 分支）
> 日期：2026-03-24

---

## 零、Neo4j + Milvus 完整使用说明
先pip install pymilvus neo4j

### 0.1 配置方式：.env 文件 + 代码显式传参

#### .env 文件加载

项目根目录的 `.env` 文件在 `raganything.py` 启动时被自动加载。加载代码（`raganything/raganything.py`）：

```python
# 修复后（2026-03-27）：find_dotenv(usecwd=True) 从当前目录向上递归查找 .env
# 无论从哪个子目录运行，都能找到项目根目录的 .env
load_dotenv(dotenv_path=find_dotenv(usecwd=True) or ".env", override=False)
```

> **修复前的问题**：原来是 `load_dotenv(dotenv_path=".env", override=False)`，使用相对路径。
> 从 `rag-anything/` 子目录运行时找不到根目录的 `.env`，导致 Neo4j/Milvus 连接变量未加载。

`.env` 文件内容（项目根目录 `D:\HUAWEI\RAG_LUND\.env`）：

```bash
# 图数据库：Neo4j 连接凭证
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme
NEO4J_DATABASE=neo4j

# 向量数据库：Milvus Lite（本地单文件，无需启动服务）
MILVUS_URI=./milvus_lite.db
MILVUS_DB_NAME=lightrag

# 以下两行对 LocalRagService 无效（见下方说明），可删除
# LIGHTRAG_GRAPH_STORAGE=Neo4JStorage
# LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage
```

#### 重要：LIGHTRAG_GRAPH_STORAGE 的生效范围

`LIGHTRAG_GRAPH_STORAGE` / `LIGHTRAG_VECTOR_STORAGE` 这两个环境变量**只对 LightRAG API Server 生效**：

| 使用方式 | env var 生效？ | 后端选择方式 |
|---|:---:|---|
| `lightrag_server.py`（API Server） | ✅ | `api/config.py:336-340` 读取并传给构造函数 |
| `LocalRagService` / `RAGAnything` | ❌ | `local_rag.py` 的 `lightrag_kwargs` 显式传参 |
| 直接 `LightRAG(...)` | ❌ | 构造函数参数 |

`LightRAG.__post_init__`（`lightrag.py:495-507`）不读取这两个变量，只做连接参数校验。

#### LocalRagService 的后端激活方式

Neo4J 和 Milvus 通过 `local_rag.py` 的 `_build_rag()` 方法中的 `lightrag_kwargs` 显式激活：

```python
# rag-anything/raganything/services/local_rag.py: _build_rag() 方法
lightrag_kwargs={
    ...
    "graph_storage": "Neo4JStorage",        # 显式指定，不依赖 env var
    "vector_storage": "MilvusVectorDBStorage",
}
```

连接凭证（`NEO4J_URI` 等）仍从 `.env` 读取，由 LightRAG 内部的 `check_storage_env_vars()` 校验。

LightRAG 读取连接变量的代码位置：
- `lightrag/lightrag/kg/__init__.py` 第 56 行（Neo4j 必需：`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`）
- `lightrag/lightrag/kg/__init__.py` 第 75 行（Milvus 必需：`MILVUS_URI`）

---

### 0.2 功能开关速查表

所有功能开关分两种类型：

**A. LightRAG 实例初始化参数（影响 Indexing 和 Retrieval）**

在 `lightrag/lightrag/lightrag.py` 的 `LightRAG` dataclass（第 163-185 行）：

| 开关 | 默认值 | 作用阶段 | 说明 |
|------|--------|---------|------|
| `enable_entity_disambiguation` | `True` | Indexing | V1：实体消歧（name\|type 复合 ID） |
| `enable_synonym_linking` | `False` | Indexing | V2：同义词 SYNONYM 边构建 |
| `synonymy_threshold` | `0.8` | Indexing | V2：cosine 相似度阈值 |
| `synonymy_topk` | `100` | Indexing | V2：KNN 候选数量 |
| `synonymy_min_entity_len` | `2` | Indexing | V2：最短实体名（字符数） |

**启用 V2 的方法（仅改 lightrag.py 这一处）：**
```python
# lightrag/lightrag/lightrag.py 第 173 行
enable_synonym_linking: bool = field(default=True)  # False → True
```

或在代码中实例化时传入：
```python
rag = LightRAG(enable_synonym_linking=True, synonymy_threshold=0.8)
```

---

**B. QueryParam 参数（影响单次查询）**

在 `lightrag/lightrag/base.py` 的 `QueryParam` dataclass（第 207-219 行）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_multi_hop` | `False` | V3：启用 PPR 多跳推理 |
| `multi_hop_depth` | `2` | V3：BFS 子图提取深度（构建虚拟 chunk 图用） |
| `ppr_damping` | `0.5` | V3：PPR damping 因子 α |
| `ppr_top_k` | `50` | V3：PPR 返回的最高分 chunk 数量 |
| `passage_node_weight` | `0.05` | V3：chunk VDB 分数在 PPR seed 中的缩放系数 |

**启用 V3 的方法（查询时传入）：**
```python
from lightrag.base import QueryParam

result = await rag.aquery(
    query="你的问题",
    param=QueryParam(
        mode="hybrid",
        enable_multi_hop=True,      # V3 启用
        multi_hop_depth=2,
        ppr_top_k=50,
        passage_node_weight=0.05,
    )
)
```

**修改 V3 默认值的方法（如果要全局生效）：**
```python
# lightrag/lightrag/base.py 第 207-219 行，直接修改 field(default=...) 的值
enable_multi_hop: bool = True          # 改为 True 则所有查询默认启用 V3
ppr_top_k: int = 100                   # 改为更大值
```

---

### 0.3 完整初始化示例

```python
from lightrag import LightRAG
from lightrag.base import QueryParam

# 全功能初始化（.env 中已配置 Neo4j + Milvus）
rag = LightRAG(
    working_dir="./rag_storage",

    # V1 实体消歧（默认 True，保持默认即可）
    enable_entity_disambiguation=True,

    # V2 同义词边（需要手动开启）
    enable_synonym_linking=True,
    synonymy_threshold=0.8,
    synonymy_topk=100,
    synonymy_min_entity_len=2,
)

# 索引
await rag.ainsert_file("document.pdf", doc_id="doc1")

# 查询（V3 多跳推理）
result = await rag.aquery(
    query="问题",
    param=QueryParam(
        mode="hybrid",
        enable_multi_hop=True,
        ppr_top_k=50,
    )
)
print(result["response"])
```

---

### 0.4 消融实验 Baseline（完全回退到 main）

```python
# 所有增强开关关闭 → 与 main 分支物理路径 100% 一致
rag = LightRAG(
    working_dir="./baseline",
    enable_entity_disambiguation=False,  # V1 关
    enable_synonym_linking=False,        # V2 关（默认）
)

# 查询时不传 enable_multi_hop（默认 False，V3 关）
result = await rag.aquery(query="问题", param=QueryParam(mode="hybrid"))
```

同时在 `local_rag.py` 的 `_build_rag()` 中注释掉（或删除）storage 指定行，使用默认 NetworkX + NanoVectorDB：
```python
# local_rag.py: _build_rag() 的 lightrag_kwargs 中
# "graph_storage": "Neo4JStorage",         # 注释掉
# "vector_storage": "MilvusVectorDBStorage", # 注释掉
```

---

## 一、分支目标

在 LightRAG + rag-anything 代码库上实现四层阶梯式增强，通过 Feature Toggles 保证可独立开关、可消融对比：

| 版本 | 功能 | 开关 | 默认值 |
|------|------|------|--------|
| V0 | Neo4j + Milvus 存储后端 | `.env` 中 `LIGHTRAG_GRAPH_STORAGE`/`LIGHTRAG_VECTOR_STORAGE` | 默认 NetworkX + NanoVectorDB |
| V1 | Entity Disambiguation（实体消歧） | `enable_entity_disambiguation`（LightRAG 初始化参数） | `True` |
| V2 | Synonym Linking（同义词边） | `enable_synonym_linking`（LightRAG 初始化参数） | `False` |
| V3 | PPR Multi-hop Reasoning（多跳推理） | `enable_multi_hop`（QueryParam） | `False` |

**关键原则**：全部开关设为 `False` 时，代码物理执行路径与 main 分支 100% 一致，保证消融实验基准组的有效性。

---

## 二、算法讲解

### 2.1 V1：Entity Disambiguation（实体消歧）

#### 问题背景

LightRAG 原版用 `compute_mdhash_id(entity_name, prefix="ent-")` 生成实体的图节点 ID 和向量数据库 ID。这意味着"苹果"这个实体，无论它在文档中指代水果、公司还是手机型号，在知识图谱中都会被合并为同一个节点。

#### 算法原理

V1 在实体 ID 计算中加入实体类型（entity_type）作为区分维度，生成"复合 ID"（composite ID）：

```
entity_id = entity_name + "|" + entity_type
graph_node_id = entity_id
vdb_id = md5(entity_id + "ent-")
```

**举例：**
- `"苹果"（ORGANIZATION）` → 图节点 ID: `"苹果|ORGANIZATION"`，VDB ID: `md5("苹果|ORGANIZATIONent-")`
- `"苹果"（FOOD）` → 图节点 ID: `"苹果|FOOD"`，VDB ID: `md5("苹果|FOODent-")`

两个实体不再发生合并，知识图谱中保留为两个独立节点，各自持有各自的边和属性。

#### 回退保证

```python
def compute_entity_id(entity_name, entity_type="", enable_disambiguation=True):
    if enable_disambiguation and entity_type:
        return f"{entity_name}|{entity_type}"
    return entity_name  # 关闭时 == 原版
```

关闭 V1 时（`enable_disambiguation=False`）：
- `compute_entity_id("苹果", "FOOD", False)` → `"苹果"`（与原版完全一致）
- `compute_entity_vdb_id(...)` → `compute_mdhash_id("苹果", prefix="ent-")`（与原版完全一致）

---

### 2.2 V2：Synonym Linking（同义词边）—— 当前版本（HippoRAG2 对齐）

#### 问题背景

知识图谱中，同一概念可能以不同名称出现：
- "AI" 和 "人工智能"
- "Beijing" 和 "北京"
- "LLM" 和 "大语言模型"

这些同义实体在图中没有连接，单次图遍历无法跨越它们，导致多语言、多写法的文档检索覆盖率低。

#### 算法原理（当前版本）

**核心思路**：用已计算好的 entity embedding 向量，直接查找向量空间中的最近邻，将余弦相似度超过阈值的实体对连接为 SYNONYM 边。

**执行步骤：**

1. **批量获取所有实体的 embedding 向量**
   ```python
   # synonym_linking.py
   all_ids = await entities_vdb.get_all_entity_ids()
   all_vecs = await entities_vdb.get_vectors_by_ids(all_ids)
   # all_vecs: {entity_id: np.array([...], dtype=float32)}
   ```

2. **过滤过短实体**
   ```python
   # 排除长度 ≤ min_entity_len 的实体（避免标点/单字成为枢纽）
   # 用正则计算字母/数字/中文字符数
   import re
   char_count = len(re.sub(r'[^\w]', '', entity_name, flags=re.UNICODE))
   if char_count <= min_entity_len:
       skip
   ```

3. **逐实体 KNN 查询（向量查询，非文本查询）**
   ```python
   # 关键：传入预计算的 embedding 向量，而非文本
   neighbors = await entities_vdb.query(
       query="",                          # 文本为空
       query_embedding=entity_vec,        # 直接用向量
       top_k=synonymy_topk,               # 默认 100
   )
   ```

4. **过滤建边**
   ```python
   for neighbor in neighbors:
       if neighbor.id == entity_id:
           continue  # 跳过自身
       sim = neighbor.score  # cosine similarity
       if sim < synonymy_threshold:  # 默认 0.8
           break  # top-k 已按相似度降序，可提前终止
       if not await graph.has_edge(entity_id, neighbor.id):
           await graph.upsert_edge(entity_id, neighbor.id, {
               "weight": sim,
               "description": f"Synonym: {entity_name} ≈ {neighbor_name}",
               "edge_type": "SYNONYM",
               "source_id": "synonym_detection",
           })
   ```

#### 关键参数

| 参数 | 值 | 对应 HippoRAG2 |
|------|-----|----------------|
| `synonymy_threshold` | 0.8 | 0.8（对齐） |
| `synonymy_topk` | 100 | ~2047（差距 20x，受 VDB 接口限制） |
| `synonymy_min_entity_len` | 2 | `len(re.sub('[^A-Za-z0-9]', '', name)) > 2`（对齐） |
| 向量来源 | 预计算 embedding | 精确向量矩阵乘法（对齐方向，实现不同） |

#### 集成点

**`lightrag/lightrag/lightrag.py` 第 2026-2034 行**（`ainsert` 方法内）：

```python
# 在 merge_nodes_and_edges() 完成后执行
if self.enable_synonym_linking:
    from lightrag.synonym_linking import build_synonym_edges
    await build_synonym_edges(
        entities_vdb=self.entities_vdb,
        knowledge_graph_inst=self.chunk_entity_relation_graph,
        new_entity_ids=None,   # None = 全量；传入列表 = 增量（待实现）
        synonymy_threshold=self.synonymy_threshold,
        synonymy_topk=self.synonymy_topk,
        min_entity_len=self.synonymy_min_entity_len,
    )
```

`enable_synonym_linking=False` 时：`if` 不进入，零开销，零副作用。

---

### 2.3 V3：PPR Multi-hop Reasoning（多跳推理）—— 当前版本（HippoRAG2 对齐）

#### 问题背景

用户查询"A 公司 CEO 的母校在哪个城市？"时，知识图谱中的路径可能是：

```
[query] → entity: "A 公司" → edge: "CEO是" → entity: "张三"
        → edge: "毕业于" → entity: "北京大学"
        → edge: "位于" → entity: "北京"
```

传统的单跳检索只能找到 "A 公司" 直接相连的节点，无法沿链条传播到 "北京"。PPR（Personalized PageRank）通过在知识图谱上模拟随机游走，将查询相关度沿边传播，实现多跳推理。

#### 算法原理（当前版本：异构图 + 虚拟 chunk 节点 + 双信号 seed）

**核心思路**：在 retrieval 时构建一个包含 entity 节点和 chunk（文档块）节点的异构图，以 VDB 分数为初始权重运行 PPR，直接输出 chunk 排序（而非仅扩展实体候选）。

**步骤 1：构建 Seed 权重（双信号）**

```python
# operate.py: _ppr_rank_chunks() 函数（第 5178 行起）

# 信号 1：entity seed weights
# 来源：entity VDB 分数 + relation VDB 分数（取最大）
for entity in entity_vdb_results:
    entity_seeds[entity.id] = entity.score
for relation in relation_vdb_results:
    # relation 两端的 entity 都增加权重
    entity_seeds[relation.src_id] = max(entity_seeds.get(relation.src_id, 0), relation.score)
    entity_seeds[relation.tgt_id] = max(entity_seeds.get(relation.tgt_id, 0), relation.score)

# 信号 2：chunk seed weights
# 来源：chunk VDB 分数 × passage_node_weight（默认 0.05，HippoRAG2 参数）
for chunk in chunk_vdb_results:
    chunk_seeds[chunk.id] = chunk.score * passage_node_weight
```

**步骤 2：构建异构图**

```python
# 从 entity seeds 的 source_id 字段反向获取关联 chunk
# entity.source_id 存储了该实体来自哪些 chunk（多个 chunk_id 用逗号分隔）
for entity_id, data in entity_data.items():
    chunk_ids = data["source_id"].split(",")
    for chunk_id in chunk_ids:
        G.add_edge(entity_id, chunk_id, weight=1.0)  # entity-chunk 双向边

# entity-entity 边（来自知识图谱）
for edge in graph_edges:
    G.add_edge(edge.src, edge.tgt, weight=edge.weight)

# 虚拟 chunk 节点（仅存于内存，不持久化）
# chunk_id → chunk 文本内容（查询时实时构建，无需提前存储）
```

**步骤 3：运行 PPR**

```python
# ppr.py: personalized_pagerank()
def personalized_pagerank(G, seed_weights, damping=0.5, top_k=50):
    """
    PPR 公式：π = α × (A × π) + (1 - α) × p
    其中：
        α = damping（默认 0.5）
        A = 归一化邻接矩阵
        p = seed 权重向量（归一化到和为 1）
        π = 稳态分布（各节点的 PPR 分数）
    """
    personalization = {node: seed_weights.get(node, 0) for node in G.nodes()}
    ppr_scores = nx.pagerank(
        G,
        alpha=damping,          # 继续游走的概率（传播权重）
        personalization=personalization,
        max_iter=100,
        tol=1e-6,
    )
    # 从 PPR 分数中提取 chunk 节点的分数（排除 entity 节点）
    chunk_scores = {nid: score for nid, score in ppr_scores.items()
                    if nid in chunk_seed_ids or nid.startswith("chunk-")}
    return sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**步骤 4：PPR Chunks 合并（最高优先级）**

```python
# operate.py: _merge_all_chunks()（第 4582 行）
def _merge_all_chunks(vector_chunks, entity_chunks, relation_chunks, ppr_chunks=None):
    if ppr_chunks:
        # V3 路径：PPR chunks 优先，vector chunks 作为补充（去重）
        merged = list(ppr_chunks)
        seen = {c["id"] for c in ppr_chunks}
        for c in vector_chunks:
            if c["id"] not in seen:
                merged.append(c)
                seen.add(c["id"])
        return merged[:final_top_k]
    else:
        # 原版路径：entity + relation + vector chunks 按得分合并
        ...
```

#### PPR 数学直觉

PPR 可以理解为一个随机游走者：
- 以概率 `α`（damping）沿图边移动到邻居
- 以概率 `1 - α` 被"传送"回 seed 节点（按 seed 权重分布）

经过足够多轮迭代后，游走者在每个节点的停留概率就是 PPR 分数。与 query 相关的 entity（高 seed 权重）会将自己的分数通过边传播给相邻的 chunk，相邻 chunk 又传播给更远的节点。

**参数直觉：**
- `damping=0.5`：50% 概率继续游走，50% 回到 seed。较小的值使传播更保守（结果更集中在 seed 附近），较大的值使传播更广泛（多跳）。HippoRAG2 也用 0.5。
- `passage_node_weight=0.05`：chunk VDB 分数在 seed 中的权重缩放。值过大会让 PPR 完全由 chunk VDB 主导（退化为 VDB 排序），值过小则 chunk 节点在 PPR 初始化时权重太低，主要靠实体边传播。0.05 来自 HippoRAG2 原始参数。

---

### 2.4 V0：Neo4j + Milvus 存储后端

#### Neo4j（图存储）

**代码层面的变化**：移除了 `neo4j_impl.py` 中的 `pipmaster` 自动安装逻辑，改为 `try/except ImportError`，使其适配无网络的离线环境。

**Neo4j 的优化**：覆写了基类的 `get_subgraph_for_ppr()` 方法，用单条 Cypher 替代 BFS 循环：

```cypher
# lightrag/lightrag/kg/neo4j_impl.py（V3 子图提取，第 ~290 行）
MATCH path = (seed)-[*1..{max_depth}]-(neighbor)
WHERE seed.entity_id IN $seed_ids
RETURN DISTINCT neighbor.entity_id, properties(neighbor),
       startNode(r).entity_id, endNode(r).entity_id, properties(r)
```

#### Milvus（向量存储）

移除 `pipmaster` 逻辑，同 Neo4j。

Milvus Lite 以 `MILVUS_URI=./milvus_lite.db` 单文件方式运行，无需启动服务，适合离线部署。

---

## 三、各版本变更明细

### V0：Neo4j + Milvus 离线兼容

**`lightrag/lightrag/kg/neo4j_impl.py`**（+55 行 / -5 行）
- 移除 `pipmaster` 自动安装，改为 `try/except ImportError`
- 新增 `get_subgraph_for_ppr()` Cypher 实现（V3 优化）

**`lightrag/lightrag/kg/milvus_impl.py`**（+8 行 / -5 行）
- 移除 `pipmaster` 自动安装，改为 `try/except ImportError`

**`.env`**（新建）
- Neo4j 连接参数 + Milvus Lite 路径（`LIGHTRAG_GRAPH_STORAGE`/`LIGHTRAG_VECTOR_STORAGE` 仅对 API Server 有效，LocalRagService 不读取）

**`rag-anything/raganything/services/local_rag.py`**（`_build_rag()` 方法）
- `lightrag_kwargs` 中新增 `"graph_storage": "Neo4JStorage"` 和 `"vector_storage": "MilvusVectorDBStorage"`，显式激活新后端

---

### V1：Entity Disambiguation — 实体消歧

**开关位置**：`lightrag/lightrag/lightrag.py` 第 170 行
```python
enable_entity_disambiguation: bool = field(default=True)
```

**新增工厂函数**：`lightrag/lightrag/utils.py` 第 560-578 行

```python
def compute_entity_id(entity_name, entity_type="", enable_disambiguation=True):
    if enable_disambiguation and entity_type:
        return f"{entity_name}|{entity_type}"
    return entity_name

def compute_entity_vdb_id(entity_name, entity_type="", enable_disambiguation=True):
    composite = compute_entity_id(entity_name, entity_type, enable_disambiguation)
    return compute_mdhash_id(composite, prefix="ent-")
```

**替换的调用点**（20+ 处）：

`lightrag/lightrag/operate.py`：

| 函数 | 行号 | 变更 |
|------|------|------|
| `_update_entity_storage` | ~1825 | VDB ID + 图节点 ID 改用 composite |
| `_rebuild_single_relationship` | ~2225 | 新建端点节点使用 composite ID |
| `_merge_nodes_then_upsert` | ~2573 | 核心：图节点 ID 和 VDB ID 均改用 composite |
| `_merge_edges_then_upsert`（新建） | ~2915 | edge 端点不存在时创建的节点使用 composite |
| `_merge_edges_then_upsert`（更新） | ~3047 | 已有节点更新 VDB 时使用 composite |
| `merge_nodes_and_edges`（分组守卫） | ~3200 | 关键守卫：见下方 |
| `merge_nodes_and_edges`（edge remap） | ~3338 | 用 name→composite 映射重写 edge key |

**分组守卫**（消融实验关键保证）：

```python
# operate.py 第 ~3200 行
_disambig = global_config.get("enable_entity_disambiguation", True)
for maybe_nodes, maybe_edges in chunk_results:
    if _disambig:
        # V1 路径：按 composite key (name|type) 分组
        for entity_name, entities in maybe_nodes.items():
            for entity in entities:
                group_key = compute_entity_id(entity_name, entity.type, _disambig)
                all_nodes[group_key].append(entity)
    else:
        # 原版路径：批量 extend（物理路径与 main 100% 一致）
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)
```

`lightrag/lightrag/utils_graph.py`：6 处（`acreate_entity`, `_edit_entity_impl`, `_merge_entities_impl`, `get_entity_info`）

`lightrag/lightrag/lightrag.py`：2 处（`ainsert_custom_kg`, `adelete_by_doc_ids`）

`rag-anything/raganything/modalprocessors.py`：1 处（`_process_single_entity` 第 558 行）

`rag-anything/raganything/processor.py`：2 处（multimodal entity VDB ID 第 1198 行，knowledge graph upsert 第 1221 行）

---

### V2：Synonym Linking — 同义词链接（HippoRAG2 对齐版）

**开关位置**：`lightrag/lightrag/lightrag.py` 第 173-182 行

```python
enable_synonym_linking: bool = field(default=False)   # 第 173 行
synonymy_threshold: float = field(default=0.8)         # 第 176 行
synonymy_topk: int = field(default=100)                # 第 179 行
synonymy_min_entity_len: int = field(default=2)        # 第 182 行
```

**核心文件**：`lightrag/lightrag/synonym_linking.py`（152 行）

**集成点**：`lightrag/lightrag/lightrag.py` 第 2026-2034 行（`ainsert` 方法末段）

**主要变化（相比旧版）**：
- 旧版：`entities_vdb.query(description_text, top_k=10)` → 重新编码文本，近似 ANN
- 新版：`entities_vdb.query("", query_embedding=entity_vec, top_k=100)` → 直接用预计算向量，精确 KNN
- threshold 0.85 → 0.8
- topk 10 → 100
- 新增 `min_entity_len=2` 短实体过滤

---

### V3：PPR Multi-hop Reasoning — 多跳推理（HippoRAG2 对齐版）

**开关位置**：`lightrag/lightrag/base.py` 第 207-219 行（`QueryParam`）

```python
enable_multi_hop: bool = False          # 第 207 行
multi_hop_depth: int = 2                # 第 210 行
ppr_damping: float = 0.5               # 第 213 行
ppr_top_k: int = 50                    # 第 216 行
passage_node_weight: float = 0.05      # 第 219 行
```

**核心文件**：`lightrag/lightrag/ppr.py`（161 行）

**集成点**：

| 位置 | 行号 | 说明 |
|------|------|------|
| `operate.py: _perform_kg_search()` | ~4374 | V3 入口：调用 `_ppr_rank_chunks()` |
| `operate.py: _ppr_rank_chunks()` | 5178 | 核心：构建异构图 + 双信号 seed + 运行 PPR |
| `operate.py: _merge_all_chunks()` | 4582 | PPR chunks 最高优先级合并 |
| `operate.py: kg_query()` | ~5118 | 将 `ppr_chunks` 传入 `_merge_all_chunks()` |

**主要变化（相比旧版）**：
- 旧版：entity-only 图，单信号 seed，PPR 结果用于扩展 entity 候选，集成在 `_get_node_data()`
- 新版：entity + chunk 异构图，双信号 seed（entity VDB + chunk VDB × 0.05），PPR 直接输出 chunk 排序，集成在 `_perform_kg_search()` + `_merge_all_chunks()`

---

## 四、文件变更总览

### 修改的文件（9 个）

| 文件 | 增 / 删 | 涉及版本 | 说明 |
|------|---------|----------|------|
| `lightrag/lightrag/utils.py` | +22 / +1 注释 | V1 | 工厂函数 `compute_entity_id`, `compute_entity_vdb_id` |
| `lightrag/lightrag/base.py` | +62 | V3 | `get_subgraph_for_ppr` 默认实现 + QueryParam 扩展（5 个 V3 字段） |
| `lightrag/lightrag/lightrag.py` | +33 | V0/V1/V2 | Feature Toggles（5 个字段）+ synonym linking 集成 |
| `lightrag/lightrag/operate.py` | +270 / -57 | V1/V3 | 实体 ID 替换 + 分组守卫 + edge remap + `_ppr_rank_chunks()` + `_merge_all_chunks()` 扩展 |
| `lightrag/lightrag/utils_graph.py` | +35 / -10 | V1 | 图操作工具函数 entity ID 更新 |
| `lightrag/lightrag/kg/neo4j_impl.py` | +71 / -5 | V0/V3 | pipmaster 移除 + PPR 子图 Cypher 优化 |
| `lightrag/lightrag/kg/milvus_impl.py` | +8 / -5 | V0 | pipmaster 移除 |
| `rag-anything/raganything/modalprocessors.py` | +21 / -10 | V1 | 多模态处理器 entity ID 更新 |
| `rag-anything/raganything/processor.py` | +16 / -5 | V1 | 处理器 entity ID 更新 |
| `rag-anything/raganything/raganything.py` | +2 / -1 | V0 | `load_dotenv` 改用 `find_dotenv(usecwd=True)`，修复从子目录运行时找不到 `.env` 的问题 |
| `rag-anything/raganything/services/local_rag.py` | +2 | V0 | `_build_rag()` 的 `lightrag_kwargs` 中显式指定 `graph_storage`/`vector_storage` |

### 新增文件（3 个）

| 文件 | 行数 | 版本 | 说明 |
|------|------|------|------|
| `lightrag/lightrag/synonym_linking.py` | 152 | V2 | 同义词边构建（HippoRAG2 对齐版） |
| `lightrag/lightrag/ppr.py` | 161 | V3 | PPR 计算（异构图 + 双信号，HippoRAG2 对齐版） |
| `.env` | 14 | V0 | 环境变量配置（Neo4j + Milvus + 存储后端选择） |
| `rag-anything/raganything/constants.py`（新增字段） | +8 | V2 | V2 默认参数（DEFAULT_ENABLE_SYNONYM_LINKING 等） |

### 未修改的文件（确认）

| 文件 | 说明 |
|------|------|
| `lightrag/lightrag/api/config.py` | `DefaultRAGStorageConfig` 保持原始默认值不变（NetworkX + NanoVectorDB） |

---

## 五、Absolute Fallback 验证矩阵

### 开关组合矩阵

| 配置 | V1 disambig | V2 synonym | V3 multi_hop | 预期行为 |
|------|:-----------:|:----------:|:------------:|---------|
| **基准组（完全关闭）** | `False` | `False` | `False` | 与 main 100% 一致 |
| V1 only | `True` | `False` | `False` | composite key 分组，无 SYNONYM 边，无 PPR |
| V1+V2 | `True` | `True` | `False` | composite key + SYNONYM 边，无 PPR chunk 排序 |
| V1+V3 | `True` | `False` | `True` | composite key + PPR 多跳，无 SYNONYM 边 |
| V1+V2+V3 | `True` | `True` | `True` | 全功能：消歧 + 同义词 + PPR 协同 |

**V2 与 V3 协同效果**：V2 建立的 SYNONYM 边会增加图的连通性，PPR 在 SYNONYM 边上传播，可以跨越同义词边界召回相关 chunk。

### 基准组路径一致性保证（全关时）

| 代码路径 | 全关时的行为 | 与原版是否一致 |
|----------|-------------|:---:|
| 分组逻辑 `merge_nodes_and_edges` | 走 `all_nodes[entity_name].extend(entities)` | ✅ 一致 |
| Entity VDB ID 计算 | `compute_mdhash_id(entity_name, prefix="ent-")` | ✅ 一致 |
| Graph node ID | `entity_name`（无 `|type` 后缀） | ✅ 一致 |
| Edge key remapping | 跳过（`if _disambig:` 不进入） | ✅ 一致 |
| Synonym linking | 跳过（`if self.enable_synonym_linking:` 不进入） | ✅ 一致 |
| PPR expansion | 跳过（`if query_param.enable_multi_hop:` 不进入） | ✅ 一致 |
| 默认存储后端 | `NanoVectorDBStorage` / `NetworkXStorage` | ✅ 一致 |

---

## 六、与 HippoRAG2 的差距分析（更新至 HippoRAG2 对齐版）

### V2 对比（升级后）

| 维度 | HippoRAG2 | 我们（升级后） | 差距 |
|------|-----------|--------------|------|
| **向量来源** | embedding 矩阵（精确） | 预计算 embedding，VDB 向量查询（对齐✅） | 无本质差距 |
| **KNN 规模** | top-2047 | top-100（受 VDB 接口限制） | 20x 差距 |
| **阈值** | 0.8 | 0.8（对齐✅） | 无差距 |
| **短实体过滤** | `len > 2` | `min_entity_len=2`（对齐✅） | 无差距 |
| **批量 GPU KNN** | 矩阵乘法 | VDB 串行逐实体查询 | 性能差距（非算法差距） |

### V3 对比（升级后）

| 维度 | HippoRAG2 | 我们（升级后） | 差距 |
|------|-----------|--------------|------|
| **图节点类型** | Entity + Passage + Fact | Entity + virtual Chunk（对齐方向✅） | Fact 节点缺失 |
| **Seed 信号** | 双信号（entity + passage DPR × 0.05） | 双信号（entity VDB + chunk VDB × 0.05）（对齐✅） | 实现不同，方向一致 |
| **PPR 输出** | Passage 节点分数（直接排序文档） | Chunk 节点分数（直接排序文档）（对齐✅） | 无本质差距 |
| **PPR 图范围** | 全图 | 全图（entity source_id 反向映射）（对齐方向✅） | 实现不同，方向一致 |
| **Recognition Memory** | LLM fact reranking（核心创新） | 无 | 缺失关键机制 |
| **Fact 三元组节点** | 独立节点 | 无（LightRAG 架构限制） | 架构差距 |
| **双 embedding 指令** | query_to_fact + query_to_passage | 单一 embedding | 缺失 |

### 总结：差距矩阵（更新版）

| HippoRAG2 核心机制 | 我们是否实现 |
|:---|:---|
| Embedding-based synonym detection | ✅ 对齐（topK 差 20x） |
| SYNONYM 边创建 | ✅ 实现 |
| PPR 算法本身 | ✅ 实现（networkx vs igraph，性能差距） |
| 全图 PPR（含 chunk/passage 节点） | ✅ 对齐（via source_id 映射） |
| 双信号 seed weights | ✅ 对齐 |
| PPR 直接决定文档排序 | ✅ 对齐 |
| Recognition Memory（LLM fact reranking） | ❌ 缺失（最大差距） |
| Fact triple 独立节点 | ❌ 架构限制 |
| 批量矩阵 KNN | ❌ 性能差距（非算法差距） |

**当前结论**：升级后的 V2+V3 已基本对齐 HippoRAG2 的核心数据流。主要残余差距为 Recognition Memory（LLM 对候选三元组进行重排序过滤）这一认知心理学启发的机制，以及 Fact 独立节点的缺失（受 LightRAG 图架构限制）。

---

## 七、已知局限 & 待改进事项

1. **V2 增量模式未启用**：`build_synonym_edges(new_entity_ids=None)` 每次处理全部实体，对大图效率不高。后续可从 `merge_nodes_and_edges` 返回本批次新增 entity_id 列表传入，实现增量模式。

2. **V1 关闭后的 VDB metadata 冗余字段**：关闭 V1 时，VDB data dict 中多了 `"entity_id"` 字段（值等于 `entity_name`），对消融实验无影响，但字节不完全一致。

3. **Neo4j PPR Cypher 待测试**：`get_subgraph_for_ppr` 的 Cypher 查询需在真实 Neo4j 实例上验证语法正确性。基类的 BFS 默认实现可作为 fallback。

4. **Recognition Memory 缺失**：HippoRAG2 最核心的创新，当前无法在不引入专门 fact 存储的情况下实现。

5. **V2 topk 受 VDB 接口限制**：当前 VDB `query()` 接口限制 topk，无法做到 HippoRAG2 的 top-2047 批量矩阵乘法。

---

## 八、2026-03-30 补丁记录

### 远端同步

从 `origin/neo4j-milvus` 拉取了 21 个新 commit（`30e5043..69394db`），主要内容：
- SurGE 评估流水线增强（surge 评估模式、并发 ingest 优化）
- rerank 可观察性改进（`rerank_score_scope` 参数）
- 服务级弹性（resilience / callbacks / single-flight init）
- `DeletionResult.status` 增加 `"not_allowed"` 枚举值

合并过程只产生 **1 个冲突**（`operate.py: _merge_nodes_then_upsert`），双方独立修复了同一个 bug（见下文 Bug 1），解法如下：
- 保留远端的防御性 `isinstance` 检查（提取 `canonical_entity_name`）
- 保留本地的 `composite_id = entity_name` 直接赋值（避免远端条件分支的残余 edge case）

---

### Bug 1 — 实体图节点 ID 重复拼接 `entity_type`

**现象**：消歧模式开启时，图中节点 ID 为 `entity_name|entity_type|entity_type`（`|type` 重复）。

**根因**：`merge_nodes_and_edges` 在消歧路径下以 `compute_entity_id(name, type, True)` = `name|type` 作为 `all_nodes` 的 key，并将该 composite key 作为 `entity_name` 传入 `_merge_nodes_then_upsert`。函数内部又调用了一次 `compute_entity_id(entity_name, entity_type, _disambig)`，导致追加两次 `|type`。

**修复**（`operate.py: _merge_nodes_then_upsert` 第 2580 行附近）：
```python
# 从 nodes_data[0] 取回真正的 plain name
canonical_entity_name = entity_name
if nodes_data:
    first = nodes_data[0].get("entity_name")
    if isinstance(first, str) and first:
        canonical_entity_name = first

# entity_name 已经是正确的图节点 key，不再重复调用 compute_entity_id
composite_id = entity_name
```
所有内容/VDB 字段改用 `canonical_entity_name`（plain name），VDB hash 改为 `compute_entity_vdb_id(canonical_entity_name, entity_type, _disambig)`。

---

### Bug 2 — `delete_entity()` 无法定位消歧模式下的向量

**现象**：消歧模式开启时，`delete_entity(entity_name)` 计算的 VDB hash 为 `hash(name)`，而插入时存储的 hash 为 `hash(name|type)`，导致删除静默失败。图节点同样找不到（graph key 是 `name|type`，但只传了 `name`）。

**修复范围**（完整调用链）：

| 层 | 文件 | 变更 |
|----|------|------|
| 抽象接口 | `base.py` | `delete_entity(entity_name, entity_type="")` |
| 编排层 | `utils_graph.py:adelete_by_entity` | 新增 `entity_type=""` 参数；`node_key = compute_entity_id(name, type, _disambig)`；所有图操作改用 `node_key` |
| LightRAG 入口 | `lightrag.py` | `adelete_by_entity` / `delete_by_entity` 透传 `entity_type` |
| API | `document_routes.py:DeleteEntityRequest` | 新增可选 `entity_type: str = ""` 字段 |
| VDB 实现（6 个） | `milvus_impl`, `nano_vector_db_impl`, `faiss_impl`, `qdrant_impl`, `mongo_impl`, `chroma_impl` | 接受 `entity_type`，改用 `compute_entity_vdb_id(name, type, _disambig)` |
| Postgres 特殊处理 | `postgres_impl` | 消歧模式下 `WHERE entity_name=$2 AND entity_type=$3`，防止跨类型误删 |

向后兼容：`entity_type=""` 默认值 → 消歧关闭时行为与旧版完全一致。

---

### 设计收口 — `_is_milvus_lite()` 副作用缓存（问题 2）

**原设计缺陷**：`_milvus_uri` 字段在 `_create_milvus_client()` 中作为副作用设置，`_is_milvus_lite()` 依赖该字段，导致隐式时序依赖——未经 `initialize()` 直接调用 `_is_milvus_lite()` 会静默返回 `False`（`getattr` fallback 掩盖了错误）。

**产生原因**：`_get_milvus_connection_kwargs()` 命名为"构建 kwargs"，开发者认为在判断模式时调用它语义不清晰，于是引入 URI 缓存。但该函数实质上是纯函数（只读 env/config，幂等），完全可以多次调用。

**修复**（`milvus_impl.py`）：
```python
# 之前
def _is_milvus_lite(self) -> bool:
    uri = getattr(self, "_milvus_uri", None)
    if uri is None:
        return False
    return not self._uri_is_remote(uri)

# 之后
def _is_milvus_lite(self) -> bool:
    uri = self._get_milvus_connection_kwargs(include_db_name=False)["uri"]
    return not self._uri_is_remote(uri)
```
同时删除 `__post_init__` 中的 `self._milvus_uri: Optional[str] = None` 字段声明，以及 `_create_milvus_client()` 中的赋值行。

---

### 问题 3（已确认，不修改）— URI fallback `milvus.db` vs `milvus_lite.db`

远端和本地均使用 `milvus.db` 作为 fallback 文件名，与上游 `milvus_lite.db` 不同。这是 commit `4595513` 中有意做出的命名决策（更中性，不暴露实现细节），不是 bug。

**迁移注意**：旧安装如有 `milvus_lite.db` 文件，需手动重命名为 `milvus.db`（或通过 `MILVUS_URI` 环境变量显式指定路径）。
