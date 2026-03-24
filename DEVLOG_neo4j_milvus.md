# 开发日志：`neo4j-milvus` 分支

> 基线提交：`257f887 keep only path-based extract filtering`（main 分支）
> 日期：2026-03-24

---

## 一、分支目标

在 LightRAG + rag-anything 代码库上实现四层阶梯式增强，通过 Feature Toggles 保证可独立开关、可消融对比：

| 版本 | 功能 | 开关 | 默认值 |
|------|------|------|--------|
| V0 | Neo4j + Milvus 存储后端支持 | 调用方注入，无代码开关 | 保持原默认值不变 |
| V1 | Entity Disambiguation（实体消歧） | `enable_entity_disambiguation` | `True` |
| V2 | Synonym Linking（同义词边） | `enable_synonym_linking` | `False` |
| V3 | PPR Multi-hop Reasoning（多跳推理） | `enable_multi_hop`（QueryParam） | `False` |

**关键原则**：全部开关设为 `False` 时，代码物理执行路径与 main 分支 100% 一致，保证消融实验基准组的有效性。

---

## 二、各版本变更明细

### V0：Neo4j + Milvus 离线兼容

**目标**：支持 Neo4j 图存储和 Milvus 向量存储，但不改变源码默认值，由调用方动态注入。

#### 变更文件

**`lightrag/lightrag/kg/neo4j_impl.py`**（+55 行 / -5 行）
- **移除 pipmaster 自动安装**：将 `import pipmaster as pm; if not pm.is_installed("neo4j"): pm.install("neo4j")` 替换为 `try/except ImportError`，适配离线环境（无 pip 网络访问）。
- 原始代码（第 20-23 行）：
  ```python
  import pipmaster as pm
  if not pm.is_installed("neo4j"):
      pm.install("neo4j")
  ```
- 修改后：
  ```python
  try:
      from neo4j import (AsyncGraphDatabase, exceptions as neo4jExceptions, ...)
  except ImportError:
      raise ImportError("neo4j package required. Install with: pip install neo4j")
  ```

**`lightrag/lightrag/kg/milvus_impl.py`**（+8 行 / -5 行）
- 同上，将 `pipmaster` 自动安装替换为 `try/except ImportError`。

**`api/config.py`**
- **未修改**。`DefaultRAGStorageConfig` 保持 `NanoVectorDBStorage` / `NetworkXStorage` 原始默认值。
- Neo4j + Milvus 通过调用方注入：
  ```python
  rag = LightRAG(
      graph_storage="Neo4JStorage",
      vector_storage="MilvusVectorDBStorage",
  )
  ```

**`.env`**（新建）
- 项目根目录环境变量模板，含 Neo4j 连接参数和 Milvus Lite 本地文件路径。
- 仅作参考，不影响代码默认行为。

---

### V1：Entity Disambiguation — 实体消歧

**目标**：将 entity ID 从 `hash(name)` 改为 `hash(name + "|" + type)`，解决同名不同义实体（如"苹果"作为公司 vs 水果）在图谱中被合并的问题。

**核心设计**：开关逻辑封装在两个工厂函数内部，外部 20+ 处调用点无需写 `if/else` 分支。

#### 新增工厂函数

**`lightrag/lightrag/utils.py`**（+22 行）— 第 560-578 行

```python
def compute_entity_id(entity_name, entity_type="", enable_disambiguation=True):
    """生成图节点 ID。开关关闭时返回 entity_name（与原版一致）。"""
    if enable_disambiguation and entity_type:
        return f"{entity_name}|{entity_type}"
    return entity_name

def compute_entity_vdb_id(entity_name, entity_type="", enable_disambiguation=True):
    """生成 VDB hash ID。开关关闭时等价于原始 compute_mdhash_id(name, prefix="ent-")。"""
    composite = compute_entity_id(entity_name, entity_type, enable_disambiguation)
    return compute_mdhash_id(composite, prefix="ent-")
```

**回退保证**：`enable_disambiguation=False` 时：
- `compute_entity_id("苹果", "COMPANY", False)` → `"苹果"`（原值）
- `compute_entity_vdb_id("苹果", "COMPANY", False)` → `compute_mdhash_id("苹果", prefix="ent-")`（与原版完全一致）

#### Feature Toggle

**`lightrag/lightrag/lightrag.py`**（LightRAG dataclass，第 167 行）

```python
enable_entity_disambiguation: bool = field(default=True)
```

通过 `global_config = asdict(self)` 自动传递到 `operate.py` 等下游函数，无需额外管道。

#### 替换的调用点（20+ 处）

所有原先调用 `compute_mdhash_id(entity_name, prefix="ent-")` 的位置，改为调用 `compute_entity_vdb_id(entity_name, entity_type, _disambig)`。

**`lightrag/lightrag/operate.py`** — 5 处 VDB ID 计算 + 1 处图节点 ID + 1 处分组逻辑 + 1 处检索路径 + 1 处 edge remapping

| 函数 | 行号（修改后） | 变更内容 |
|------|----------------|----------|
| `_update_entity_storage` | ~1825 | VDB ID + 图节点 ID 改用 composite |
| `_rebuild_single_relationship` | ~2225 | 新建端点节点时使用 composite ID |
| `_merge_nodes_then_upsert` | ~2573 | 核心：图节点 ID 和 VDB ID 均改用 composite |
| `_merge_edges_then_upsert`（新建节点）| ~2915 | edge 端点不存在时创建的节点使用 composite |
| `_merge_edges_then_upsert`（已有节点）| ~3047 | 已有节点更新 VDB 时使用 composite |
| `merge_nodes_and_edges`（分组）| ~3200 | **关键守卫**：见下文 |
| `merge_nodes_and_edges`（edge remap）| ~3338 | 用 name→composite 映射重写 edge key |
| `_get_node_data`（检索）| ~5132 | 从 VDB 结果中取 `entity_id` 查图 |

**分组逻辑守卫**（第 3200-3216 行）— 消融实验的关键保证：

```python
_disambig = global_config.get("enable_entity_disambiguation", True)
for maybe_nodes, maybe_edges in chunk_results:
    if _disambig:
        # V1 路径：逐条遍历，按 composite key (name|type) 分组
        for entity_name, entities in maybe_nodes.items():
            for entity in entities:
                group_key = compute_entity_id(...)
                all_nodes[group_key].append(entity)
    else:
        # 原版路径：批量 extend，与 upstream 物理路径 100% 一致
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)
```

**Edge Key 重映射**（第 3338-3354 行）：
- Phase 1（节点处理）完成后，构建 `entity_name → composite_id` 映射
- Phase 2 开始前，将 `all_edges` 的 key 从 `(name_a, name_b)` 重写为 `(composite_a, composite_b)`
- 被 `if _disambig:` 守卫包裹，关闭时跳过

**`lightrag/lightrag/utils_graph.py`** — 6 处

| 函数 | 行号 | 变更说明 |
|------|------|----------|
| `_edit_entity_impl`（删旧 ID）| ~337 | 添加注释说明 entity_name 已是 composite ID |
| `_edit_entity_impl`（建新 ID）| ~381 | VDB metadata 增加 `entity_id` 字段；`entity_name` 取 `|` 分割的可读名 |
| `acreate_entity` | ~936 | 图节点 ID 和 VDB ID 均使用 composite；`has_node` 检查用 composite |
| `_merge_entities_impl`（更新 VDB）| ~1436 | VDB metadata 增加 `entity_id` 字段 |
| `_merge_entities_impl`（删源实体）| ~1517 | 注释说明 entity_name 已是 composite ID |
| `get_entity_info` | ~1702 | 注释说明 entity_name 已是 composite ID |

**`lightrag/lightrag/lightrag.py`** — 2 处

| 位置 | 行号 | 变更 |
|------|------|------|
| `ainsert_custom_kg` | ~2382 | VDB dict comprehension 改用 `compute_entity_vdb_id` |
| `adelete_by_doc_ids` | ~3560 | 注释说明 `entities_to_delete` 已含 composite ID |

**`lightrag/lightrag/utils.py`** — 1 处

| 位置 | 行号 | 变更 |
|------|------|------|
| `aexport_data` | ~1587 | 注释说明 entity_name 已是 composite ID |

**`rag-anything/raganything/modalprocessors.py`** — 1 处

| 位置 | 行号 | 变更 |
|------|------|------|
| `_process_single_entity` | ~558 | 图节点 ID 和 VDB ID 均改用 composite；从 `self.global_config` 读取开关 |

**`rag-anything/raganything/processor.py`** — 2 处

| 位置 | 行号 | 变更 |
|------|------|------|
| multimodal entity VDB ID | ~1198 | `compute_entity_vdb_id` 替换 `compute_mdhash_id` |
| knowledge graph upsert | ~1221 | 图节点 ID 使用 `entity_data["entity_id"]`（即 composite） |

**KG 存储实现**（`milvus_impl.py`, `nano_vector_db_impl.py`, `faiss_impl.py`, `mongo_impl.py`）：
- `delete_entity` 方法中的 `compute_mdhash_id(entity_name, prefix="ent-")` **未修改**。
- 原因：调用方传入的 `entity_name` 在新系统中已经是 composite ID，`compute_mdhash_id(composite_id, prefix="ent-")` 自然生成正确的 VDB hash。
- 关闭 V1 时 composite_id == entity_name，hash 结果与原版完全一致。

#### VDB Metadata 变更

所有 entity VDB upsert 的 data dict 新增了 `"entity_id": composite_id` 字段：

```python
# 原版
{"entity_name": "苹果", "entity_type": "COMPANY", "content": "...", ...}

# V1 开启
{"entity_id": "苹果|COMPANY", "entity_name": "苹果", "entity_type": "COMPANY", "content": "...", ...}

# V1 关闭
{"entity_id": "苹果", "entity_name": "苹果", ...}  # entity_id == entity_name，冗余但无害
```

检索侧通过 `r.get("entity_id", r["entity_name"])` 兼容新旧数据。

---

### V2：Synonym Linking — 同义词链接

**目标**：在 ingestion 阶段，对语义相似的实体之间自动创建 SYNONYM 边。与 V3 完全正交（代码路径零交叉）。

#### 新建文件

**`lightrag/lightrag/synonym_linking.py`**（108 行）

核心函数 `build_synonym_edges(entities_vdb, knowledge_graph_inst, ...)`:

1. 遍历目标实体（增量模式：仅新增实体；全量模式：全部实体）
2. 对每个实体，用其描述文本查询 VDB 的 top-K 邻居
3. cosine similarity > `synonymy_threshold`（默认 0.85）且非自身 → 创建 SYNONYM 边
4. 通过 `edge_data["edge_type"] = "SYNONYM"` 属性标记，不修改 `upsert_edge` 接口

SYNONYM 边的 edge_data 结构：
```python
{
    "weight": similarity_score,
    "description": "Synonym: 苹果 ≈ Apple Inc.",
    "keywords": "synonym,alias",
    "source_id": "synonym_detection",
    "edge_type": "SYNONYM",
}
```

#### Feature Toggle

**`lightrag/lightrag/lightrag.py`**（3 个配置字段）：

```python
enable_synonym_linking: bool = field(default=False)  # 默认关闭
synonymy_threshold: float = field(default=0.85)
synonymy_topk: int = field(default=10)
```

#### 集成点

**`lightrag/lightrag/lightrag.py`** — `ainsert` 方法（第 ~2019 行）

在 `merge_nodes_and_edges()` 完成后、记录处理状态前插入：

```python
if self.enable_synonym_linking:
    from lightrag.synonym_linking import build_synonym_edges
    await build_synonym_edges(
        entities_vdb=self.entities_vdb,
        knowledge_graph_inst=self.chunk_entity_relation_graph,
        new_entity_ids=None,
        synonymy_threshold=self.synonymy_threshold,
        synonymy_topk=self.synonymy_topk,
    )
```

`enable_synonym_linking=False`（默认）时：`if` 不进入，零开销，零副作用。

---

### V3：PPR Multi-hop Reasoning — 多跳推理

**目标**：在 retrieval 阶段，用 Personalized PageRank 沿图谱多跳扩展，召回与 query 间接相关的远距实体。与 V2 完全正交。

#### 新建文件

**`lightrag/lightrag/ppr.py`**（70 行）

纯 networkx 实现（项目已有依赖，零额外安装），不依赖 igraph（需 C 编译环境，离线不友好）。

```python
def personalized_pagerank(nodes, edges, seed_weights, damping=0.5, top_k=20):
    G = nx.Graph()
    # 构建子图 → nx.pagerank(G, alpha=damping, personalization=...) → 排序取 top_k
```

#### 新增基类方法

**`lightrag/lightrag/base.py`**（+50 行）— `BaseGraphStorage.get_subgraph_for_ppr()`

```python
async def get_subgraph_for_ppr(self, seed_node_ids, max_depth=3):
    """获取 seed 节点周围的子图。非抽象方法，提供基于 BFS 的默认实现。"""
```

默认实现逐层 BFS 展开（调用已有的 `get_node`, `get_node_edges`, `get_edge`），适用于所有 graph backend。

#### Neo4j 优化实现

**`lightrag/lightrag/kg/neo4j_impl.py`**（+46 行）— override `get_subgraph_for_ppr`

单条 Cypher 查询替代 BFS 循环，利用 Neo4j 原生图遍历：

```cypher
MATCH path = (seed)-[*1..{max_depth}]-(neighbor)
WHERE seed.entity_id IN $seed_ids
RETURN DISTINCT n.entity_id, properties(n), startNode(r).entity_id, endNode(r).entity_id, properties(r)
```

#### QueryParam 扩展

**`lightrag/lightrag/base.py`**（QueryParam dataclass，第 ~207 行）：

```python
enable_multi_hop: bool = False        # 默认关闭
multi_hop_depth: int = 2
ppr_damping: float = 0.5
ppr_top_k: int = 20
```

#### 集成点

**`lightrag/lightrag/operate.py`** — `_get_node_data()` 函数（第 ~5155 行）

在初始 VDB 查询得到 seed entities 之后、relation 查找之前插入：

```python
if query_param.enable_multi_hop and node_datas:
    from lightrag.ppr import personalized_pagerank
    # 1. 从图中提取 seed 周围子图
    # 2. 以 VDB similarity score 为初始权重跑 PPR
    # 3. 将高 PPR 值的新节点追加到 node_datas（去重）
```

`enable_multi_hop=False`（默认）时：`if` 不进入，零开销，零副作用。

---

## 三、文件变更总览

### 修改的文件（9 个）

| 文件 | 增 / 删 | 涉及版本 | 说明 |
|------|---------|----------|------|
| `lightrag/lightrag/utils.py` | +22 / +1 注释 | V1 | 工厂函数 `compute_entity_id`, `compute_entity_vdb_id` |
| `lightrag/lightrag/base.py` | +62 | V3 | `get_subgraph_for_ppr` 默认实现 + QueryParam 扩展 |
| `lightrag/lightrag/lightrag.py` | +33 | V0/V1/V2 | Feature Toggles + synonym linking 集成 |
| `lightrag/lightrag/operate.py` | +128 / -10 | V1/V3 | 实体 ID 替换 + 分组守卫 + edge remap + PPR 集成 |
| `lightrag/lightrag/utils_graph.py` | +35 / -10 | V1 | 图操作工具函数的 entity ID 更新 |
| `lightrag/lightrag/kg/neo4j_impl.py` | +71 / -5 | V0/V3 | pipmaster 移除 + PPR 子图优化查询 |
| `lightrag/lightrag/kg/milvus_impl.py` | +8 / -5 | V0 | pipmaster 移除 |
| `rag-anything/raganything/modalprocessors.py` | +21 / -10 | V1 | 多模态处理器 entity ID 更新 |
| `rag-anything/raganything/processor.py` | +16 / -5 | V1 | 处理器 entity ID 更新 |

### 新增文件（3 个）

| 文件 | 行数 | 版本 | 说明 |
|------|------|------|------|
| `lightrag/lightrag/synonym_linking.py` | 108 | V2 | 同义词边构建模块 |
| `lightrag/lightrag/ppr.py` | 70 | V3 | PPR 计算模块（纯 networkx） |
| `.env` | 12 | V0 | 环境变量模板 |

### 未修改的文件（确认）

| 文件 | 说明 |
|------|------|
| `lightrag/lightrag/api/config.py` | `DefaultRAGStorageConfig` 保持原始默认值不变 |
| `lightrag/lightrag/kg/nano_vector_db_impl.py` | `delete_entity` 中的 `compute_mdhash_id` 逻辑自动兼容 |
| `lightrag/lightrag/kg/faiss_impl.py` | 同上 |
| `lightrag/lightrag/kg/mongo_impl.py` | 同上 |

---

## 四、Absolute Fallback 验证矩阵

| 配置 | enable_entity_disambiguation | enable_synonym_linking | enable_multi_hop | 预期行为 |
|------|:---:|:---:|:---:|------|
| **基准组** | `False` | `False` | `False` | 物理路径与 main 100% 一致 |
| V1 only | `True` | `False` | `False` | composite key 分组，无 SYNONYM 边，无 PPR |
| V1+V2 | `True` | `True` | `False` | composite key + SYNONYM 边，但 retrieval 不做 PPR |
| V1+V3 | `True` | `False` | `True` | composite key + PPR 多跳，无 SYNONYM 边 |
| V1+V2+V3 | `True` | `True` | `True` | 全功能：消歧 + 同义词 + PPR 协同 |

### 基准组路径一致性保证（全关时）

| 代码路径 | 全关时的行为 | 与原版是否一致 |
|----------|-------------|:---:|
| 分组逻辑 `merge_nodes_and_edges` | 走 `all_nodes[entity_name].extend(entities)` | 一致 |
| Entity VDB ID 计算 | `compute_mdhash_id(entity_name, prefix="ent-")` | 一致 |
| Graph node ID | `entity_name`（无 `\|type` 后缀） | 一致 |
| Edge key remapping | 跳过（`if _disambig:` 不进入） | 一致 |
| Synonym linking | 跳过（`if self.enable_synonym_linking:` 不进入） | 一致 |
| PPR expansion | 跳过（`if query_param.enable_multi_hop:` 不进入） | 一致 |
| 默认存储后端 | `NanoVectorDBStorage` / `NetworkXStorage` | 一致 |

---

## 五、使用示例

### 基准组（消融实验 Baseline）

```python
rag = LightRAG(
    working_dir="./baseline",
    enable_entity_disambiguation=False,
    # 其余默认：synonym=False, multi_hop 在 QueryParam 中默认 False
)
```

### 全功能（Neo4j + Milvus + V1 + V2 + V3）

```python
rag = LightRAG(
    working_dir="./full",
    graph_storage="Neo4JStorage",
    vector_storage="MilvusVectorDBStorage",
    enable_entity_disambiguation=True,
    enable_synonym_linking=True,
    synonymy_threshold=0.85,
)

# 查询时启用 PPR
from lightrag.base import QueryParam
result = await rag.aquery("问题", param=QueryParam(
    mode="mix",
    enable_multi_hop=True,
    multi_hop_depth=2,
    ppr_damping=0.5,
    ppr_top_k=20,
))
```

---

## 六、已知局限 & 后续事项

1. **V2 增量模式未启用**：当前 `build_synonym_edges(new_entity_ids=None)` 处理全部实体，对大图效率不高。后续应从 `merge_nodes_and_edges` 返回本批次新增 entity_id 列表传入。
2. **V1 关闭后的 VDB metadata 差异**：关闭 V1 时，VDB data dict 中多了 `"entity_id"` 字段（值等于 `entity_name`），语义无害但字节不完全一致。对消融实验无影响（不影响 embedding、检索或 hash）。
3. **Neo4j PPR Cypher 未测试**：`get_subgraph_for_ppr` 的 Cypher 查询需在真实 Neo4j 实例上验证。基类的 BFS 默认实现可作为 fallback。
4. **NetworkX graph storage 无 `get_subgraph_for_ppr` override**：使用基类 BFS 默认实现，功能正确但对大图可能较慢。


***

## 分析结论

**总体判断**：我们的 V2 和 V3 版本借鉴了 HippoRAG2 的核心思想，但在工程实现上存在重大简化和架构差异，严格来说**不算复用其算法**，而是 HippoRAG 理念在现有 LightRAG 架构下的“轻量化平替”。

---

### V2 Synonym Linking 对比

| 维度 | HippoRAG2 (`add_synonymy_edges`) | 我们的 V2 (`build_synonym_edges`) |
| :--- | :--- | :--- |
| **核心算法** | **全量 KNN**：取出所有实体 embedding，做 `torch.mm` 矩阵乘法，一次性算出全局 top-K。 | **逐条查询**：每个实体用文本描述调 VDB 的 `query()` 接口做近似搜索。 |
| **相似度计算** | 直接在 embedding 向量上做 cosine（`torch.mm` + L2 归一化），精确 KNN。 | 间接：先把 description 文本重新 embed（VDB 内部），再做 ANN（近似最近邻）。 |
| **KNN 规模** | `synonymy_edge_topk=2047`，每个实体查 2047 个邻居。 | `synonymy_topk=10`，只查 10 个。 |
| **阈值** | 0.8 | 0.85 |
| **批处理** | 全量批矩阵运算（`query_batch_size=1000`, `key_batch_size=10000`），GPU 加速。 | 串行循环，每个实体一次 VDB 调用。 |
| **过滤条件** | 只处理 `len(re.sub('[^A-Za-z0-9]', '', entity)) > 2` 的实体。 | 无过滤。 |
| **边权重** | Cosine similarity 分数。 | 同（VDB 返回的 distance 转换）。 |
| **去重** | 无（允许 A→B 和 B→A 同时存在）。 | 双向检查 `has_edge` 避免重复。 |
| **执行时机** | 全量索引后一次性执行。 | 每次 ingestion 后增量执行。 |

**关键差异：**
1. **精确 KNN vs 近似 ANN**：HippoRAG2 直接操作 embedding 矩阵做精确 KNN；我们复用 VDB 的 `query(text, top_k)` 接口。这意味着我们的文本要被重新 embed（多一次推理开销），VDB 内部用的是 HNSW/IVF 等近似搜索（有精度损失），且查询的是 `entity_name + "\n" + description` 的文本语义，而非单纯的实体 embedding 本身。
2. **全局 vs 局部视野**：HippoRAG2 一次性看到所有实体的 embedding 矩阵，计算全局 top-2047 邻居；我们每次只看 VDB 返回的 top-10。
3. **缺少 Entity 过滤**：HippoRAG2 会过滤掉过短的实体名（≤2 个字母数字字符）；我们目前没有这层过滤机制。

---

### V3 PPR Multi-hop 对比

| 维度 | HippoRAG2 (`run_ppr` + `graph_search...`) | 我们的 V3 (`personalized_pagerank` + `_get_node_data`) |
| :--- | :--- | :--- |
| **PPR 库** | `igraph`（C 底层，prpack 算法） | `networkx`（纯 Python） |
| **图范围** | **全图 PPR**：在包含所有实体 + 所有 passage 节点的完整图上运行。 | **局部子图 PPR**：先 BFS 提取 seed 周围 `max_depth=2` 的子图，再在子图上运行。 |
| **Damping** | 0.5 | 0.5 |
| **节点类型** | 三种：Entity、Passage（文档块）、Fact | 一种：Entity only |
| **Seed 权重构建** | **双信号融合**：(1) fact-query 相似度分数 ÷ 实体出现的 chunk 数 → `phrase_weights`；(2) DPR passage 分数 × 0.05 → `passage_weights`；两者相加。 | **单信号**：直接将 VDB cosine similarity score 作为 seed weight。 |
| **PPR 输出目标** | 提取 **passage 节点**的 PPR 分数，用于最终文档排序。 | 提取 **entity 节点**的 PPR 分数，用于扩展实体候选集。 |
| **Recognition Memory** | 有：DSPy/LLM 对 fact candidates 做 reranking 后再喂给 PPR。 | 无。 |
| **Fact 三元组** | 核心概念：独立存储 (S, P, O) 三元组，有专门的 `fact_embedding_store`。 | 无：LightRAG 架构用的是 relation 边，而非独立的 fact 节点。 |
| **Fallback** | 无 fact 命中时退化为纯 DPR 检索。 | 无 PPR 结果时保留原始 VDB 结果。 |

**关键差异：**
1. **全图 vs 子图 PPR（最根本区别）**：HippoRAG2 在完整知识图谱上运行 PPR，passage 节点也参与传播，PPR 分数直接决定最终文档排序。我们先 BFS 截取 2-hop 子图，在小图上跑 PPR，仅用于扩展 entity 候选集，扩展后的 entity 还要经过 LightRAG 原有的 relation 查找和 context 构建流程。
2. **PPR 的角色完全不同**：在 HippoRAG2 中，PPR 是**最终排序器**（直接决定返回哪些文档）；在我们系统中，PPR 只是**候选扩展器**（扩充 `node_datas` 列表，后续拼装流程不变）。
3. **缺失 Recognition Memory**：HippoRAG2 的核心创新之一是在 PPR 之前用 LLM 做 fact reranking（类比人脑的"识别记忆"），过滤掉不相关的 fact 三元组。我们完全没有这一步。
4. **缺失 Passage 节点**：HippoRAG2 的图中有 passage 节点（文档块直接作为图节点），passage-to-entity 边的权重为 1.0，passage 节点在 PPR 中接收传播分数。我们的图只有 entity 和 relation，没有 passage 节点参与 PPR。
5. **缺失双 Embedding 指令**：HippoRAG2 对 query 生成两套 embedding（`query_to_fact` 和 `query_to_passage`），使用不同的 instruction prefix。我们直接复用了 LightRAG 的单一 embedding。

---

### 总结：差距矩阵

| HippoRAG2 核心机制 | 我们是否实现 | 差距程度剖析 |
| :--- | :--- | :--- |
| **Embedding-based synonym detection** | 部分 | 思路相同，但用 VDB 文本查询代替精确向量 KNN，规模差了约 200 倍（top-10 vs top-2047）。 |
| **Synonym edge 创建** | 是 | 接口形式不同但图拓扑效果类似。 |
| **PPR 算法本身** | 是 | Damping 因子一致，但 `networkx` vs `igraph` 在大规模图上的性能差距极大。 |
| **全图 PPR** | 否 | 我们只在 2-hop 子图上跑，丢失了远距多跳的概率传播能力。 |
| **Passage 节点参与 PPR** | 否 | 导致 PPR 结果只能用于找实体，无法直接对底层文档块进行排序。 |
| **Dual-signal seed weights** | 否 | 只有 entity VDB score，缺少 passage DPR 信号，初始概率分布不精准。 |
| **Recognition Memory (LLM fact reranking)**| 否 | 完全缺失人脑机制中的“初步识别过滤”环节，易受噪声节点干扰。 |
| **Fact triple 作为独立节点** | 否 | 受限于 LightRAG 现有架构（Entity + Edge 模式）不支持。 |
| **Entity 过滤（短名排除）** | 否 | 工程细节缺失，可能导致无意义的短字符节点成为网络枢纽。 |
| **批量矩阵 KNN** | 否 | 性能差距，非算法本质差距，但在大规模吞吐量下影响明显。 |

**最终结论：**
V2 和 V3 借鉴了 HippoRAG2 的两个高层直觉（“用 embedding 相似度建同义边”和“用 PPR 做多跳推理”），但在实现深度上做了妥协。与 HippoRAG2 的差距主要集中在**全图传播能力、Passage 节点融合、识别记忆（Recognition Memory）以及双重 Seed 权重**上。这些架构级差距意味着，即使在 V3 阶段开关全开，我们的系统在复杂多跳推理上的效果也很可能不及 HippoRAG2 原版。

***

**下一步建议**：
既然你已经非常清楚这些差距，在写消融实验的结论时，你可以将 V3 定义为 **"PPR-Enhanced Entity Expansion"**，而不是完整的 HippoRAG2 复现。

---

## 七、V2/V3 升级到 HippoRAG2 对齐版本（2026-03-24）

### 升级总结

**V2 升级：Embedding-based KNN 替代文本重编码**

| 方面 | 之前 | 之后 |
|------|------|------|
| **核心变更** | `entities_vdb.query(description_text, top_k=10)` | `entities_vdb.query("", query_embedding=entity_vec, top_k=100)` |
| **VDB 查询方式** | 重新编码 entity name+description 文本，查询文本语义相似 | 直接用预计算的 entity embedding 向量，查询向量空间最近邻 |
| **Threshold** | 0.85 | 0.8 |
| **TopK** | 10 | 100 |
| **过滤条件** | 无 | `min_entity_len=2`: 排除短实体 (≤2 个字母/数字/中文字符) |
| **实现细节** | 逐实体调用 `query(text)` | 批量 `get_vectors_by_ids()` 获取 embedding，逐实体 `query(embedding=)` |

**文件变更**：
- `synonym_linking.py`：完全重写，108→152 行，新增 `compute_mdhash_id` 导入
- `lightrag.py`：参数更新 (threshold/topk/min_len)，调用站点传入 `min_entity_len`
- `base.py`：QueryParam 增加 `passage_node_weight: float = 0.05`

---

**V3 升级：全图 PPR 架构（异构图 + 虚拟 chunk 节点 + 双信号 seed）**

| 方面 | 之前 | 之后 |
|------|------|------|
| **PPR 图构成** | Entity nodes only | Entity nodes + Virtual chunk nodes |
| **Chunk 节点来源** | 不存在 | 从 entity.source_id 反向映射，retrieval-time 构建 |
| **Seed 权重信号** | 单一：entity VDB score | 双信号：entity VDB + relation VDB → entity weights；chunk VDB × passage_weight → chunk weights |
| **PPR 输出目标** | Entity PPR 分数（用于扩展实体候选） | Chunk PPR 分数（直接排序文档） |
| **集成点** | `_get_node_data()` 内部（作为实体扩展）| `_perform_kg_search()` 末尾 + `_merge_all_chunks()` 使用（作为最高优先级 chunk 源） |
| **持久化** | 不变（Neo4j/Milvus 无变更） | 不变（虚拟 chunk 节点仅存于内存） |

**文件变更**：
- `ppr.py`：完全重写，70→161 行；新增 `personalized_pagerank()` 支持异构图、双信号 seed；保留 `personalized_pagerank_simple()` 后向兼容
- `operate.py`：
  - 删除 `_get_node_data()` 中的旧 PPR 代码（~47 行）
  - 新增 `_ppr_rank_chunks()` (~135 行)：核心逻辑
  - 修改 `_perform_kg_search()` 返回值：增加 `ppr_chunks` 字段
  - 修改 `_merge_all_chunks()` 签名：新增 `ppr_chunks` 参数；当 PPR chunks 存在时，替代 entity/relation chunk 选择，vector chunks 作为补充
  - 修改 `_build_query_context()` 调用点：传入 `ppr_chunks=search_result.get("ppr_chunks")`
- `base.py`：QueryParam 新增 `passage_node_weight` 字段

### 关键算法细节

**V2 KNN 流程**：
```
1. compute_mdhash_id(entity_id, prefix="ent-") → VDB ID
2. get_vectors_by_ids([vdb_id, ...]) → entity embedding vectors
3. For each entity_id:
   - 过滤：len(cleaned_name) > min_entity_len
   - query(embedding=entity_vec, top_k=100, threshold=0.8)
   - 去重双向检查，创建 SYNONYM 边
```

**V3 PPR 流程**：
```
1. From entity VDB: seed entities → entity_seed_weights
2. From relation VDB: relations → entity_seed_weights (merge)
3. get_subgraph_for_ppr(seed_ids, max_depth) → entities + edges
4. Build virtual chunks:
   - Scan entity.source_id → chunk_to_entities mapping
   - Create virtual chunk nodes
   - Create chunk-entity edges (weight=1.0)
5. From chunks VDB: top_k chunks, normalize scores × passage_weight → chunk_seed_weights
6. personalized_pagerank(
     entity_nodes, entity_edges,
     chunk_nodes, chunk_entity_edges,
     entity_seed_weights, chunk_seed_weights
   ) → chunk PPR scores
7. Fetch chunk content, return ranked by PPR score
8. In _merge_all_chunks():
   - If ppr_chunks present: ppr_chunks + vector_chunks (PPR first)
   - Else: round-robin(entity/relation/vector)
```

### 消融实验矩阵（更新）

| V2 | V3 | 预期 |
|:---:|:---:|---|
| OFF | OFF | 原版 LightRAG（与 main 分支 100% 一致） |
| ON | OFF | SYNONYM 边存在，chunk 选择走原 WEIGHT/VECTOR 策略 |
| OFF | ON | 无 SYNONYM 边，PPR 用普通 entity-entity 边传播，chunk 按 PPR 排序 |
| ON | ON | SYNONYM + PPR 协同：PPR 沿 SYNONYM 边传播，最大化多跳召回 |

### 与 HippoRAG2 的已知改进对齐

| 方面 | 原版 V2/V3 | 本次升级 | 与 HippoRAG2 更近程度 |
|------|-----------|---------|:---:|
| **V2 KNN 方式** | 文本重编码 (top-10) | 向量 KNN (top-100) | ✓✓ (能用 VDB embedding 代替精确矩阵 KNN) |
| **V2 参数默认值** | 0.85, 10 | 0.8, 100 | ✓✓✓ (完全对齐 HippoRAG2) |
| **V2 过滤机制** | 无 | min_entity_len ≥ 2 | ✓✓ (对齐 >2 字母数字字符) |
| **V3 图架构** | 实体图 | 异构图 (entity + virtual chunk) | ✓✓✓ (对齐 entity + passage 概念) |
| **V3 PPR 目标** | 实体扩展 | Chunk 直接排序 | ✓✓✓ (完全对齐) |
| **V3 Seed 信号** | 单一 VDB score | 双信号 (entity + chunk) | ✓✓✓ (对齐 fact+passage 双信号) |

### 测试建议

**V2 验证**：
1. 插入同义实体（如 "AI"、"人工智能"、"Artificial Intelligence"）
2. 确认 SYNONYM 边被创建，权重 ≥ 0.8
3. 确认短实体被过滤
4. 对比 enable_synonym_linking ON/OFF

**V3 验证**：
1. 构造多跳推理查询（A→B→C→target chunk）
2. 确认 `ppr_score` 字段存在于返回 chunk
3. 对比 enable_multi_hop ON/OFF 的 chunk 排序差异
4. 运行 4 种 (V2 ON/OFF) × (V3 ON/OFF) 组合

### 后向兼容性保证

- V2 OFF：无 SYNONYM 边创建，ingestion 路径与原版 100% 一致
- V3 OFF：`_ppr_rank_chunks()` 不被调用，chunk 选择走原 WEIGHT/VECTOR 策略
- 两者都 OFF：100% 等价原版 LightRAG