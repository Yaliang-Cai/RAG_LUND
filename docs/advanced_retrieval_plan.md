# Advanced Retrieval Features Plan

## Context

当前 LightRAG 的检索管线（4-stage: Search → Truncate → Merge → BuildContext）功能朴素：
- VDB 查询用原始 query embedding，对模糊/抽象查询召回率差
- 多源 chunk 合并用 round-robin 交替取，无信息融合
- query mode 需手动选择，无自动路由
- 图上推理仅有 V3 PPR，缺少社区级全局视角

目标：添加 4 个正交、可独立开关的检索增强功能，兼容所有数据库后端。

---

## 实施顺序与依赖

```
Phase 1: RRF        — 零 LLM 调用，零新存储，纯算法替换
Phase 2: HyDE       — 1 LLM 调用，1 个新 prompt，钩入 embedding 计算
Phase 3: Query Router — 1 LLM 调用，1 个新 prompt，钩入 kg_query 入口
Phase 4: Communities — 新存储 + 摄取时社区检测 + 查询时检索（最大范围）
```

四者完全无依赖，可任意组合启用。

---

## Phase 1: RRF (Reciprocal Rank Fusion)

### 目标
替换 `_merge_all_chunks()` 中的 round-robin 合并为 RRF 算分排序。

### 改动

**`base.py` — QueryParam 新增字段：**
```python
rrf_k: int = int(os.getenv("RRF_K", "60"))
```

**`operate.py` — 新增函数 + 修改合并逻辑：**

1. 新增 `_rrf_merge(ranking_lists: list[list[dict]], k: int = 60) -> list[dict]`
   - 纯函数（非 async），O(n)
   - 每个 chunk 按 `chunk_id` 去重，计算 `rrf_score = Σ 1/(k + rank_i)`
   - 返回按 `rrf_score` 降序排列的 chunk list，附带 `rrf_score` 字段

2. 修改 `_merge_all_chunks()` (line 4694-4783)
   - 将 round-robin 循环替换为：收集 `[vector_chunks, entity_chunks, relation_chunks]` → `_rrf_merge()`
   - PPR 路径（line 4655-4693）不变——PPR 已有图结构信号，RRF 仅用于非 PPR 路径
   - 单一来源时直接返回，无需融合

### 与现有功能交互
- CrossEncoder reranking（Stage 4）在 RRF 之后执行，两者自然组合
- PPR 启用时绕过 RRF，互不干扰

---

## Phase 2: HyDE (Hypothetical Document Embeddings)

### 目标
LLM 生成假设性回答文档 → 用其 embedding 替代原始 query embedding → 提升 VDB 召回率。

### 改动

**`base.py` — QueryParam 新增字段：**
```python
enable_hyde: bool = False
```

**`prompt.py` — 新增 prompt：**
```python
PROMPTS["hyde_generation"] = """Given the following question, write a short passage (1-2 paragraphs) that would directly answer this question. Write as if it were an excerpt from an authoritative document. Do not include any preamble.

Question: {query}

Passage:"""
```

**`operate.py` — 新增函数 + 修改 embedding 计算：**

1. 新增 `async def _generate_hyde_embedding(query, global_config, embedding_func, hashing_kv=None) -> list[float]`
   - LLM 生成假设回答（用 `handle_cache` 缓存，`cache_type="hyde"`）
   - 对假设回答调用 `embedding_func` 得到 embedding
   - 返回 embedding vector

2. 修改 `_perform_kg_search()` (line 4308-4320)
   - 在 `query_embedding = await actual_embedding_func([query])` 之后：
   - 如果 `query_param.enable_hyde`，调用 `_generate_hyde_embedding()` 替换 `query_embedding`
   - 此 embedding 自动流入所有下游 VDB `.query(query_embedding=...)` 调用

### 与现有功能交互
- 不改 BaseVectorStorage 接口（所有后端已支持 `query_embedding=`）
- CrossEncoder reranking 用原始 query 文本（非 embedding），不受影响
- PPR seed weights 来自 VDB scores → HyDE 改善 VDB scores → PPR 间接受益

---

## Phase 3: Query Router

### 目标
LLM 自动分类 query 意图 → 选择最优 mode + 参数组合。

### 改动

**`base.py` — QueryParam 新增字段：**
```python
enable_query_router: bool = False
```

**`prompt.py` — 新增 prompt：**
```python
PROMPTS["query_router"] = """You are a query routing expert. Given a user query, determine the optimal retrieval strategy.

Available modes:
- "local": specific entities/properties ("who is X?", "what is X?")
- "global": broad themes/trends ("how does X relate to Y?")
- "hybrid": both specific details and broader context
- "mix": complex questions needing both KG and vector search

Output JSON only:
{{"mode": "local|global|hybrid|mix", "enable_multi_hop": true/false, "top_k_factor": 1.0}}

Query: {query}
Output:"""
```

**`operate.py` — 新增函数 + 修改 kg_query 入口：**

1. 新增 `async def _route_query(query, query_param, global_config, hashing_kv=None) -> QueryParam`
   - LLM 调用（`handle_cache`，`cache_type="router"`）
   - 解析 JSON → 创建 query_param 副本（`dataclasses.replace()`）覆盖 mode、enable_multi_hop、top_k
   - 解析失败时返回原始 query_param（静默降级）

2. 修改 `kg_query()` (line 3909 之前插入)
   - `if query_param.enable_query_router: query_param = await _route_query(...)`
   - 在 keyword extraction 之前执行，因为 mode 决定了 keyword 的使用方式

### 与现有功能交互
- Router 可以同时决定是否启用 multi_hop (V3)
- 不影响 HyDE（它们独立：HyDE 改 embedding，Router 改 mode）
- 用户显式设置 mode 时（disable router），行为完全不变

---

## Phase 4: Graph Community Retrieval

### 目标
摄取时运行社区检测 → 为每个社区生成 LLM 摘要 → 查询时检索相关社区摘要作为额外上下文。

### 改动

**`namespace.py` — 新增命名空间：**
```python
KV_STORE_COMMUNITY_SUMMARIES = "community_summaries"
VECTOR_STORE_COMMUNITIES = "communities"
```

**`base.py` — QueryParam 新增字段：**
```python
enable_community_retrieval: bool = False
community_top_k: int = int(os.getenv("COMMUNITY_TOP_K", "3"))
```

**`lightrag.py` — LightRAG dataclass 新增：**
```python
enable_community_detection: bool = False  # 摄取时开关
```
新增存储实例（与其他 KV/VDB 并列初始化）：
- `self.community_summaries: BaseKVStorage`
- `self.communities_vdb: BaseVectorStorage`

在 `ainsert()` 末尾（graph 构建完成后）调用社区检测。

**`prompt.py` — 新增 prompt：**
```python
PROMPTS["community_summary"] = """Summarize the following group of related entities and their relationships into a coherent paragraph.

Entities:
{entities}

Relationships:
{relationships}

Write a concise summary (2-4 sentences) capturing the main theme and key connections.

Summary:"""
```

**新文件 `community_detection.py`：**（类似 `synonym_linking.py` 模式）

1. `async def detect_and_summarize_communities(knowledge_graph_inst, community_kv, communities_vdb, global_config, ...)`
   - **DB-agnostic方法**：调用 `get_all_nodes()` + `get_all_edges()` 构建本地 NetworkX 图
   - 运行 `networkx.community.louvain_communities()` 或 `greedyModularityCommunities`
   - 过滤小社区（< 3 个实体）
   - 对每个社区：收集实体描述 + 关系描述 → LLM 生成摘要 → 存入 KV + embed 入 VDB
   - 支持增量：对比已存社区 ID，仅处理新增/变化的社区

2. 查询时函数在 `operate.py`：
   `async def _retrieve_community_context(query, communities_vdb, community_kv, query_param, query_embedding=None) -> list[dict]`
   - VDB 查询找到 top-k 相关社区 → KV 获取完整摘要
   - 返回社区摘要 list

**`operate.py` 修改：**
- `_build_query_context()` 中 Stage 1 之后、Stage 4 之前：调用 `_retrieve_community_context()`
- `_build_context_str()` 中：在 entities + relations 之后、chunks 之前插入 community summaries 段

### 与现有功能交互
- 摄取时运行（与 V2 synonym_linking 同级），查询时仅 VDB 查询（快速）
- HyDE embedding 可传给 `communities_vdb.query(query_embedding=)` → 间接受益
- RRF 不受影响（community 是 context 段落，不是 chunk 排序列表）
- Router 未来可学习对全局性问题自动启用 community retrieval

### 性能考虑
- 社区检测：Louvain 对 100K 节点 < 10s，只在摄取时运行
- 不在每次 `ainsert()` 都跑：提供 `rebuild_communities()` 显式方法，或设置脏标志（新增 > 10% 节点时自动触发）

---

## 关键文件清单

| 文件 | 改动类型 |
|------|---------|
| `lightrag/base.py` | QueryParam 新增 4 个字段 |
| `lightrag/operate.py` | 新增 4 个函数，修改 3 个函数 |
| `lightrag/prompt.py` | 新增 3 个 prompt 模板 |
| `lightrag/namespace.py` | 新增 2 个命名空间（Phase 4） |
| `lightrag/lightrag.py` | 新增存储实例 + 摄取钩子（Phase 4） |
| `lightrag/community_detection.py` | 新文件（Phase 4） |

---

## 插入点速查

| 功能 | 钩入位置 | 行号 |
|------|---------|------|
| HyDE | `_perform_kg_search()` → query_embedding 计算后 | operate.py:4308-4320 |
| Router | `kg_query()` → keyword extraction 之前 | operate.py:3909 之前 |
| RRF | `_merge_all_chunks()` → round-robin 循环替换 | operate.py:4694-4783 |
| Community (查询) | `_build_query_context()` → Stage 1 后、Stage 4 前 | operate.py:5145-5271 |
| Community (摄取) | `ainsert()` → graph 构建完成后 | lightrag.py:2055 附近 |

---

## 正交性矩阵

| | RRF | HyDE | Router | Community |
|---|---|---|---|---|
| V1 Disambig | 无关 | 无关 | 无关 | 无关 |
| V2 Synonym | 无关 | 无关 | 无关 | 社区包含 SYNONYM 边 |
| V3 PPR | PPR 绕过 RRF | HyDE 改善 PPR seed | Router 可切换 multi_hop | 互补（局部 vs 全局） |
| Rerank | RRF→Rerank 顺序组合 | 无关 | 无关 | 无关 |

---

## 验证方案

1. **RRF**: 对比 round-robin 和 RRF 在相同 query 下的 chunk 排序差异，确认 rrf_score 正确计算
2. **HyDE**: 用抽象查询（如"这个领域的发展趋势"）对比启用/禁用 HyDE 的召回差异
3. **Router**: 用不同类型 query（实体查询、关系查询、综合查询）验证 mode 自动选择正确性
4. **Community**: 验证社区摘要质量 + 查询时返回的社区与 query 相关性
5. **全量正交测试**: 2^4 = 16 种开关组合，验证无互相干扰
