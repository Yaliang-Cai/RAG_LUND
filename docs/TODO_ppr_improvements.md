# V3 PPR 多跳推理优化方案

## 背景

当前 V3 基于 HippoRAG2 风格的 Personalized PageRank (PPR) 实现了图上多跳推理，核心逻辑分布在以下文件：

| 文件 | 行号 | 职责 |
|------|------|------|
| `lightrag/lightrag/ppr.py` | 19-110 | PPR 算法核心：`personalized_pagerank()`，NetworkX 图构建 + 计算 |
| `lightrag/lightrag/operate.py` | 5358-5495 | `_ppr_rank_chunks()`：完整编排（seed → 子图 → PPR → chunk 内容取回） |
| `lightrag/lightrag/operate.py` | 4467-4491 | `_perform_kg_search()`：V3 入口，`enable_multi_hop` 守卫 |
| `lightrag/lightrag/operate.py` | 4700-4760 | `_merge_all_chunks()`：PPR 优先合并 + vector 补充 |
| `lightrag/lightrag/base.py` | 733-780 | `get_subgraph_for_ppr()`：基础 BFS 实现（通用后端） |
| `lightrag/lightrag/kg/neo4j_impl.py` | 1103-1152 | `get_subgraph_for_ppr()`：Cypher 变长路径优化实现 |

### 当前流程（对应实际代码）

```
用户查询
  ↓
[1] operate.py:4467 — _perform_kg_search() 检测 enable_multi_hop=True
  ↓
[2] operate.py:5383-5406 — 构建 entity_seed_weights
    · 来源 1：_get_node_data() 已检索的 entity VDB 分（nd["vdb_score"]）
    · 来源 2：relationships_vdb.query(top_k) → rel["distance"] 赋给 src_id / tgt_id
    · 两信号对同一实体取 max
  ↓
[3] operate.py:5411-5415 — get_subgraph_for_ppr(seed_ids, max_depth=2)
    · 基础实现（base.py:745）：BFS max_depth 轮，逐节点 get_node/get_node_edges/get_edge
    · Neo4j 实现（neo4j_impl.py:1103）：单条 Cypher MATCH path=(seed)-[*1..depth]-(n)
    · 返回 (nodes_list, edges_list)；边 dict 仅含 {src, tgt, weight}，无 source_id
  ↓
[4] operate.py:5417-5439 — 从节点 source_id 构建虚拟 chunk 节点 + chunk-entity 边
    · 仅遍历节点的 source_id 字段（GRAPH_FIELD_SEP 分隔）
    · 边的 source_id 未使用（见方案 B）
    · chunk-entity 边 weight 固定 1.0（见方案 A）
  ↓
[5] operate.py:5441-5460 — chunks_vdb.query(top_k = ppr_top_k × 2)
    · 对检索结果做 min-max 归一化
    · chunk_seed_weights[cid] = normalized × passage_node_weight(0.05)
    · 仅对已存在于 chunk_to_entities 中的 chunk 赋权重
  ↓
[6] ppr.py:81-110 — personalized_pagerank()
    · 合并 entity + chunk seed 权重，归一化到 sum=1
    · nx.pagerank(G, alpha=damping, personalization=..., weight="weight")
    · 抽取 chunk 节点 PPR 分，降序取 top_k
    · 失败时 fallback：直接返回 seed chunk_seed_weights 排序
  ↓
[7] operate.py:5477-5495 — text_chunks_db.get_by_ids(ranked_chunk_ids)
    · 附加 source_type="ppr", ppr_score 字段
  ↓
[8] operate.py:4720-4760 — _merge_all_chunks() PPR 优先路径
    · PPR chunks 全部入列（图结构排序）
    · vector_chunks 去重后补充到末尾
    · entity/relation chunk 推导被跳过（enable_multi_hop 时）
```

### 已识别的问题（对应实际代码位置）

| 问题 | 代码位置 | 影响 |
|------|----------|------|
| chunk-entity 边统一 `weight=1.0` | `ppr.py:76` | 忽略共现频率，高频关联 chunk 未得额外权重 |
| 边的 `source_id` 未参与 chunk 映射 | `operate.py:5417-5428`，`base.py:768-773` | 遗漏通过关系而非实体关联的文档（如"华为与5G"关系对应的 chunk） |
| hub 节点未被惩罚 | `operate.py:5383-5406`（seed 构建） | 高度数通用实体（"中国"、"技术"）获得 seed 权重后子图膨胀，PPR 信号稀释 |
| `damping=0.5` 固定 | `operate.py:5463` | 对简单/复杂查询一视同仁，多跳深度无自适应 |
| PPR 输出未经语义重排 | `operate.py:5477-5495`（取回后直接返回） | 图结构高分但语义无关的 chunk 排在前面 |
| SYNONYM 边与普通关系边权重相同 | `ppr.py:63` | 同义关系传播强度与事实关系相同，语义相似性被过估 |

---

## 优化方案

### 方案 A：Hub 节点惩罚（P0）

**问题**：VDB 检索可能把高连接度的通用实体（"技术"、"公司"、"中国"）选为种子，
导致子图膨胀，PPR 信号被大量无关邻居稀释。

**方案**：在构建 `entity_seed_weights` 后，用已有的 `node_degrees_batch()` 查询种子实体的度数，
对高度数节点施加对数惩罚：

```python
adjusted_weight = weight / log(1 + degree)
```

**实现位置**：`operate.py:5408`（Step 1 entity_seed_weights 构建完成后，Step 2 子图提取前）

```python
# 插入位置：operate.py _ppr_rank_chunks() Step 1 末尾，约 5408 行
if query_param.hub_penalty_threshold > 0:
    degrees = await knowledge_graph_inst.node_degrees_batch(seed_ids)
    import math
    for eid in list(entity_seed_weights):
        deg = degrees.get(eid, 0)
        if deg > query_param.hub_penalty_threshold:
            entity_seed_weights[eid] /= math.log(1 + deg)
```

**复用的已有接口**：
- `knowledge_graph_inst.node_degrees_batch(seed_ids)` — `_get_node_data()` 中（`operate.py:5519`）已并发调用过，可参考其用法。基类逐条 fallback，Neo4j 有批量 Cypher。

**兼容性**：✅ 所有图后端均兼容

**参数建议**：
```python
# base.py QueryParam 新增
hub_penalty_threshold: int = 50   # 度数超过此值才施加惩罚；0 = 禁用
```

---

### 方案 B：边 source_id 扩充 chunk-entity 映射（P1）

**问题**：当前虚拟 chunk 节点仅从**节点**的 `source_id` 建立（`operate.py:5417-5428`）。
边（关系）也有 `source_id`，记录了哪些 chunk 描述了该关系，
但 `base.py:768-773` 构建边 dict 时只保留了 `{src, tgt, weight}`，丢弃了 `source_id`。

例：chunk_42 描述了"华为与5G"这条关系，但"华为"节点的 `source_id` 里可能没有 chunk_42。
这个 chunk 在当前实现中对 PPR 完全不可见。

**改动 1：`base.py:768-773`** — BFS 实现的 edge dict 加入 `source_id`：
```python
# 当前
visited_edges.append({
    "src": src,
    "tgt": tgt,
    "weight": float(edge_data.get("weight", 1.0)) if edge_data else 1.0,
})

# 修改后
visited_edges.append({
    "src": src,
    "tgt": tgt,
    "weight": float(edge_data.get("weight", 1.0)) if edge_data else 1.0,
    "source_id": edge_data.get("source_id", "") if edge_data else "",  # +1 行
})
```

**改动 2：`neo4j_impl.py`**（Cypher 已有 `RETURN properties(r) AS rprops`，找到构建 edge dict 处）：
```python
"source_id": rprops.get("source_id", "")   # +1 行
```

**改动 3：`operate.py:5417-5439`** — chunk 映射循环追加对边的处理：
```python
# 当前：仅遍历节点
for node in subgraph_nodes:
    for chunk_id in split_string_by_multi_markers(node["source_id"], [GRAPH_FIELD_SEP]):
        chunk_to_entities.setdefault(chunk_id.strip(), []).append(node["entity_id"])

# 新增：遍历边（追加在节点循环之后，~5430 行）
for edge in subgraph_edges:
    for chunk_id in split_string_by_multi_markers(
        edge.get("source_id", ""), [GRAPH_FIELD_SEP]
    ):
        chunk_id = chunk_id.strip()
        if chunk_id:
            chunk_to_entities.setdefault(chunk_id, []).append(edge["src"])
            chunk_to_entities.setdefault(chunk_id, []).append(edge["tgt"])
```

**兼容性**：✅ 三处改动各约 1-5 行，不改变函数签名

---

### 方案 C：SYNONYM 边折扣（P1，与方案 B 配套）

**问题**：V2 同义链接（`synonym_linking.py`）创建的 SYNONYM 边与普通事实关系边在 PPR 传播中权重相同（`ppr.py:63`），但语义相似性不等于事实关系，不应有同等强度的传播。

**依赖**：方案 B 扩展 edge dict 时同步携带 `edge_type` 字段。

**改动 1：`base.py:768-773`**（与方案 B 合并改动）：
```python
"edge_type": edge_data.get("edge_type") if edge_data else None,
```

**改动 2：`neo4j_impl.py`**（同方案 B）：
```python
"edge_type": rprops.get("edge_type"),
```

**改动 3：`ppr.py:59-64`** — 图构建的边权重处理：
```python
# 当前
for edge in entity_edges:
    src, tgt = edge.get("src"), edge.get("tgt")
    if src and tgt:
        G.add_edge(src, tgt, weight=float(edge.get("weight", 1.0)))

# 修改后
for edge in entity_edges:
    src, tgt = edge.get("src"), edge.get("tgt")
    if src and tgt:
        w = float(edge.get("weight", 1.0))
        if edge.get("edge_type") == "SYNONYM":
            w *= synonym_edge_discount   # QueryParam 参数，默认 0.5
        G.add_edge(src, tgt, weight=w)
```

**参数建议**：
```python
# base.py QueryParam 新增
synonym_edge_discount: float = 0.5   # SYNONYM 边权重折扣；1.0 = 无折扣
```

**兼容性**：✅ `edge_type` 字段在非 SYNONYM 边为 None，条件分支有 default；V2 未启用时无 SYNONYM 边，逻辑不触发

---

### 方案 D：自适应 damping（P0）

**问题**：`damping=0.5` 固定传入 `nx.pagerank(alpha=damping, ...)`（`operate.py:5463` / `ppr.py:98`）。
PageRank 中 `alpha`（即 damping）越高，分布越集中在 seed 附近；越低传播越广，多跳效果越强。

| 查询类型 | 期望行为 | 建议 damping |
|---------|---------|-------------|
| "X 是什么" | 集中在直接相关实体 | 0.6–0.7 |
| "A 通过什么影响 C" | 允许远距离传播 | 0.3–0.4 |
| 默认 | 折中 | 0.5 |

**实现位置**：`operate.py:5462`（`personalized_pagerank()` 调用前），插入如下逻辑：

```python
# 根据种子数量自动调整 damping（无需 LLM 调用）
n_seeds = len(entity_seed_weights)
if n_seeds <= 2:
    effective_damping = max(query_param.ppr_damping - 0.15, 0.2)   # 精确种子→更多传播
elif n_seeds >= 8:
    effective_damping = min(query_param.ppr_damping + 0.1, 0.8)    # 大量种子→集中在近邻
else:
    effective_damping = query_param.ppr_damping

ppr_ranked = personalized_pagerank(
    ...,
    damping=effective_damping,   # 替换原来的 query_param.ppr_damping
    ...
)
```

**兼容性**：✅ 纯 Python 逻辑，与所有后端无关；`ppr_damping` 参数仍作为基准值

---

### 方案 E：PPR 输出 + Reranker 二次排序（P0）

**问题**：`_ppr_rank_chunks()` 在 `operate.py:5481-5495` 取回 chunk 内容后直接返回，排序完全由 PPR 图结构分数决定，可能选出"图上重要但内容无关"的 chunk。系统已有 `apply_rerank_if_enabled()`（已在 `_build_query_context()` 等处调用），但未集成到 PPR 路径。

**实现位置**：`operate.py:5493`，`text_chunks_db.get_by_ids()` 取回并组装完 `result_chunks` 之后，`return result_chunks` 之前：

```python
# operate.py _ppr_rank_chunks() 末尾，约 5493 行
if query_param.enable_rerank and result_chunks:
    from lightrag.operate import apply_rerank_if_enabled   # 或直接 import（同文件内可直接调用）
    reranked = await apply_rerank_if_enabled(
        query=query,
        retrieved_docs=result_chunks,
        global_config=global_config,   # 需作为参数传入 _ppr_rank_chunks
        top_n=len(result_chunks),
    )
    if reranked:
        # 融合：α × PPR分(归一化) + (1-α) × rerank分
        alpha = query_param.ppr_rerank_alpha   # 默认 0.6
        ppr_scores = {c["chunk_id"]: c.get("ppr_score", 0.0) for c in result_chunks}
        max_ppr = max(ppr_scores.values(), default=1.0) or 1.0
        for chunk in reranked:
            cid = chunk.get("chunk_id")
            ppr_norm = ppr_scores.get(cid, 0.0) / max_ppr
            rerank_s = chunk.get("rerank_score", 0.0)
            chunk["ppr_score"] = alpha * ppr_norm + (1 - alpha) * rerank_s
        result_chunks = sorted(reranked, key=lambda x: x["ppr_score"], reverse=True)

return result_chunks
```

**函数签名变动**：`_ppr_rank_chunks()` 需新增 `global_config: dict` 参数（调用方 `operate.py:4472` 处已有 `global_config` 可传入）。

**参数建议**：
```python
# base.py QueryParam 新增
ppr_rerank_alpha: float = 0.6   # PPR 分权重；0.0 = 纯 rerank，1.0 = 纯 PPR
```

**兼容性**：✅ `apply_rerank_if_enabled` 内部检查 reranker 是否配置，未配置时返回原列表不变

---

### 方案 F：文档级聚合加分（P2）

**问题**：PPR 独立评估每个 chunk，忽略了"同一文档多个 chunk 都被选中"这一强信号。多个相关 chunk 来自同一文档，说明该文档整体与查询高度相关。

**实现位置**：`operate.py:4722-4740`，`_merge_all_chunks()` PPR 优先路径，在 PPR chunks 去重入列之前插入：

```python
# _merge_all_chunks() PPR 分支入口，约 operate.py:4722
if ppr_chunks:
    import math
    from collections import Counter

    # full_doc_id 已由 text_chunks_db.get_by_ids() 取回（operate.py:5479）
    doc_chunk_counts = Counter(
        c.get("full_doc_id") for c in ppr_chunks if c.get("full_doc_id")
    )
    for chunk in ppr_chunks:
        doc_id = chunk.get("full_doc_id")
        if doc_id and doc_chunk_counts[doc_id] > 1:
            bonus = math.log(1 + doc_chunk_counts[doc_id]) * 0.05
            chunk["ppr_score"] = chunk.get("ppr_score", 0.0) + bonus
    ppr_chunks.sort(key=lambda x: x.get("ppr_score", 0.0), reverse=True)

    # 之后是已有的去重入列逻辑 ...
```

**数据来源**：`full_doc_id` 字段来自 `TextChunkSchema`，在 `_ppr_rank_chunks()` 的 `text_chunks_db.get_by_ids()`（`operate.py:5479`）已取回，`chunk_data.copy()` 后原样传入 `result_chunks`，无需额外 IO。

**兼容性**：✅ 依赖 `text_chunks_db` KV 存储，所有后端均有此字段；`full_doc_id` 为空时 Counter 跳过

---

## 兼容性汇总

| 方案 | Neo4j | Base图后端 | Qdrant | NanoVectorDB | 是否需改 API |
|------|-------|-----------|--------|--------------|------------|
| A. Hub 惩罚 | ✅ | ✅ | — | — | 否 |
| B. 边 source_id 扩充 | ✅ 改1行 | ✅ 改1行 | — | — | 是（轻微） |
| C. SYNONYM 折扣 | ✅ 改1行 | ✅ 改1行 | — | — | 是（轻微，与B配套） |
| D. 自适应 damping | ✅ | ✅ | — | — | 否 |
| E. PPR + Reranker | ✅ | ✅ | ✅ | ✅ | 否 |
| F. 文档级聚合 | ✅ | ✅ | ✅ | ✅ | 否 |

---

## 新增 QueryParam 参数汇总

```python
# base.py QueryParam 中新增
hub_penalty_threshold: int = 50
"""度数超过此值的实体在 PPR 种子权重中受对数惩罚。0 表示禁用。"""

synonym_edge_discount: float = 0.5
"""SYNONYM 边在 PPR 图中的权重折扣系数（需方案 B/C 支持）。"""

ppr_rerank_alpha: float = 0.6
"""PPR 分与 rerank 分的融合权重。1.0 = 纯 PPR，0.0 = 纯 rerank。"""
```

现有参数无需修改：

- `enable_multi_hop: bool = False`
- `multi_hop_depth: int = 2`
- `ppr_damping: float = 0.5`（自适应 damping 在此基础上浮动）
- `ppr_top_k: int = 50`
- `passage_node_weight: float = 0.05`

---

## 实施顺序建议

```
阶段 1（无 API 改动，可独立验证）
  └─ A. Hub 节点惩罚
  └─ D. 自适应 damping
  └─ E. PPR + Reranker

阶段 2（统一扩展 get_subgraph_for_ppr 接口）
  └─ B. 边 source_id 扩充（改 base.py + neo4j_impl.py 各 1 行）
  └─ C. SYNONYM 折扣（依赖 B）

阶段 3（锦上添花）
  └─ F. 文档级聚合
```

---

## 验证方案

1. **单元测试**：构造包含 hub 节点、SYNONYM 边的测试图，验证各方案对 PPR 输出排序的影响
2. **集成测试**：对比开启/关闭各方案时，固定查询的 PPR chunk 排序变化
3. **端到端**：准备多跳查询集（如"A 通过什么机制影响 C"），对比回答质量
4. **延迟测试**：各方案的额外查询开销（主要是方案 A 的 `node_degrees_batch` 调用）
