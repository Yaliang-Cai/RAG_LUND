# PPR 查询模式

**适用版本：** V3（feat/retrieval-router 分支及之后）  
**上次更新：** 2026-04-27

---

## 目录

1. [概述](#1-概述)
2. [论文方法论](#2-论文方法论)
   - 2.1 [Personalized PageRank 基础](#21-personalized-pagerank-基础)
   - 2.2 [图构建与边权语义](#22-图构建与边权语义)
   - 2.3 [种子选择问题](#23-种子选择问题)
   - 2.4 [Recognition Memory（HippoRAG2）](#24-recognition-memoryhipporag2)
   - 2.5 [Hub 惩罚机制](#25-hub-惩罚机制)
3. [系统实现架构](#3-系统实现架构)
   - 3.1 [整体流程](#31-整体流程)
   - 3.2 [种子构建阶段](#32-种子构建阶段)
   - 3.3 [Recognition Memory Filter 细节](#33-recognition-memory-filter-细节)
   - 3.4 [PPR 传播与 Chunk 得分](#34-ppr-传播与-chunk-得分)
   - 3.5 [Chunk 重排](#35-chunk-重排)
4. [使用说明](#4-使用说明)
   - 4.1 [基本命令](#41-基本命令)
   - 4.2 [参数参考](#42-参数参考)
   - 4.3 [常见用法示例](#43-常见用法示例)
5. [与其他模式的对比](#5-与其他模式的对比)
6. [调优建议](#6-调优建议)

---

## 1. 概述

`mode="ppr"` 是本系统中面向**知识图谱多跳检索**的核心查询模式。它基于 LightRAG 的实体-关系图，将 Personalized PageRank（PPR）算法作为检索引擎，通过从查询相关实体出发在图上传播重要性分数，找到语义上相关但不在浅层向量空间中直接相邻的 chunk。

**适合场景：**

- 需要跨多文档或多段落进行推理的事实性问题（如"A 和 B 之间通过什么机制相关？"）
- 查询涉及具体实体名称、数字指标、技术术语等精确信息
- 标准向量检索结果中 hub 实体（高度数节点）噪声较大的场景

**不适合场景：**

- 广义主题描述类问题（宜用 `global` 或 `hybrid` 模式）
- 知识图谱节点极少（图结构太稀疏，PPR 退化为向量检索）

---

## 2. 论文方法论

### 2.1 Personalized PageRank 基础

PageRank 算法原始设计用于 Web 图的全局排名。**Personalized PageRank（PPR）** 在此基础上引入个性化向量 $\mathbf{s}$，使每次随机游走有概率 $(1-\alpha)$ 被"传送回"个性化节点集合，而非均匀回到全图：

$$\mathbf{r} = \alpha \mathbf{W} \mathbf{r} + (1-\alpha) \mathbf{s}$$

其中：
- $\mathbf{r}$：各节点的 PPR 稳态分数向量
- $\mathbf{W}$：归一化邻接矩阵（按出度归一化）
- $\alpha$：阻尼系数（damping factor），控制图游走深度
- $\mathbf{s}$：**个性化向量**，即种子权重，代表查询与各实体节点的初始相关性

$\alpha$ 越大，算法越"遵守图结构"传播；越小，越倾向于回到初始种子。本系统默认 $\alpha=0.85$（`ppr_damping=0.85`）。

PPR 的核心价值在于：**与查询直接相关的实体能够将重要性沿图边传播给邻居**，使得那些通过多跳推理才能关联到查询的实体和对应 chunk 获得较高分数——这是纯向量检索无法完成的。

### 2.2 图构建与边权语义

本系统在 LightRAG 知识图谱基础上构建 PPR 使用的有向异构图，节点类型包括：

| 节点类型 | 说明 |
|---------|------|
| 实体节点 | 抽取自文档的命名实体，实体消歧后唯一化（格式：`名称\|类型`） |
| Chunk 节点（段落节点）| 文档切片，通过 DPR 向量检索定位 |

边类型及权重语义：

| 边类型 | 权重语义 | 权重计算 |
|--------|---------|---------|
| `FACTUAL`（实体–实体） | 关系被多来源文档支撑的强度 | `weight = log1p(weight_raw)`，`weight_raw` 按 source_id 增量累加 |
| `SYNONYM`（实体–实体） | 同义词余弦相似度 | `weight = cos`（raw 模式）或 `1 + cos`（plus_one 模式） |
| `chunk-entity`（chunk–实体） | 段落与实体的归属关系 | 固定值 `1.0` |

FACTUAL 边权使用 `log1p` 压缩（而非线性累加）有两方面原因：一是防止被大量文档重复提及的通用实体（hub）在 PPR 中获得压倒性权重；二是使边权分布更为平缓，从而减少 PPR 收敛时的数值不稳定。

### 2.3 种子选择问题

PPR 的检索质量高度依赖**初始种子选择**。传统做法直接使用实体向量数据库（VDB）和关系 VDB 的相似度分数作为种子权重：

$$s_e = \max(\text{vdb\_score}_e,\; \text{fact\_score}_e)$$

这一方式存在核心缺陷：**纯向量相似度无法区分"词义相近"和"对回答该问题有用"**。高相似度但语义无关的实体（例如同名词歧义、领域通用词）会污染个性化向量，使 PPR 从错误节点出发传播，最终检索到语义偏离的 chunk。

### 2.4 Recognition Memory（HippoRAG2）

HippoRAG2 提出 **Recognition Memory** 机制，在 PPR 种子确定前增加 LLM 语义验证步骤。本系统在 `mode="ppr"` 的全局路径中实现了该机制，流程为三步混合过滤：

**Step 1 — 向量检索（Numpy argsort）**

从实体 VDB 和关系 VDB 分别检索候选实体，构成候选池：

- **关系三元组**：取前 `recognition_top_k` 条关系，每条格式为 `src_id | description | tgt_id`
- **独立实体**：取前 `recognition_top_k × 2` 条实体 VDB 结果

两个来源共同组成发送给 LLM 的候选列表。注意：实体 VDB 检索量仍由 `top_k` 控制，`recognition_top_k` 仅限制发送给 LLM 的上下文大小。

**Step 2 — LLM 判断（DSPy Filter 变体）**

将候选实体和三元组组织为结构化 Prompt，要求 LLM 从中**精确挑选**与当前查询直接相关的实体标识符：

```
You are an entity relevance judge.

Query: {query}

Below are retrieved entities and facts. Select ONLY those directly relevant
to answering the query.
You MUST copy each entity identifier EXACTLY as it appears in the list below,
including any "|TYPE" suffixes and special characters.

Standalone entities:
{entity_id, one per line}

Retrieved facts:
{src_id | description | tgt_id, one per line}

Return the relevant entity identifiers only, one per line.
```

Prompt 中的严格复制约束（"copy EXACTLY"）加上后续 difflib 映射，共同防止 LLM 幻构不存在的实体。

**Step 3 — Difflib 映射（字符串模糊匹配）**

LLM 输出的文本行通过 `difflib.get_close_matches(cutoff=0.85)` 逐行映射回图中真实的 `entity_id`。cutoff=0.85 的设计意图是：容忍 LLM 输出的轻微格式误差（如尾部空格、标点变化），同时拒绝超出编辑距离阈值的幻构实体。

**权重合并**

通过识别后，对每个被认可的 `entity_id` 取 VDB 归一化分数和事实归一化分数的 max：

$$s_e^{\text{recognized}} = \max(\hat{s}_e^{\text{vdb}},\; \hat{s}_e^{\text{fact}})$$

其中 $\hat{s}$ 表示 min-max 归一化后的分数（各来源独立归一化）。独立归一化的原因是 vdb_score 和 fact_score（distance）量纲不同，直接混合会造成某一来源主导权重。

**降级策略**

LLM 判断属于非核心路径，设有多重降级保护：
- LLM 调用失败 → 回退到直接 max-merge 种子
- LLM 返回空结果 → 同上回退
- 图中不存在 recognition_top_k > 0 的配置 → 跳过 LLM 步骤直接 max-merge

### 2.5 Hub 惩罚机制

Recognition Memory 过滤之后，对高度数节点追加 hub 惩罚：

$$s_e \leftarrow s_e \;/\; \log(1 + \deg(e)), \quad \text{if } \deg(e) > \tau$$

其中 $\tau$（`hub_penalty_threshold`，默认 50）为度数阈值。此步骤作为结构性安全网补充语义过滤：即使一个高度数实体通过了 LLM 判断，其种子权重也会被图结构压缩，防止 PPR 过度集中在图中心枢纽节点附近传播。

---

## 3. 系统实现架构

### 3.1 整体流程

```
query
  │
  ├─ 关键词抽取 → 查询嵌入
  │
  ├─ [种子构建阶段]
  │     ├── 实体 VDB 检索 (top_k)      → node_datas
  │     ├── 关系 VDB 检索 (top_k)      → rel_results
  │     ├── Recognition Memory Filter  → entity_seed_weights  (LLM验证)
  │     │     └── fallback: direct max-merge
  │     └── Hub 惩罚                   → entity_seed_weights (修正后)
  │
  ├─ [PPR 传播阶段]
  │     ├── Chunk VDB 检索 (DPR)       → chunk_seed_weights × passage_node_weight
  │     └── engine.run_ppr(entity_seeds, chunk_seeds)
  │           └── → ppr_scores {node_id: score}
  │
  ├─ [Chunk 选择阶段]
  │     ├── 取 ppr_top_k 个 Chunk 节点
  │     └── 提取对应文本 → candidate_chunks
  │
  └─ [重排阶段]
        ├── CrossEncoder 重排 (enable_rerank=True)
        └── 取前 ppr_qa_top_k 条 → 送入 LLM 生成答案
```

### 3.2 种子构建阶段

种子向量 $\mathbf{s}$ 由两部分组成：

**实体种子（entity_seed_weights）**

由 Recognition Memory Filter 输出的 `{entity_id: weight}` 字典（或降级后的 direct max-merge 结果）。这部分代表查询在实体-实体图层面的初始锚点。

**Chunk 种子（chunk_seed_weights）**

通过 DPR（dense passage retrieval）向量检索从 chunk VDB 中直接检索与查询相关的段落，权重乘以 `passage_node_weight`（默认 1.0）。Chunk 种子使得 PPR 不仅从实体节点出发，也可以从与查询直接匹配的段落节点出发，提高浅层相关段落的召回率。

两套种子共同构成个性化向量输入 PPR 引擎。

### 3.3 Recognition Memory Filter 细节

`_recognition_memory_filter()` 的完整逻辑：

1. **候选池裁剪**：`top_rels = rel_results[:recognition_top_k]`，`top_nodes = node_datas[:recognition_top_k * 2]`
2. **fact_scores 构建**：同一实体在多条关系中出现时，取 `max(distance)` 作为其 fact_score
3. **独立 min-max 归一化**：`norm_vdb`、`norm_fact` 分别归一化到 [0,1]；全等分且非零时归一化为 1.0，防止有效信号被压成 0
4. **Prompt 构建与 LLM 调用**
5. **Difflib 映射**：`cutoff=0.85`，无法映射的行静默跳过
6. **权重合并**：`max(norm_vdb[eid], norm_fact[eid])`，权重为 0 的实体不纳入种子

token 保护：Prompt 超出 `recognition_prompt_max_tokens - recognition_prompt_reserved_tokens` 预算时，按优先级裁剪实体行和事实行（查询不裁剪），并记录 warning 日志。

### 3.4 PPR 传播与 Chunk 得分

PPR 引擎（`ppr_engine.py`）基于稀疏邻接矩阵进行幂迭代，收敛后每个节点得到稳态分数 $r_i$。

边权在矩阵构建时的处理：
- FACTUAL 边：使用存储的 `weight = log1p(weight_raw)`
- SYNONYM 边：使用 `weight = cos`（raw 模式，默认）或 `1 + cos`（plus_one 模式）
- chunk-entity 边：固定 `1.0`

PPR 收敛后，取所有 Chunk 节点的分数，按降序排列，截取前 `ppr_top_k` 条。

### 3.5 Chunk 重排

从 PPR 结果中选出的候选 chunk 可选择通过 CrossEncoder 重排（`enable_rerank=True`，默认开启）：

- **KG 重排**（`enable_kg_rerank`）：对实体/关系结果重排，PPR 模式中通常禁用，因为 PPR 已经完成了全图范围的结构排序
- **Chunk 重排**（`enable_rerank`）：对最终候选段落用 CrossEncoder 重打分，适合进一步精排

重排后取前 `ppr_qa_top_k` 条 chunk 送入 LLM 生成最终答案。

---

## 4. 使用说明

### 4.1 基本命令

```bash
python scripts/query_ppr.py \
  -w <workspace_id> \
  --cache-dir <writable_cache_dir> \
  -q "<问题>" \
  --mode ppr
```

`-w` 指定已建索引的 workspace 名称；`--cache-dir` 在 workspace 路径只读时（如使用他人建库的目录）指定可写缓存路径。

### 4.2 参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `ppr` | 查询模式。本文档只讨论 `ppr` |
| `--top-k` | 40 | 实体 VDB 和关系 VDB 各自的检索候选数 |
| `--ppr-damping` | 0.85 | PPR 阻尼系数 $\alpha$，越大图游走越深 |
| `--ppr-top-k` | 50 | PPR 完成后保留的候选 chunk 数量（检索宽度） |
| `--ppr-qa-top-k` | 5 | 最终送入 LLM 的 chunk 数量（答案生成上限） |
| `--passage-node-weight` | 1.0 | DPR chunk 种子权重缩放系数 |
| `--recognition-top-k` | 10 | Recognition Memory 发送给 LLM 的关系三元组数量；设为 0 则禁用 |
| `--linking-top-k` | 5 | Recognition Memory 输出的最大实体种子数（0 表示不限） |
| `--chunk-top-k` | 10 | 重排后最终 chunk 窗口大小（来自重排阶段） |
| `--no-rerank` | (关闭) | 禁用 chunk CrossEncoder 重排 |
| `--no-kg-rerank` | (关闭) | 禁用实体/关系 KG 重排（PPR 模式默认已无需此项） |
| `--trace` | (关闭) | 打印完整检索 trace JSON（含 chunks、entities、relations 列表） |

**参数语义梳理：**

- `ppr_top_k` 控制**检索宽度**：PPR 后从图中拉出多少候选 chunk
- `ppr_qa_top_k` 控制**答案生成输入量**：经过重排后实际送入 LLM prompt 的 chunk 数
- `recognition_top_k` 控制**LLM 上下文大小**：三元组越多，语义过滤更充分，但每次查询增加一次 LLM 调用延迟

### 4.3 常见用法示例

**标准 PPR 查询（含 Recognition Memory）：**
```bash
python scripts/query_ppr.py \
  -w My_Graph \
  -q "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?" \
  --mode ppr
```

**禁用 Recognition Memory（消融基线）：**
```bash
python scripts/query_ppr.py \
  -w My_Graph \
  -q "..." \
  --mode ppr \
  --recognition-top-k 0 \
  --trace
```

**扩大检索宽度，减少答案 chunk（精排严格模式）：**
```bash
python scripts/query_ppr.py \
  -w My_Graph \
  -q "..." \
  --mode ppr \
  --ppr-top-k 80 \
  --ppr-qa-top-k 3
```

**禁用重排（速度优先）：**
```bash
python scripts/query_ppr.py \
  -w My_Graph \
  -q "..." \
  --mode ppr \
  --no-rerank \
  --trace
```

**打印完整 trace（调试用）：**
```bash
python scripts/query_ppr.py \
  -w My_Graph \
  -q "..." \
  --mode ppr \
  --trace
```

trace 输出包含 `chunks`（候选段落列表及分数）、`entities`（种子实体）、`relations`（关系三元组）三个字段，可用于诊断 PPR 检索路径。

---

## 5. 与其他模式的对比

| 维度 | `ppr` | `hybrid` | `global` | `naive` |
|------|-------|----------|----------|---------|
| 检索机制 | 实体图 PPR 传播 | 实体 VDB + 关系 VDB 融合 | 全局实体图摘要 | chunk VDB 直接检索 |
| 多跳推理 | 是（图传播天然支持） | 有限（单跳实体扩展） | 否 | 否 |
| LLM 调用（种子验证） | 是（recognition memory） | 否 | 否 | 否 |
| 适合问题类型 | 精确事实、多跳推理 | 中等复杂度混合查询 | 宏观主题描述 | 浅层相似度匹配 |
| 对图稀疏性的敏感度 | 高 | 中 | 低 | 无 |
| 查询延迟 | 较高（含 LLM 验证） | 中 | 中 | 低 |

---

## 6. 调优建议

**提升多跳质量：**  
增大 `recognition_top_k`（如 15–20）让 LLM 看到更多候选关系三元组，提高种子过滤精度。同时适当增大 `ppr_top_k`（如 80–100）扩展 PPR 传播深度。

**降低查询延迟：**  
将 `recognition_top_k` 设为 0 禁用 LLM 验证（退化为直接 max-merge），或减小其数值（如 5）缩短 LLM 上下文。

**控制答案 chunk 质量：**  
保持 `ppr_qa_top_k` 较小（3–5）并启用 `enable_rerank=True`，让 CrossEncoder 在宽候选池中精选最相关段落。

**消融对比实验：**  
- 关闭 recognition memory：`--recognition-top-k 0`
- 关闭 chunk 重排：`--no-rerank`
- 固定 PPR 深度对比：在不同 `--ppr-damping`（如 0.5 vs 0.85）下比较多跳问题命中率

**synonym 边权模式：**  
在实体消歧准确率较高的库上，可尝试 `ppr_synonym_weight_mode=plus_one`（环境变量 `RAGANYTHING_PPR_SYNONYM_WEIGHT_MODE=plus_one`），为同义词边增加固定偏置，增强同义词实体间的图传播连通性。
