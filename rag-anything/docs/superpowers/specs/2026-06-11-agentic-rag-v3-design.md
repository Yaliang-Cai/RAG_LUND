# Agentic RAG v3 设计：真 Agent Loop + Session 记忆

日期：2026-06-11
状态：已与需求方逐节讨论定稿，待审阅
前置讨论：v2（`raganything/retrieval/agent_graph_v2.py`）问题诊断 + 八个细节专题讨论

---

## 1. 背景与问题

v2（`AdaptiveAgentGraphV2`）是固定状态机式的"反思 RAG 管线"，经代码核实存在四个结构性问题：

| # | 问题 | 代码根源 |
|---|---|---|
| P1 | 不是真 agent loop：决策由硬编码查找表（`RecoveryPolicy`）完成，LLM 只做分类填空 | `agent_graph_v2.py:121` 固定循环；`recovery_policy.py:49-100` if/else 表；multihop 识别是英文正则（中文全 miss） |
| P2 | 单模型身兼数职（classifier/grader/rewriter/checker/generator），self-preference bias | `agent_graph_v2.py:73` 同一 `self._llm` 注入全部角色 |
| P3 | 大量重复检索与 rerank：每轮全量重检索、证据不跨步累积、恢复阶梯收敛到重复动作、检查循环再追加检索 | 每 cycle 全 profile 多路 + 每路 rerank 30 候选；`tried_signatures` 只按三元组去重 |
| P4 | 不自适应调节连续参数：top_k/rerank 上限/阈值全程静态 | `QueryParam` 入口构造后只读；profile 参数为类常量 |

另有体验缺陷：ungrounded 时返回空字符串（`agent_graph_v2.py:251`）；`conversation_history` 仅送 LLM、不参与检索（`query_routes.py:387`），session 内指代消解与查询改写缺失。

## 2. 目标与非目标

**目标**
- LLM 驱动的决策循环替代查找表；检索路径工具化，参数成为每步决策变量
- 证据池跨步累积，rerank 增量化，消灭重复劳动
- Session 级上下文管理：指代消解、查询改写、跨轮证据复用
- 成本点预算 + 墙钟/token 双护栏（交互场景墙钟 60s）
- 查询画像驱动 Recall/Precision/性能三角取舍与生成模式选择
- 角色模型分离架构：默认单模型（现有 Qwen3-VL-30B 单端点），小模型 judge 端点即插即用零代码改动
- 保留 v2 trace 契约，前端 AgenticTrace 不改即可用

**非目标**
- 不动现有 `/query` 端点与 v2 实现（v2 完整保留为回退模式与确定性 fallback）
- 不替换底层检索路径实现（Qdrant/PPR/graph 等只被包装，不被重写）
- 不在本期实现 EpisodicMemory/SemanticMemory 持久化（SessionMemory 预留 dump/load 接口）

## 3. 总体架构

新建 `raganything/agent/` 包，与 `retrieval/`（v2）并存，`agentic_mode=v2|agent` 切换：

```
raganything/agent/
├── loop.py        # AgentLoop：决策-执行-观察主循环 + 快速通道
├── tools.py       # ToolRegistry + 检索工具（包装现有 router paths）+ inspect_image
├── evidence.py    # EvidencePool + 事实账本（fact ledger）
├── session.py     # SessionMemory（WorkingMemory 落地）
├── models.py      # ModelPool：角色→端点映射 + 回退/熔断
├── budget.py      # 成本点预算 + 墙钟/token 护栏
├── planner.py     # 改写+分类+实体登记合并调用；初始计划
└── trace.py       # v2 兼容 trace 输出 + agent_decisions
```

新端点 `POST /agent/chat`，携带 `workspace_id`、`session_id`、`query` 及可选覆盖参数。

## 4. 决策机制

### 4.1 决策 schema

每步一次 LLM 调用（planner 角色，temperature=0），输出单个 JSON：

```json
{
  "thought": "一句话推理（必填，限一句，进 trace）",
  "action": "search_hybrid",
  "params": {"query": "...", "top_k": 15, "expand": "none"},
  "stop": false,
  "reclassify": null
}
```

- 终结动作 `action: "answer"`，params 带 `generation_mode: direct|map_reduce|cot_reflect`；`stop` 仅为冗余校验位，与 action 不一致时以 action 为准
- `reclassify`：agent 中途改判画像时填新画像名（见 §9.3）
- 不要求长 CoT——决策质量靠 observation 的结构化，兼容未来小模型 planner

### 4.2 Prompt 组装（token 经济）

```
[system] 角色定义 + 工具卡片（严格静态 → vLLM prefix cache 命中）
[user]   画像 + 改写后查询 + 证据池摘要 + 行动历史 + 剩余预算 + 工具动态状态 + 输出格式
```

- 工具卡片含：用途、何时该用/不该用、参数及取值范围、成本点数、典型耗时。**静态**，成本数字只在档位跨越时更新（§8.2）
- 证据池摘要不含 chunk 原文，只含：chunk 数、coverage、found/missing facts、行动历史每步一行（含新增/重复命中数，如 `+8 new (7 dup)`）
- 工具动态状态（如 PPR ready/cold）放 user 段，不进 system，保护前缀缓存
- 单次决策输入数百 token，输出 <150 token

### 4.3 无效输出防御（四层确定性代码）

1. 解析层：复用 `retrieval/json_utils.load_json_object`
2. 校验归一层：`_normalize_decision`——未知 action 用 difflib 匹配最近工具名；参数越界 clamp 至工具声明的 [min,max]；未知参数丢弃；缺 query 回填当前改写查询
3. 重复动作守卫：`(action, query 规范化, 关键参数)` 签名查重，**轮内生效**（跨轮允许重检索，保证新文档可见性）。重复动作拒绝执行并在下步 observation 告知
4. 硬上限：步数绝对上限 8，独立于预算

### 4.4 失败降级

```
决策解析失败 → 带错误信息重问 1 次 → 仍失败 → 确定性降级：RecoveryPolicy 查表选动作
```

**v2 的 `RecoveryPolicy` 不删除，降级为 agent 的确定性 fallback 大脑**。LLM 规划失效（解析错误/连续无效/端点超时）时系统退化为 v2 行为，下限有保证。

### 4.5 快速通道

planner 在画像分类时对 `factoid + 高置信度（confidence ≥ 0.8，可配）` 直接产出直通计划：`检索 → grade → 充分则 answer`，全程零决策调用；首 grade 不过才落入完整 loop（代价仅一次 T1 检索）。其余画像正常进 loop，但首步检索即 sufficient 时 agent 下一决策自然是 answer——等效于"一次决策调用的短通道"。快速通道全程预计 10-20s。

## 5. EvidencePool

### 5.1 分数体系（正确性关键）

cross-encoder 分数是 (query, chunk) 对的函数，不可跨 query 复用。分数分两类：

- **发现分数**（rrf/path 分数，相对子查询）：仅用于入池准入判断，进池后作废，仅留溯源
- **规范分数** canonical_score（相对本轮规范查询，即改写后的自包含问题）：池内排序、装填、grader 取样唯一依据

规范查询一轮内固定 → **增量 rerank 严格成立**：新 chunk 入池时对规范查询打一次分，本轮终生有效。各工具内部的 per-path 全量 rerank 取消，rerank 集中到池准入口。

### 5.2 数据结构

```python
PoolEntry:
  chunk_id          # LightRAG chunk id（content 哈希，内容寻址）
  content, file_path, modal_type
  image_paths       # 入池时解析缓存（原为生成前扫描，提前）
  canonical_score
  provenance: [{step, tool, sub_query, rrf_score}, ...]  # 每次发现追加
  hit_count         # 多路独立佐证计数，默认不加权，仅作装填 tiebreaker
  supports: [fact_id, ...]
```

- 去重键 chunk_id；无 id 时合成键用 **content 哈希**（修复 v2 用 content 前 200 字符 + index 的潜在误判）
- 重复命中：保留条目、追加 provenance、hit_count+1

### 5.3 事实账本（coverage 量化）

grader 维护跨步骤账本：

```python
ledger = {"facts": [{"id","text","status: found|missing|unverifiable","chunks":[...],"attempts":[tool,...]}],
          "coverage": found / (total - unverifiable)}
```

- 每步 grader 输入 = 上版账本 + 新增 chunk + **池内与当前 missing facts 相关的旧 chunk（按 canonical score 取前几条）**——后者修复结构性盲区：后发现的 fact 必须能拿早期入池 chunk 核对，否则假阴性 missing 导致定点检索全命中旧 chunk（去重后 grader 永远看不到）而死循环
- coverage 定义可解释：必要事实确认比例
- missing facts 直接作为定点补检的查询目标
- **unverifiable 放弃阈值（防死循环）**：同一 missing fact 被 ≥2 个不同工具定点补检仍无法证实 → 标记 `unverifiable`，退出补检目标集。防止 grader/planner 幻觉出的"缺失细节"或语料中根本不存在的细节耗尽预算与步数。有效 coverage 分母同步剔除（否则永远到不了 sufficient，死循环换个形式存在）；unverifiable 事实必须在最终答案中显式披露（"以下细节在语料中无法证实：…"）
- supports 反向标记：装填时支撑事实的 chunk 优先于高分但未关联事实的 chunk

### 5.4 条件终审（锚定保险）

账本锚定的两形态：假阳性 found 延续（grader 错判后不再补检）；结构性盲区（已由 §5.3 输入窗口修复）。终审针对前者：

- **快速通道永不终审**（首 grade 本就是无账本全量评估）
- **长路径条件触发**（满足其一，阈值均可配）：账本携带 ≥3 步 / 某 found fact 仅由单一 chunk 支撑且其 canonical_score < 0.4 / 近两步检索新增 chunk 去重率 > 50%
- 终审 = 无账本、全池 top-K、fresh grade。分歧处理：fresh coverage 明显低于账本 → 用 fresh 结果重建账本回到 loop（预算允许），否则按部分答案路径收尾
- 终审评估对象是外部 chunk 文本而非模型自产物，**不引入 self-preference**；残余风险是同模型同盲点（误差相关性），单模型模式无解，靠小模型 judge 对冲——已知局限，明示

### 5.5 池上限与淘汰

| 层 | 内容 | 上限 | 淘汰 | 生命周期 |
|---|---|---|---|---|
| 轮内工作池 | PoolEntry 全量 | ~200 | canonical_score 最低先逐出；支撑 found fact 者豁免 | 一轮 |
| session 缓存 | chunk_id → content/meta（无分数） | ~1000 LRU | LRU | session（随 TTL） |

跨轮：新轮工作池清空（规范查询变了，旧分数全失效）；检索命中 session 缓存跳过取内容 IO，rerank 照常做（毫秒级，不省）。

**三级预算的角色区分**（易混淆，明确）：

| 数字 | 角色 | 边界对象 |
|---|---|---|
| 池上限 ~200 条 | 候选管理边界（内存/管理成本） | 轮内工作池 |
| `max_context_tokens`（默认 12k） | **单次生成调用**的上下文预算，装填器严格按 canonical_score + supports 优先级装填，装不下即抛弃——低分无支撑 chunk 即使在池中也进不了 prompt | 每次 LLM 调用 |
| 30k token 护栏（§8.1） | **全轮累计**全部 LLM 调用的总量护栏 | 一轮问答 |

200 条候选远超 12k 装填容量是预期内状态：池是"可供挑选的候选集"，prompt 是"装填结果"，两者天然不等。

### 5.6 失效（与治理层联动）

chunk_id 内容寻址 → 新增文档不产生过期内容：

- **文档删除 → 外科手术失效**：复用治理层 provenance（doc_id→chunk_ids），删除任务 callback 中从该 workspace 全部 session 缓存移除对应条目（治理合规要求：被删文档不得再出现在答案中）
- **文档新增 → 不失效**；配套纪律：动作签名查重仅轮内，跨轮永远允许重检索
- workspace 删除/冻结 → 该 workspace 全部 session 缓存丢弃
- 挂载点：`LocalRagService._register_callbacks_to_rag`（与 `GlobalPPREngine.invalidate()` 同路）

## 6. SessionMemory

### 6.1 结构

```python
{
  "session_id": ...,
  "active_entities": [{"name","note","last_turn"}],  # 上限 12，按 last_turn 最旧逐出
  "recent_turns": [...],                             # 最近 N=6 轮原文（可配）
  "history_summary": "...",                          # N 轮前压缩摘要，上限 ~300 token
  "evidence_cache": <session 级 chunk 缓存，见 §5.5>,
}
```

### 6.2 更新机制（零感知延迟）

- **轮初**：改写+分类合并调用（§9.1）输出 `entities_referenced`，顺手登记（指代消解本来就要做，零边际成本）
- **轮末**：答案返回**之后**异步 summarizer 调用：提取/更新实体、把滑出 recent_turns 窗口的轮次合并进 history_summary（滚动压缩）

### 6.3 上下文边界与 KV cache

- 最近 N 轮原文进 context window，LLM 自然消解窗口内指代；窗口外靠 history_summary + active_entities
- Prompt 前缀顺序固定：`system → history_summary → recent_turns → 本轮内容`；摘要只在轮末变化，轮内多次调用前缀稳定 → vLLM `--enable-prefix-caching`（已开启）命中

### 6.4 生命周期与并发

- 内存 dict + 滑动 TTL 2h（访问续期），后台清扫；session 总数上限 256，LRU 逐出
- **同 session 并发请求直接拒绝，不排队**：第一个查询运行中收到同 session 第二个请求 → `409 Conflict`，响应体含进行中查询摘要与动作提示（等待 / 调用 cancel）。理由：静默排队导致意外 token 消耗，且排队请求会基于完成后的 session 状态做指代消解，偏离用户发问时的意图
- 跨 session 用全局信号量限制并发 agent loop 数保护 vLLM 端点
- 预留 `dump()/load()`（JSON 可序列化），对接后续 PG 持久化（目标架构 WorkingMemory→存储抽象层预埋）

### 6.5 取消与改问

- `POST /agent/sessions/{id}/cancel`：置取消标志，loop 在**每个决策步边界**（天然检查点）优雅退出，返回结构化部分结果（账本快照 + 当前最优候选答案或拒答）
- 取消后状态处理：session 缓存中已取 chunk 保留（内容寻址，依然有效）；轮内工作池丢弃；recent_turns 记录该轮为"已取消"，不进 history_summary
- vLLM 侧客户端断连即中止生成，取消的计算浪费极小
- "改问" = 前端"停止并重新提问"（cancel + 新请求），与 §6.4 的 409 形成闭环

## 7. 工具清单与性价比路由

### 7.1 工具与成本档位

| 工具 | 包装对象 | 档位/点数 | 备注 |
|---|---|---|---|
| search_sparse | qdrant_sparse (BM25) | T1 / 1 | 精确词项首选 |
| search_dense | qdrant dense | T1 / 1 | 语义查询首选 |
| rewrite_query | rewriter | T1 / 1 | |
| search_hybrid | qdrant_hybrid (RRF) | T2 / 2 | 标准武器 |
| search_graph | local_kg / global_kg | T2 / 2 | |
| inspect_image | VLM 定向看图 | T2 / 2 | 见 §11.3 |
| search_ppr | ppr（含 recognition LLM 调用） | T3 / 4 | 多跳专用 |
| decompose_search | 分解+并行多路 | T4 / 8 | 最后手段 |
| answer | 生成（终结） | — | |

`top_k`、`rerank_candidate_cap`、`min_rerank_score` 等为工具入参（带 [min,max] 声明），不再是全局常量。

### 7.2 路由铁律（写入 planner prompt 与工具卡片）

- **Qdrant sparse/dense 是默认武器，hybrid 是标准武器，图谱 PPR 是多跳专用武器**（需求方实测背书）
- 升级顺序：同工具调参（扩 top_k，最便宜，增量 rerank 成本极低）→ 换同级工具 → 升档（T3 PPR）→ decompose（T4）。绝不开局 full 全开
- 账本 missing facts 定点补检优先于一切生成式查询扩展（证据驱动 > 猜测驱动）

### 7.3 PPR 冷启动（延迟问题，不计点数）

1. `GlobalPPREngine` 全图加载挂入现有 `_ensure_workspace_warmed` 后台预热，不占查询墙钟
2. 工具动态状态进 observation user 段：`search_ppr: cold (预计构建 ~Xs)`，剩余墙钟不足时 agent 自然绕开
3. 入库后 `invalidate()`（已有）→ 状态回 cold → 下次后台再预热

### 7.4 查询扩展：MQE / HyDE 作为 expand 参数

不增设工具，检索工具统一参数 `expand: none|mqe|hyde`：

| expand | 实现 | 额外点数 | 适用 | 禁忌 |
|---|---|---|---|---|
| none | 原样检索 | 0 | 默认；精确词项 | — |
| mqe | 1 次 LLM 调用产 3 变体（单次生成）→ 并行检索 → RRF → 去重入池 | +1 | summary 广召回 | factoid（稀释 precision） |
| hyde | 1 次 LLM 调用生成假设答案 → 以其向量检索 | +1 | 措辞与文档词汇差距大 | 仅 dense 路；领域术语模型不熟时慎用 |

MQE 多变体同时命中 → hit_count 升高，多视角佐证免费获得。

## 8. 预算模型

### 8.1 结构

```
主预算：成本点（agent 决策唯一显式约束，随画像分配）
护栏一：墙钟 60s（交互场景，需求方确认；软阈值 45s）
护栏二：LLM token 累计上限（默认每轮 30k，可配；模型窗口 64k，留足生成余量）
```

画像初始点数：factoid 6 / comparison 10 / summary 12 / multihop 16 / unknown 10。评测场景允许关墙钟护栏（避免污染质量对比）。

### 8.2 标定：两阶段

- 冷启动：上表静态档位（按路径操作构成估算）
- 运行时校准：工具调用墙钟/token 已进 trace（Phoenix 链路可观测），按 workspace 滑动平均；**仅跨档位时更新工具卡片**（频繁改数字击穿 prefix cache，档位离散化使更新成为罕见事件）

### 8.3 耗尽兜底

```
剩余 ≤20% → observation 注入"预算告急，评估当前最优候选是否可作答"
点数耗尽/任一护栏触发 → 检索工具全部下架，仅剩 answer：
  coverage ≥ 0.5 → 部分答案 + 显式声明"基于不完整证据，已确认X、缺失Y"
  coverage < 0.5 → 结构化拒答（缺哪些事实、试过哪些检索）
```

修复 v2 缺陷：**任何终止路径都返回结构化结果**（答案或带原因的拒答 + 账本快照），不再返回空字符串。

## 9. 查询画像分类器

### 9.1 一次调用三产出（与改写合并）

轮初单次 LLM 调用（rewriter 角色）：

```json
{
  "standalone_query": "自包含改写（指代消解在此完成）",
  "archetype": "factoid|summary|multihop|comparison|unknown",
  "confidence": 0.85,
  "exact_terms": ["..."],
  "suggested_expand": "none|mqe|hyde",
  "visual_intent": false,
  "entities_referenced": [...]
}
```

结果按规范化查询缓存（复用 `RouterCache` 机制）。

### 9.2 类别集合（刻意只有五个）

factoid / summary / multihop / comparison / unknown。画像唯一职责是定**初始计划与预算**；类别多则准确率降，错判代价由纠偏机制兜底。unknown（含低置信）走 T2 hybrid + 10 点的稳妥默认。

画像 → 策略矩阵：

| 画像 | 初始检索 | top_k 起点 | 生成模式 |
|---|---|---|---|
| factoid（精确词项） | sparse + none | ~5 | direct |
| factoid（语义型） | dense + none，miss 后试 hyde | ~5 | direct |
| summary | hybrid + mqe | ~25-30 | map_reduce（超预算时） |
| multihop | hybrid + none 试探 → missing_relation 则 ppr → 账本定点补检 | ~15 | cot_reflect |
| comparison | 双实体并行 | 每侧 ~10 | 结构化对比 |
| unknown | hybrid + none | ~15 | direct |

### 9.3 画像是先验不是镣铐

1. 工具永远全量可用，画像从不裁剪工具空间
2. observation 中证据信号（账本/failure_type）排在画像前，prompt 明示"与证据冲突时以证据为准"
3. **改判触发一次性预算升档**：decision 带 `reclassify` 或 grader 连续 `needs_decomposition`/`missing_relation` → 预算补到新画像额度，每轮最多升一次（防骗预算）。改判事件进 trace，作为分类器质量监控指标

## 10. 生成模式

- **direct**：装填器按 canonical score + fact 支撑优先装填至 `max_context_tokens`（§5.5 三级预算）→ 单次生成
- **map_reduce**（summary 且池超上下文预算才触发）：按 file_path 聚类分组 → 并行 map（每组就问题总结，~300 token，vLLM 自然 batch）→ reduce 综合作答，引用保留组→chunk 映射。池在预算内退化为 direct
- **cot_reflect**（multihop）：账本作推理脚手架（"已确认事实 f1(c12)、f3(c7)…沿事实链推理"）→ checker 验证 → ungrounded 声明定点检索 + 约束重生成，修复上限 1 次（预算不足 0 次），不过则部分答案+声明缺口返回。**检索修复归 loop（answer 前），生成修复归 cot_reflect（answer 后）**

## 11. 多模态：multimodal_top_k 在 agent 路径退役

### 11.1 原机制的本质缺陷

配额同时代理两个真实约束（上下文成本、VLM 多图退化），但对"图与问题是否相关"零判断力。

### 11.2 替代：三道门 + 预算定价

关键事实：图片 chunk 的 content 是入库时 VLM 生成的描述 → canonical_score 本来就在为"图的内容 vs 问题"打分（配额时代被浪费的信号）。

```
门1 意图门：visual_intent==true（§9.1 合并调用产出，零额外调用）；否则图片仅以描述文字存在
门2 相关门：chunk 凭 canonical score 应进上下文，且支撑 ≥1 fact 或分数过线
门3 预算门：每图按估算 vision token 计价进装填器统一优化
安全护栏：max_images_in_prompt=6（针对 VLM 多图退化，非相关性机制）
```

`max_images_in_prompt` 是**每次生成调用（per-generation-call）**的限制：direct/cot_reflect 生成调用 ≤6；map_reduce 下 map 调用 0 张（纯文本）、reduce 调用 ≤6；inspect_image 单次调用同受此上限约束。非 session 级、非每轮累计。

### 11.3 inspect_image 工具

```json
{"action": "inspect_image", "params": {"chunk_ids": ["c12"], "question": "从图中读出2023年数值"}}
```

VLM 定向看图，提取事实以**文字**进证据池与账本（T2/2 点）。解决"桥事实在图里"的多跳场景；无关图片最多被看一眼即否决，不污染最终 prompt。此通路是目标架构 PerceptualMemory 的雏形。

### 11.4 兼容

现有 `/query`（v2、`aquery_vlm_enhanced`）原样保留 `multimodal_top_k`，API 契约不破坏。三重验证/base64/交错消息构件抽共享函数复用（自 `query.py` 现有逻辑，不重写）。

## 12. ModelPool：角色模型分离

### 12.1 角色与默认映射

```
角色：planner / generator / rewriter / grader / checker / summarizer
默认：全部 → 现有唯一端点（Qwen3-VL-30B, :8001）
可选：RAGANYTHING_JUDGE_API_BASE + RAGANYTHING_JUDGE_MODEL
      → grader/checker/rewriter/summarizer 切 judge 端点；planner/generator 留大模型
```

判别角色 prompt 从第一天按小模型上限设计（严格 JSON、短输出、无长推理）→ 接小模型零 prompt 改动。

### 12.2 单模型去 bias（锚定外部证据 + 代码裁决）

- 判别角色 temperature=0；generator 保持生成配置
- **checker 强制引文**：先机械拆解答案为原子声明，逐条要求从 chunk **逐字引用**支撑片段，引不出标 unsupported；**代码字符串匹配验证引文确在 chunk 中**——LLM 提案、代码裁决，伪造引文无法通过
- 信息隔离：checker 只见 (query, answer, chunks)，不见 thought/账本/trace
- 已知局限（明示）：同模型同盲点的误差相关性单模型下无解，仅靠小模型接入对冲

### 12.3 回退与熔断

- 启动健康检查：judge 不可达 → 警告 + 全角色回主端点，服务照常起
- 单次回退：judge 调用超时/连接错误 → 该次即时走主端点并记录
- 熔断：连续 5 次回退 → 角色映射整体回主端点，60s 探活，恢复自动切回

## 13. Trace 契约

`trace.py` 输出 v2 同构字段（retrieval_steps/grader_events/hallucination_events/...），新增 `agent_decisions`（每步 thought + action + 预算快照）与 `reclassify_events`。前端 AgenticTrace 不改即用，决策链展示为后续增量。

## 14. 评测与切换

### 14.1 A/B 设计

复用 `evaluate_local/`（DocBench、MultiHopQA 2wiki/musique、SurGE、KG_Eval、`run_ablation_evals.py`）与预构建 workspace：同 workspace 同题库，`agentic_mode=v2` vs `agent` 各跑一遍，评测模式关墙钟护栏。

### 14.2 指标

| 组 | 指标 | 用途 |
|---|---|---|
| 质量 | 答案正确率（沿用各 benchmark 评分）、grounded 率、不可答正确拒答率 | 不退步 |
| 成本 | 每问检索次数 / rerank 对数 / LLM 调用与 token / 墙钟 p50、p95 | 显著改善 |
| 行为 | 快速通道命中率、改判率、RecoveryPolicy 降级率、预算利用分布、终审触发率、unverifiable 标记率 | 验证设计假设、定位问题模块 |

**降级查询的统计处理**（防止 fallback 污染对比）：每条查询的 trace 标记是否触发过 RecoveryPolicy 降级。质量报告**分层呈现**：纯 v3（无降级）/ v3 含降级（ITT 口径）/ v2 三列。注意**不采用"剔除降级查询"的做法**——降级非随机发生（难题更易触发），剔除等于从 v3 卷面撕掉最难的题，会反向人为拔高纯 v3 指标。分层用于诊断，门槛判定见 §14.3。

### 14.3 切换判据（写死）

```
质量门槛：DocBench 与 MultiHopQA 正确率均 ≥ v2 基线（容差 -1pp），ITT 口径（含降级查询）
        ——上线后的真实表现就含降级，质量门槛必须按真实口径
降级率门槛：RecoveryPolicy 降级率 ≤ 10%
        ——防止 v3 靠 v2 的大脑考试及格；降级率本身是 30B 决策能力的直接度量，
          超限说明该修决策 prompt 而非该上线
成本门槛：每问 LLM token 降 ≥30%，墙钟 p95 降 ≥40%（v2 同批实测对照）
新能力门槛：自建 ~30 题多轮 session 测试集（指代/追问/跨轮复用/中途取消改问），人工评审通过
```

四关全过 → `agentic_mode` 默认切 `agent`；v2 保留至少一个版本周期作回退 flag。任一不过 → 行为组指标定位模块，修后重跑。

## 15. 与目标记忆架构的关系

本设计是记忆增强 Agent 迁移计划中"Agent 前端层"的落地形态（新增 + 替换，而非纯新增）：

- SessionMemory = WorkingMemory 第一个真实用例（TTL 内存实现 + dump/load 预留存储抽象层对接）
- session 级 evidence_cache = retrieved_context_cache
- inspect_image + 图片描述通路 = PerceptualMemory 雏形
- EpisodicMemory/SemanticMemory 的写入路径为后续 Phase，本设计不阻塞也不依赖

## 16. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| 决策调用质量不达预期（30B 模型规划能力） | RecoveryPolicy 确定性降级保底；行为组指标监控降级率 |
| 同模型同盲点 | 已知局限，明示；judge 端点就绪后自然缓解 |
| 成本点初值偏差 | 两阶段标定，trace 实测校准 |
| 快速通道误伤复杂问题 | 首 grade 不过即落入完整 loop，代价一次 T1 检索 |
| session 缓存与治理删除竞态 | 删除 callback 同步清缓存；缓存查询时二次校验 chunk 存在性（可选加固） |
