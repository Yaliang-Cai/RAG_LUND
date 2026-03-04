## 固定执行流程（2026-02-27 新增，必须遵守）

1. 修改前先读本文件
- 先阅读本.md，确认当前阶段、保留项和禁用项。

2. 修改后必须做逻辑级测试（不能只看编译）
- 必测：主链路、一致性、边界、反例。

3. 测试方式
- 优先内联测试，不落地临时文件；若创建临时文件，测试后删除。

4. 完成标准
- 语法通过 + 逻辑通过 + 边界/反例通过，三者同时满足才算完成。

5. 先查现有功能，再做增量完善
- 修改前先检索同名/相近能力（如 normalize、filter、repack、token 预算），确认是否已实现。
- 若已有实现，优先复用并最小改动；禁止平行新增冗余逻辑。

# RAGAnything LocalRAG VLM 重构记录

## 防错原则（必须遵守）

- 不能只做 `py_compile` 语法检查；必须补做“链路一致性 + 关键分支行为”检查。
- 每次改动都要验证：用边界情况，反例测试。
- 反例复盘（本项目已发生）：曾出现“编译通过，但重组逻辑与原始 prompt 形态不契合”的问题（如换行场景下占位符处理偏差）。
- 以后执行标准：参考同事做法，至少包含轻量行为断言或等价链路检查，不允许仅以“能编译”作为完成标准。




## 历史执行摘要（2026-02-20 ～ 2026-03-02）

> 说明：以下为已完成工作的压缩摘要，详细过程以 Git 历史为准。

### 已完成主线
- Query 消息链路收敛：enhanced 与 non-enhanced 组装规则对齐，修复固定索引假设错位问题。
- Qwen3-VL 适配完成：文本 tokenizer 与图像 token 估算切换到 Qwen 本地模型口径。
- 图片 token 对齐验证通过：本地估算与 vLLM prompt token 差分在样本集上对齐（diff=0）。
- `multimodal_top_k` 语义稳定：仅控制发给 VLM 的图片上限，不直接裁剪文本 chunk。
- enhanced 查询对齐官方交错发送：按 marker 位置插入 image_url，不再强制全图前置。
- 过滤链路已增强：路径碎片、低价值结构词（含 page/layout 类噪声）过滤加强，减少 index 与图谱污染。
- token 配置完成收敛：query/ingest 上限拆分落地，并与 evaluate 参数对齐。
- 历史与预算链路打通：history/image token 预算已接入查询预算，并提供官方路径回退开关。
- 存储安全修复完成：控制字符与 GraphML 安全写入问题已处理。

### 当前状态
- 2026-03-02 之前的执行项已完成并复核。
- 下文仅保留待计划事项与后续增量记录。

## 待计划补充（2026-03-02，Index 健康检查：重复/孤儿/悬挂）

### 背景
- 今天讨论过 index 侧“数据健康”问题，但尚未形成最终实现。
- 当前文档已有 `entity_key` 方向（去重）计划；本节补充“图结构健康检查”维度。

### 待讨论问题（挂起）
1. 重复节点（duplicates）
- 同义实体分裂、大小写/标点变体造成多节点并存。
- 需要明确“检测阶段”与“修复阶段”是否分离。

2. 孤儿节点（orphans）
- 节点存在但缺少有效边或缺少可追溯来源关联。
- 需要明确“仅告警”还是“自动清理”策略。

3. 悬挂关系（dangling edges）
- 关系边存在但 source/target 在图中不存在。
- 旧图/外部删改/异常迁移导致的历史脏数据兜底项，而不是当前 merge 主路径会常规产生的问题。

### 初步计划（未执行）
1. 先做体检报告，不自动修复
- 输出 duplicates/orphans/dangling 的计数和样本 ID。
- 将报告挂到 index 后处理日志与统计文件。

2. 再做可回滚修复（第二阶段）
- 去重合并、孤儿清理、悬挂边修复分别独立开关。
- 所有修复操作保留 before/after 快照与回滚入口。

### 当前状态
- 本节仅为待计划/待讨论挂起，尚未提交对应代码改动。

## 待计划补充（2026-03-03，vLLM 批处理参数与 MAX_ASYNC 联调）

### 参数定义（先统一口径）
- `--max-num-batched-tokens`：
  - vLLM scheduler 单轮可处理的总 token 上限（跨请求合计，包含 prefill/decode）。
  - 调大后通常提升吞吐，但会增加显存压力与尾延迟抖动风险。
- `--max-num-seqs`：
  - vLLM 同时调度的序列（请求）上限。
  - 调大后可提高并发处理能力，但会带来 KV cache/调度竞争压力。
- `MAX_ASYNC`（LightRAG API 侧）：
  - 应用层并发请求上限（发往 LLM/VLM 的并发数）。
  - 它不是 GPU 并行度本身，最终仍受 vLLM 的 `--max-num-seqs` 与 token 预算约束。

### 三者关系（关键）
- 有效并发近似受 `min(MAX_ASYNC, --max-num-seqs)` 约束。
- 若 `--max-num-batched-tokens` 过小，即使 `--max-num-seqs` 很大，也会因 token 预算不足导致排队。
- 若 `MAX_ASYNC` 远大于 `--max-num-seqs`，会把排队前移到 vLLM 服务侧，表现为请求堆积和 p95/p99 变差。

### 待计划联调步骤（未执行）
1. 配置入口
- 服务侧（`start_server_qwen3_vl.sh`）新增：
  - `--max-num-batched-tokens 16384`
  - `--max-num-seqs 8`
- 应用侧（`lightrag/api/config.py`）先将 `MAX_ASYNC` 从 4 提到 6，观察后再评估到 8。

2. 观察指标
- 吞吐：requests/s、tokens/s。
- 延迟：p50/p95/p99、超时率、400/500 比例。
- 资源：GPU 显存占用、OOM、KV cache 紧张告警。

3. 调参顺序
- 先固定 `--max-num-batched-tokens=16384`，只调 `MAX_ASYNC`（4 -> 6 -> 8）。
- 若吞吐无提升且延迟上升，回退 `MAX_ASYNC`。
- 若并发受限明显，再评估 `--max-num-seqs` 与 `--max-num-batched-tokens` 联动上调。

### 当前状态
- 本节为待计划补充，尚未提交参数改动与压测结果。

## 待计划补充（2026-03-03，Stage 3.5 主实体去噪与同名防污染）

### 背景
- 当前 Stage 3.5 会将多模态主实体直接写入 graph / entities_vdb / full_entities。
- 该路径不经过 `operate.py` 抽取阶段的低质量实体过滤。
- 主实体主键以 `entity_name` 为全局键，跨文档同名时存在合并污染风险。

### 问题 A：`page_number` 类主实体优化
- 典型高风险：`Page Number 1 (page_number)` 这类布局元信息，跨文档重复概率高、检索价值低。

### 待计划方案（未执行）
1. Stage 3.5 直写前增加轻量过滤（仅针对主实体）
- 新增主实体过滤函数（建议复用/对齐 `operate.py` 的 `page_number_label`、`layout_metadata` 口径）。
- 默认过滤：`page_number`、纯布局元信息（header/footer token、坐标类标签）。
- 保留：image/table/equation 的语义主实体。

2. 可观测性
- 记录主实体过滤计数与样本（按 `doc_id` 汇总）：
  - `stage35_filtered_count`
  - `stage35_filtered_examples`

### 问题 B：`full_entities` 覆盖导致维护链路风险
- 按 doc 删除/重建时，`full_entities` 不完整会导致受影响实体识别不全。主实体只在vdb_entities里，kv_store_full_entities里没有。同样kv_store_full_relations里也米有，注意排查其他kv_store里有没有影响。kv_store_text_chunks里有，因为是生成多模态chunk的肯定存在多模态主实体。
- 可能留下陈旧实体/关系引用，后续维护链路排障成本上升。
- 存储一致性变差：graph / entities_vdb / full_entities 三者可能出现不一致。

### 问题 C：同名实体跨文档污染
- 风险：同名实体在图中被当作同一全局实体，`source_id/file_path` 混杂。

### 问题 D：`context_window=1` 造成主实体语义跨页漂移
- 当前多模态描述生成默认 `context_mode=page`、`context_window=1`。
- 这意味着主实体命名/摘要会使用“前一页 + 当前页 + 后一页”的文本上下文，而不是严格当前页。
- 对 `page_number/footer/footnote` 等布局型对象，易引入跨页噪声，增加命名不稳定与同名冲突概率。

### 待计划方案（未执行）
1. 增加对照开关（仅影响多模态描述阶段）
- 方案 A：保持现状（`context_window=1`）。
- 方案 B：严格当前页（`context_window=0`）。
- 评估维度：主实体稳定性、同名率、检索准确率、人工可解释性。

2. 分类型上下文策略（可选）
- 对布局型对象（`page_number/footer/footnote`）默认使用 `context_window=0`。
- 对语义型对象（image/table/equation）保留 `context_window=1`。

3. 观测指标
- `stage35_entity_name_collision_rate`
- `stage35_page_scoped_name_consistency`
- 采样记录“主实体名 + page_idx + doc_id + context_window”用于复盘。

### 当前状态
- 本节为待计划补充，尚未提交对应代码改动。

## 增量更新（2026-03-03，tokenizer 环境变量收敛）

- 结论：tokenizer 环境变量已收敛为 `TOKENIZER_MODEL_PATH -> VISION_MODEL_PATH`。
- 价值：移除项目私有冗余变量，降低误配置与排障成本。
- 校验：`local_rag.py` 中 `RAGANYTHING_TOKENIZER_MODEL_PATH` 已清理，语法检查通过。

## 增量更新（2026-03-03，enhanced 图文交错对齐官方实现）

- 结论：enhanced 消息已改为 marker 驱动的图文交错（官方语义），不再“全图前置”。
- 行为：无效 marker 保留文本；无 marker 直接 fail-fast，避免静默退化。
- 校验：语法与关键分支断言通过（顺序交错、越界 marker、无 marker 异常）。
---

## 增量更新（2026-03-04，默认预算回退 + JSON 严格输出 + 抽取前后协同降噪）

- Query token 预算默认行为已回退为官方 `available_chunk_tokens` 路径：
  - `DEFAULT_ENABLE_IMAGE_TOKEN_BUDGET=False`；
  - `operate.py` 两条链路（KG/naive）缺省值对齐为 `False`，避免漏参时误走图片联合预算分支；
  - history token 仍计入固定开销预算。

- VLM `json_schema` 功能保留（含 fallback），未回退：
  - ingest 结构化输出仍由 `local_rag.py` 的 schema 分支控制；
  - `strict=False` 保持兼容优先。

- 官方 `raganything/prompt.py` 已并入严格 JSON 输出句（不新建 Qwen 专用 prompt 分叉）：
  - 覆盖 8 个模态分析 prompt（with/without context 全覆盖）；
  - 采用公共前后缀拼接，避免重复维护；
  - 关键句使用 `{{` / `}}` 转义，保证后续 `.format(...)` 不误解析占位符。

- KG 抽取“前置约束 + 后置过滤”协同增强：
  - system prompt 增加广义定位标签规则、泛词无指代过滤、关系软约束；
  - 后置 filter 扩展到广义 locator 裸标签，且保留“带语义标题”的高价值实体；
  - 示例：`Table 7: Ablation Results (table)` 会保留；`Table 7` / `Figure 2` / `Section 3` / `Ref [12]` 会过滤。

---

## 架构备忘：Query 完整链路（2026-02-26）

> 纯参考性记录，无代码改动。覆盖从 HTTP 请求入口到最终 LLM 消息返回的每一层。

---

### 概览：10 层调用栈

```
HTTP POST /query
  └─ [L1]  server/app.py:208          QueryRequest 验证 + 参数钳位
       └─ [L2]  local_rag.py:751      LocalRagService.query()
            └─ [L3]  query.py:105     QueryMixin.aquery()
                 │
                 ├─ vlm=False
                 │    └─ [L4]  lightrag.py:2422   LightRAG.aquery()
                 │         └─ [L5]  lightrag.py:2684   aquery_llm()
                 │              └─ [L6]  operate.py:3100   kg_query()
                 │                   ├─ 关键词提取（一次 LLM 调用）
                 │                   ├─ [L7]  operate.py:4138   _build_query_context()
                 │                   │    ├─ Stage 1: _perform_kg_search()
                 │                   │    ├─ Stage 2: _apply_token_truncation()
                 │                   │    ├─ Stage 3: _merge_all_chunks()
                 │                   │    └─ Stage 4: _build_context_str()
                 │                   │         └─ [L8]  utils.py:2702   process_chunks_unified()
                 │                   └─ [L9]  operate.py:3252   use_model_func()
                 │                        └─ [L10] local_rag.llm_model_func → vLLM text API
                 │
                 └─ vlm=True
                      └─ query.py:317   aquery_vlm_enhanced()
                           ├─ lightrag.aquery(only_need_prompt=True) → raw_prompt（走 L4-L8）
                           ├─ _process_image_paths_for_vlm() → base64（上限 multimodal_top_k=3）
                           ├─ _build_vlm_messages_with_images() → 交错多模态 messages
                           └─ [L10'] local_rag.vision_model_func → vLLM Vision API
```

---

### L1：HTTP 入口（`server/app.py`）

**QueryRequest 模型**（行 93–100）：
```python
class QueryRequest(BaseModel):
    doc_id: str
    query: str
    mode: str = "hybrid"                    # API 默认；LightRAG 内部默认为 "mix"
    top_k: int = DEFAULT_TOP_K              # = 20
    chunk_top_k: int = DEFAULT_CHUNK_TOP_K  # = 10
    enable_rerank: bool = True
    vlm_enhanced: bool = True
```

**路由处理器**（行 208–225）：
1. `verify_api_key` — 校验请求头 `X-API-Key`
2. 参数钳位：`top_k = max(1, min(top_k, MAX_TOP_K))`，`chunk_top_k` 同理
   - `MAX_TOP_K` / `MAX_CHUNK_TOP_K` 读自环境变量，默认等于常量值（20 / 10）
3. `await service.query(doc_id, query, mode=..., top_k=..., chunk_top_k=..., enable_rerank=..., vlm_enhanced=...)`
4. 返回 `{"answer": result}`（非流式，纯 JSON）

---

### L2：Service 层（`local_rag.py:751`）

```python
async def query(self, doc_id: str, query: str, **kwargs) -> str:
    rag = await self.get_rag(doc_id)   # 按 doc_id 缓存 RAGAnything 实例
    return await rag.aquery(query, **kwargs)
```

`get_rag()` 以 `{working_dir_root}/{doc_id}` 为工作目录构建并缓存 `RAGAnything` 实例，注入 `llm_model_func`、`vision_model_func`、`embedding_func`、`rerank_model_func`。参数通过 `**kwargs` 原样透传，不做额外处理。

---

### L3：QueryMixin.aquery()（`raganything/query.py:105`）

1. `_ensure_lightrag_initialized()` — 检查 LightRAG 实例已就绪
2. 补齐常量默认值：
   ```python
   kwargs.setdefault("top_k", DEFAULT_TOP_K)              # 20
   kwargs.setdefault("chunk_top_k", DEFAULT_CHUNK_TOP_K)  # 10
   ```
3. **VLM 决策树**（行 138–161）：
   ```
   vlm_enhanced 参数
   ├─ 显式 True  + vision_model_func 可用 → aquery_vlm_enhanced()
   ├─ 显式 False                           → 普通文本查询
   └─ None（未传）                          → 根据 vision_model_func 是否存在自动判定
   ```
4. 普通路径：`QueryParam(mode=mode, **kwargs)` → `self.lightrag.aquery(query, param=query_param)`

---

### L4–L5：LightRAG.aquery() / aquery_llm()（`lightrag/lightrag.py:2422 / 2684`）

`aquery()` 是向后兼容包装器，调用 `aquery_llm()` 后从返回字典提取 `llm_response`：

```python
result = await self.aquery_llm(query, param, system_prompt)
llm_response = result.get("llm_response", {})
if llm_response.get("is_streaming"):
    return llm_response.get("response_iterator")   # AsyncIterator[str]
else:
    return llm_response.get("content", "")         # str
```

`aquery_llm()` 按 mode 路由（行 2711–2746）：

| mode | 调用目标 | 检索方式 |
|------|---------|---------|
| local / global / hybrid / mix | `kg_query()` | KG + 向量 |
| naive | `naive_query()` | 纯向量 |
| bypass | 直接 `use_llm_func()` | 无检索 |

---

### L6：kg_query()（`lightrag/operate.py:3100`）

**关键词提取**（行 3152–3169）：
- 调用 LLM 一次，提取 `hl_keywords`（高层语义关键词）和 `ll_keywords`（低层实体关键词）
- 结果用于后续 KG 向量索引检索

**快捷出口**：
- `only_need_context=True`（行 3193）→ 仅返回 context 字符串，不调用答案 LLM
- `only_need_prompt=True`（行 3214）→ 拼接 `sys_prompt + "\n\n---User Query---\n" + user_query` 后直接返回，不调用答案 LLM
  - **VLM 增强查询的第一步就用此标志获取 raw_prompt**

**缓存检查**（行 3240–3250）：
- 哈希键覆盖：mode + query + response_type + top_k + chunk_top_k + max_entity/relation_tokens + hl/ll_keywords + enable_rerank + user_prompt
- 命中则跳过 `_build_query_context()` 和答案 LLM 调用

**Prompt 组装**（行 3205–3212）：
```python
sys_prompt = PROMPTS["rag_response"].format(
    response_type=response_type,
    user_prompt=user_prompt,
    context_data=context_result.context,  # 来自 _build_query_context()
)
```

**答案 LLM 调用**（行 3252–3258）：
```python
response = await use_model_func(
    user_query,
    system_prompt=sys_prompt,
    history_messages=query_param.conversation_history,
    enable_cot=True,
    stream=query_param.stream,
)
```

**返回**：
- 非流式 → `QueryResult(content=str)`
- 流式 → `QueryResult(response_iterator=AsyncIterator[str], is_streaming=True)`

---

### L7：_build_query_context()（`lightrag/operate.py:4138`）—— 4 阶段检索

#### Stage 1：`_perform_kg_search()`（行 3510–3671）

按 mode 进行 KG + 向量检索，返回实体、关系、chunk 候选：

| mode | 实体来源 | 关系来源 | chunk 额外来源 |
|------|---------|---------|---------------|
| local | 向量检索 ll_keywords → 邻域实体 | 实体的邻接关系 | 无额外 |
| global | 关系端点 | 向量检索 hl_keywords | 无额外 |
| hybrid | local + global 合并 | local + global 合并 | 无额外 |
| mix | local + global 合并 | local + global 合并 | `_get_vector_context()` 纯向量 chunk |

mix 模式同时调用 `_get_vector_context()` 做纯向量 chunk 检索（`chunks_vdb.query(query, top_k)`），与 KG chunk round-robin 合并。

同时返回：
- `query_embedding`（预计算一次，后续合并复用）
- `chunk_tracking`（每个 chunk 的来源标记）

#### Stage 2：`_apply_token_truncation()`（行 3679–3764）

按 `max_entity_tokens`（默认 6000）、`max_relation_tokens`（默认 8000）截断实体和关系列表，返回：
- `entities_context` / `relations_context`：截断后格式化为 JSON-lines 字符串（供 Stage 4 使用）
- `filtered_entities` / `filtered_relations`：截断后的对象列表（供 Stage 3 关联 chunk 使用）

#### Stage 3：`_merge_all_chunks()`（行 3850–3949）

1. 从过滤后实体获取关联 chunk（`_find_related_text_unit_from_entities`，支持 WEIGHT / VECTOR 两种选择策略）
2. 从过滤后关系获取关联 chunk（`_find_related_text_unit_from_relations`）
3. Round-robin 合并：`vector → entity_chunk → relation_chunk` 循环交替取块，去重后形成候选池
4. **每个 dict 重建处均携带 `is_multimodal` 字段**（2026-02-25 T1 修复，4 处：`_get_vector_context` L3496 + `_merge_all_chunks` 三处 L3911/L3926/L3941）

#### Stage 4：`_build_context_str()`（行 3955–4134）

**Token 预算计算**（行 4021–4038）：
```
available_chunk_tokens = max_total_tokens（默认 30000）
    - sys_prompt_tokens（模板 + response_type + user_prompt，context 留空计算）
    - kg_context_tokens（截断后 entity + relation 真实 token 数）
    - query_tokens（用户问题）
    - buffer_tokens（固定 200）
```

调用 `process_chunks_unified()` 进行最终 chunk 筛选，再用 `kg_context_template` 格式化：
```
Knowledge Graph Data (Entity):
{entities_str}                 ← JSON-lines

Knowledge Graph Data (Relationship):
{relations_str}                ← JSON-lines

Document Chunks:
{text_chunks_str}              ← 截断后的 chunk 内容

Reference Document List:
{reference_list_str}           ← DC1、DC2、... 参考列表
```

---

### L8：process_chunks_unified()（`lightrag/utils.py:2702`）

从候选 chunk 池到最终 context chunk 列表，共 5 步：

**Step 1 — Rerank**（行 2730–2744）：
- `enable_rerank=True` 时调用 CrossEncoder（本地 `bge-reranker-v2-m3`）
- `multimodal_top_k` 激活时对**全量** chunk 重排（`rerank_top_k = len(unique_chunks)`），保证文本候选充足；否则仅对前 `chunk_top_k` 个重排

**Step 2 — Rerank 分数过滤**（行 2746–2769）：
- `min_rerank_score=0.5`（来自 global_config）
- 低于阈值的 chunk 过滤掉

**Step 2.5 — 多模态预算分配**（行 2771–2811，`multimodal_top_k ≠ None` 时激活）：
```
top_window     = unique_chunks[:chunk_top_k]   # 类型无关，取前 10（rerank 排名）
remaining_pool = unique_chunks[chunk_top_k:]   # 第 11 名以后备用

selected_mm    = top_window 内 is_multimodal=True 的 chunk（数量不限）
text_in_window = top_window 内 is_multimodal=False 的 chunk
text_budget    = chunk_top_k - multimodal_top_k   （= 10 - 3 = 7）
selected_text  = text_in_window[:text_budget]
  └─ 不足 text_budget 时从 remaining_pool 中按 rerank 顺序补充文本 chunk

final = sorted(selected_mm + selected_text, by rerank_score desc)
budgets_applied = True
```

> 图片数量上限由 `query.py` 的 `_process_image_paths_for_vlm(max_images=multimodal_top_k)` 独立控制，超限的 `Image Path:` 保留原文不附图。

**Step 3 — chunk_top_k 截断**（行 2813–2820）：
- `budgets_applied=True` 时跳过此步（已由 Step 2.5 控制数量）

**Step 4 — Token 截断**（行 2822–2848）：
- `truncate_list_by_token_size()` 按 `available_chunk_tokens` 限制总 token 数

**Step 5 — 添加 ID**（行 2850–2857）：
- 依次打 `DC1`、`DC2`、... 便于 reference list 引用

---

### L9–L10：LLM 实际调用（`local_rag.py` 自定义 `llm_model_func`）

`build_llm_model_func()` 构建的闭包被 `kg_query` 以 `use_model_func(user_query, system_prompt=..., ...)` 调用时，按内部参数分三条路径：

**路径 A — 关键词提取**（`keyword_extraction=True`）：
- 设置 `response_format=GPTKeywordExtractionFormat`
- 调用 `client.chat.completions.parse(...)` → 序列化为 JSON 字符串返回（与上游 `json_repair.loads()` 兼容）

**路径 B — 文本 query 答案**（无 messages 参数，keyword_extraction=False）：
- `_try_repack_text_query(system_prompt, prompt)` 拆分 LightRAG 结构：
  - `---Role--- / ---Instructions---` 部分 → `system` 消息
  - `---Context--- / User Question:` 部分 → `user` 消息
- 调用 `text_client.chat.completions.create(model, messages, max_tokens=query_max_tokens=2048, ...)`
- 返回 `response.choices[0].message.content`

**路径 C — VLM query**（messages 参数有值，来自 `aquery_vlm_enhanced` 预构建）：
- 直通调用：`vision_client.chat.completions.create(model, messages, max_tokens=query_max_tokens=2048, ...)`
- 返回 `response.choices[0].message.content`

---

### VLM 增强路径详解（`query.py:317`）

```
aquery_vlm_enhanced(query, mode, **kwargs)
  │
  ├─ 1. kwargs.setdefault("multimodal_top_k", 3)
  │
  ├─ 2. QueryParam(only_need_prompt=True, multimodal_top_k=3, ...)
  │      → lightrag.aquery() 走完 L4–L8 后跳过 LLM，直接返回 prompt 字符串
  │        （格式：sys_prompt + "\n\n---User Query---\n" + user_query）
  │
  ├─ 3. _process_image_paths_for_vlm(raw_prompt, max_images=3)
  │      - 正则扫描 r"Image Path:\s*([^\r\n]*\.(?:jpg|png|...))"
  │      - validate_image_file()：文件存在 + 扩展名合法 + 大小 ≤ 50 MB
  │      - base64 编码；达到 max_images=3 后剩余路径保留原文，不附图
  │      - 路径行后追加 [VLM_IMAGE_n] 标记，返回 enhanced_prompt
  │
  ├─ 4. _build_vlm_messages_with_images(enhanced_prompt, query, system_prompt)
  │      - 按 [VLM_IMAGE_n] 标记分割文本
  │      - 构建交错 content_parts：[{text}, {image_url:base64}, {text}, ...]
  │      - messages = [
  │          {"role": "system", "content": full_system_prompt},
  │          {"role": "user",   "content": content_parts},
  │        ]
  │
  └─ 5. vision_model_func("", messages=messages)
         → vision_client.chat.completions.create(model, messages, max_tokens=2048)
         → return response.choices[0].message.content
```

---

### Prompt 模板速查（`lightrag/prompt.py`）

**标准文本 RAG**（`PROMPTS["rag_response"]`，行 224–276）：
```
---Role---
You are an expert AI assistant specializing in synthesizing information...

---Instructions---
1. Step-by-Step Instruction: ...
...
6. Additional Instructions: {user_prompt}

---Context---
{context_data}
```

`local_rag.py` 的 `_try_repack_text_query` 将上述结构拆分为 system（Role+Instructions）和 user（Context+问题）两条消息。

**KG Context 模板**（`PROMPTS["kg_query_context"]`，行 332–357）：
```
Knowledge Graph Data (Entity): {entities_str}
Knowledge Graph Data (Relationship): {relations_str}
Document Chunks: {text_chunks_str}
Reference Document List: {reference_list_str}
```

**Naive RAG**（`PROMPTS["naive_rag_response"]`）：结构类似，去掉 KG 数据，仅含向量检索 chunks。

---

### QueryParam 关键字段（`lightrag/base.py:85`）

| 字段 | 默认值 | 来源 / 说明 |
|------|--------|------------|
| `mode` | `"mix"` | HTTP QueryRequest 传入（API 默认 `"hybrid"`） |
| `top_k` | 20 | HTTP → service → aquery（rag-anything 常量）|
| `chunk_top_k` | 10 | HTTP → service → aquery（rag-anything 常量）|
| `enable_rerank` | `True` | HTTP 传入 |
| `multimodal_top_k` | `None` → `3`（VLM时）| `aquery_vlm_enhanced` 注入，控制发给 VLM 的图片上限 |
| `max_entity_tokens` | 6000 | LightRAG 内部（env/常量）|
| `max_relation_tokens` | 8000 | LightRAG 内部（env/常量）|
| `max_total_tokens` | 30000 | LightRAG 内部（env/常量）|
| `only_need_prompt` | `False` | VLM 增强第一步置为 `True`，获取 raw_prompt |
| `stream` | `False` | HTTP 层未暴露；内部支持 |
| `conversation_history` | `[]` | HTTP 层未暴露；内部支持多轮对话 |

---

### 参数传递路径

```
HTTP QueryRequest
    ↓ 参数钳位（app.py:214–215）
    ↓ max(1, min(val, MAX_VAL))
LocalRagService.query(**kwargs)
    ↓ 透传
RAGAnything.aquery(**kwargs)
    ↓ setdefault top_k / chunk_top_k
QueryParam(**kwargs)
    ↓
kg_query(query_param)
    ↓
process_chunks_unified(query_param)
    ↓
use_model_func(system_prompt=sys_prompt)
    ↓
vLLM API → LLM 回答
    ↓
return str → {"answer": "..."}
```

---

### 最终 HTTP 响应格式

**非流式**（当前 server/app.py 唯一暴露的模式）：
```json
{"answer": "<LLM 生成的文本>"}
```

**流式**（LightRAG 内部支持，通过 `QueryParam.stream=True` 启用，当前 HTTP 层未暴露）：
LightRAG 的 `query_routes.py` 提供 NDJSON 格式：
- 首行（可选）：`{"references": [...]}`
- 内容块：`{"response": "chunk content"}`
- 错误：`{"error": "message"}`

---

### 关键文件速查

| 层级 | 文件 | 行号 | 职责 |
|------|------|------|------|
| HTTP 入口 | `server/app.py` | 93–100, 208–225 | QueryRequest 定义 + 路由 + 参数钳位 |
| Service | `raganything/services/local_rag.py` | 751–753 | 参数透传 + RAGAnything 实例缓存 |
| Query 决策 | `raganything/query.py` | 105–175 | VLM 决策树 + QueryParam 构建 |
| VLM 增强 | `raganything/query.py` | 317–383 | raw_prompt → 图像提取 → 多模态消息 |
| VLM 图像处理 | `raganything/query.py` | 552–722 | `_process_image_paths_for_vlm` + `_build_vlm_messages_with_images` |
| LightRAG 入口 | `lightrag/lightrag/lightrag.py` | 2422, 2684 | `aquery()` 包装 + `aquery_llm()` mode 路由 |
| KG 查询主逻辑 | `lightrag/lightrag/operate.py` | 3100–3307 | 关键词提取 + prompt 组装 + 缓存 + LLM 调用 |
| 4 阶段检索 | `lightrag/lightrag/operate.py` | 4138–4256 | `_build_query_context()` |
| Context 字符串构建 | `lightrag/lightrag/operate.py` | 3955–4134 | token 预算 + `process_chunks_unified` + 格式化 |
| Chunk 处理 | `lightrag/lightrag/utils.py` | 2702–2857 | rerank + 预算分配 + token 截断 + ID 分配 |
| Prompt 模板 | `lightrag/lightrag/prompt.py` | 224–357 | `rag_response` / `kg_query_context` / `naive_rag_response` |
| LLM 调用（自定义）| `raganything/services/local_rag.py` | `build_llm_model_func` | keyword / text / VLM 三条路径 |
| 默认常量 | `raganything/constants.py` | — | 所有默认值统一出口 |

---

## 增量更新（2026-03-03，WebUI v2 三面板重设计 + 后端 API 扩展）

### 改动范围

仅修改两个文件：`server/app.py`、`server/templates/index.html`。不涉及 `raganything/` 或 `lightrag/` 核心代码。

### 后端新增端点（`server/app.py`）

**三层存储**：`uploads/{doc_id}/`（原件）+ `output/{doc_id}/`（解析产物）+ `rag_workspace/{doc_id}/`（KG + 向量）

| 端点 | 说明 |
|------|------|
| `GET /graph/{doc_id}/labels` | 调用 `rag.lightrag.get_graph_labels()` |
| `GET /graph/{doc_id}/subgraph?label=&max_depth=2&max_nodes=50` | 调用 `get_knowledge_graph()`，序列化 `KnowledgeGraphNode/Edge` |
| `GET /graph/{doc_id}/stats` | 通过 networkx 读取 graphml，返回实体/关系数量 |
| `GET /graph/{doc_id}/search?q=&limit=20` | 调用 `search_labels()` 模糊搜索 |
| `DELETE /workspace/{doc_id}` | 删除三层目录 + 清除 `_rag_instances` 缓存 |
| `GET /workspace/{doc_id}/stats` | 文件数、实体/关系数、磁盘占用 |
| `GET /workspaces` | 增强版：合并三层目录 doc_id，含 `uploaded_files` 列表 |
| `GET /uploads/{doc_id}/{filename}` | 静态服务上传的 PDF 原件 |

**`/query` 增强**：两阶段查询（`aquery_data()` 结构化检索 + `service.query()` LLM 生成），返回 `entities`、`relationships`、`chunks`、`references`、`metadata`、`graph`（可选）。

### 前端重写（`server/templates/index.html`，~700 行）

**布局**：三面板——左侧 PDF 阅读器（48%）+ 右侧聊天区 + 底部知识图谱面板（可折叠）

**CDN 依赖**：
- marked 4.3.0（Markdown）
- KaTeX 0.16.9（LaTeX 公式）
- highlight.js 11.9.0（代码高亮）
- PDF.js 3.11.174（PDF 渲染，IntersectionObserver 懒加载）
- graphology 0.25.4 + sigma 2.4.0（知识图谱可视化，自定义 Fruchterman-Reingold 力导向布局）

**11 个 JS 模块**：AppState、API、PDFViewer、Renderer、ReasoningTrace、Citations、Chat、GraphViewer、FileManager、UI、Init

**核心交互**：
- PDF 页面懒渲染 + 缩放 + 引用点击跳转对应页
- 推理过程 4 步动画（关键词提取 → KG 搜索 → 重排序 → 生成）
- 图谱节点点击显示详情面板 + 实体搜索
- 渲染管线：marked → KaTeX auto-render → hljs → 引用链接替换 → 图片点击放大

**设计延续**：琥珀色点缀、DM Serif Display + IBM Plex Sans/Mono 字体、深色/浅色主题 CSS 变量切换

### 启动方式

```bash
export RAGANYTHING_API_KEY=your_key
cd rag-anything
uvicorn server.app:app --host 0.0.0.0 --port 9621
# 浏览器打开 http://localhost:9621
```

## 增量更新（2026-03-04，Rerank-first 联合预算打包 + 官方回退开关）

### 摘要
- 预算逻辑改为“Rerank-first + 联合预算前缀打包”，解决“预扣图片 token 与最终 chunk 错位”问题。
- 保持原位实现：KG 与 naive 各自处理，不引入额外统一入口。
- 增加开关 `enable_image_token_budget`：
  - `False`：回退官方 `available_chunk_tokens` 截断路径；
  - `True`：按 chunk 顺序联合计算 text + new image token，超预算即停（不后跳）。
- 默认值收敛：`DEFAULT_ENABLE_IMAGE_TOKEN_BUDGET` 放在 `lightrag/constants.py`，`base.py` 仅做 env 覆盖读取。

### 校验
- 语法：`operate.py/base.py/constants.py` 编译通过。
- 逻辑断言：开关开/关两条预算分支行为均符合预期。

## 增量更新（2026-03-04，默认模型切换到 Qwen3.5-35B-A3B-FP8）

### 摘要
- 默认模型入口统一切换为 `Qwen/Qwen3.5-35B-A3B-FP8`，避免 `constants` 默认值与当前部署脚本不一致。
- `local_rag.py` 的 tokenizer 报错文案改为通用 Qwen 目录提示，不再写死 Qwen3-VL。
- `evaluate_local/DocBench/evaluate.py` 的默认生成模型、模型路径、启动说明统一切换为 Qwen3.5。

### 变更文件
- `raganything/constants.py`
  - `DEFAULT_VISION_MODEL_PATH` -> `/data/y50056788/Yaliang/models/Qwen3.5-35B-A3B-FP8`
  - `DEFAULT_LLM_MODEL_NAME` -> `Qwen/Qwen3.5-35B-A3B-FP8`
- `raganything/services/local_rag.py`
  - tokenizer 加载失败提示从 “Qwen3-VL model directory” 改为 “Qwen model directory”
- `evaluate_local/DocBench/evaluate.py`
  - `RAG_MODEL_NAME` / `RAG_VISION_MODEL_PATH` / 提示文案与 quick workflow 同步到 Qwen3.5

### 校验
- 仅做默认值与文案收敛，未引入新分支逻辑。
- 目标是”默认即当前生产模型”，减少环境变量漏配时的错模型风险。

---

## 待计划补充（2026-03-04，WebUI 离线静态资源托管）

### 背景
- WebUI（`server/templates/index.html`）目前依赖 8 个外部 CDN 资源，断网环境下全部加载失败。
- 后端（vLLM 推理、RAG、Embedding、Rerank）均为本地部署，不受网络影响。
- 仅前端 UI 显示质量下降，核心功能（上传、入库、查询）不受影响。

### 当前外部依赖清单

| 资源 | CDN | 断网影响 |
|------|-----|---------|
| Google Fonts（DM Serif Display / IBM Plex Sans / Mono）| fonts.googleapis.com | 退回系统字体，视觉影响小 |
| KaTeX 0.16.9（CSS + JS + auto-render）| cdn.bootcdn.net | **公式显示原始 LaTeX 文本** |
| highlight.js 11.9.0（CSS × 2 + JS）| cdnjs.cloudflare.com | 代码无高亮，功能不影响 |
| marked.js 4.3.0 | cdn.bootcdn.net | **Markdown 答案显示原始文本** |
| PDF.js 3.11.174 | cdnjs.cloudflare.com | PDF 预览已改为 iframe，实际不影响 |

### 待实现方案（未执行）

1. **FastAPI 挂载静态目录**
   - `app.mount(“/static”, StaticFiles(directory=”server/static”), name=”static”)`
   - 在 `server/static/` 下存放各库文件

2. **一次性下载脚本**（`server/download_static.py`）
   - 联网时执行一次，把上述 CDN 资源下载到 `server/static/`
   - PDF.js worker 需单独下载

3. **HTML 改用本地路径**
   - 将 `index.html` 中所有 CDN URL 替换为 `/static/...`
   - 字体部分用系统字体 fallback 替代 Google Fonts（无需下载字体文件）

### 优先级
- 当前环境（校内服务器）有网络，暂不阻塞。
- 若迁移至完全离线环境（如 HPC 隔离区），再执行本节。

### 当前状态
- 本节为待计划，尚未提交任何代码改动。
