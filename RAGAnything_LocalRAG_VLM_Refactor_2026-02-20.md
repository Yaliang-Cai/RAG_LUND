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

- 日期：2026-02-20
- 状态：已完成（本轮）
- 范围：`local_rag.py`、`evaluate.py`、`examples/raganything_local_v2.py`


## 一、相较改造前的核心变化

### 1) 消息组织与提示词

- 改造前：query 阶段存在本地系统提示叠加与引用提醒注入，存在重复与冲突风险。
- 改造后：
  - 不再使用本地 `base_system_prompt` 兜底，也不注入 `citation_reminder`。
  - 从最终 user 文本里拆分 LightRAG 结构：
    - `---Role--- ... ---Instructions---` 进入 system
    - `---Context--- ...` 进入 user 文本
  - 问题只保留一次（`User Question` 在 user 末尾）。
  - 若结构化解析失败，回退发送原始 messages（不阻断）。

### 2) 图像来源与路径噪声清理

- 改造前：图像路径可能从不期望区域传播，且实体/关系中有路径噪声。
- 改造后：
  - 仅依据 chunk 区 `Image Path: ...` 建立路径映射与图片候选。
  - entity 区路径型实体直接删除（`Image Path` / 完整路径 / 文件名）。
  - relation 路径端点替换规则：
    - 可映射 -> `[VLM_IMAGE_n]`
    - 不可映射 -> `[IMAGE_ENTITY]`
  - chunk 区处理为“保留原路径 + 追加标记”：
    - `Image Path: /a/b/c.jpg`
    - `[VLM_IMAGE_n]`

### 3) 装箱与超长处理

- 改造前：逻辑与目标不一致（曾出现按标记位置插图）。
- 改造后：
  - 装箱策略固定为：图片在前、文本在后。
  - 图片去重后按 marker 顺序加入。
  - 超长仅通过降图重试：`10 -> 8 -> 6 -> 4 -> 2`。
  - 不做文本裁剪。

### 4) 模型端点与多模态 JSON

- 改造前：text/vision 配置边界不清晰，调用参数存在重复过滤代码。
- 改造后：
  - 拆分并解析 text/vision 端点：
    - `vllm_api_base/key/llm_model_name`
    - `vision_vllm_api_base/key/vision_model_name`
  - `internvl2` 模型名自动触发 `prompt_internvl2` 覆盖。
  - 入库多模态分析支持 `json_schema`（可开关），失败自动降级重试。

## 二、文件级变更

### A. `raganything/services/local_rag.py`

- 新增与整理：
  - context 结构化解析与清洗函数（entity/relation/chunk）
  - 图像路径映射与 basename 匹配
  - 图片前置装箱函数
  - 上下文超长重试函数
  - text/vision 端点解析与 internvl2 提示词开关
- 精简与去重：
  - 删除未使用参数 `vlm_max_text_chars`
  - 删除未使用导入 `List`
  - 删除未使用中间变量 `fence_open_start`
  - 统一内部 kwargs 过滤函数，去掉重复代码块
  - raw messages 回退调用收敛为单一路径

### B. `evaluate_local/DocBench/evaluate.py`

- 对齐 `LocalRagSettings` 新字段（text/vision 同步赋值）。
- `--dump_final_messages` 调试钩子使用 `service.vision_client`。
- 删除重复的 `PROMPTS.update(PROMPTS_INTERNVL2)`（已由 local_rag 自动处理）。
- 删除对已移除参数 `vlm_max_text_chars` 的设置。
- 删除未使用变量 `rag_cleaned`。
- 删除未使用导入 `Optional`。

### C. `examples/raganything_local_v2.py`

- 删除冗余的 vision 回写赋值（`from_env` 已含回退逻辑）。
- 注释更新为“示例参数，不影响 local_rag 通用能力”。

## 三、当前确认点

1. `[VLM_IMAGE_n]` 是 rag-anything 既有约定（非 LightRAG 原生），用于图文对位装箱。
2. 当前保留策略是：chunk 路径不替换，仅追加 `[VLM_IMAGE_n]`。
3. 关系不可映射路径端点占位符已统一为 `[IMAGE_ENTITY]`。
4. 本轮未修改 `query.py`。

## 四、验证记录

- 语法校验通过：
  - `raganything/services/local_rag.py`
  - `evaluate_local/DocBench/evaluate.py`
  - `examples/raganything_local_v2.py`
- 无未使用导入（针对本轮 3 个改动文件逐一检查）。

- 行数（本轮结束）：
  - `local_rag.py`: 1054
  - `evaluate.py`: 894
  - `raganything_local_v2.py`: 85

## 五、最终复核（2026-02-20）

1. `local_rag.py`：
   - 核心链路闭环：配置 -> 端点解析 -> text/vision client -> query/ingest 分支。
   - 路径清洗与图像映射逻辑一致：entity 删除、relation 替换、chunk 追加标记。
   - 装箱策略固定：图片前置、文本后置；超长仅降图重试。
2. `evaluate.py`：
   - 与 `LocalRagSettings` 字段对齐，`dump_final_messages` 使用 `vision_client`。
   - 无重复 prompt 覆盖代码，无冗余参数设置。
3. `raganything_local_v2.py`：
   - 保持示例脚本定位，删除冗余回写配置，仅保留必要参数与调用流程。


## 增量更新（2026-02-20 14:46:51）

本次增量主要覆盖 `raganything/services/local_rag.py`，并同步检查 `evaluate.py` 与 `examples/raganything_local_v2.py` 的调用兼容性。

### 1) Query 侧可观测性增强
- 增加 ingest 阶段 JSON Schema 相关日志：
  - `Vision ingest schema check: enabled=..., prompt_has_json=..., using_schema=...`
  - `Vision ingest calling with response_format=json_schema (strict=...)`
  - `Vision ingest json_schema call succeeded.`

### 2) System/User 组装清理
- 清理空的 `Additional Instructions:` 行，避免空模板污染 system。
- 处理 `User Question` 重复问题：
  - 预清理旧问句块；
  - 拼装前再次清理残留 `User Question` 行。

### 3) Context 解析降级策略
- 不再把 `Reference Document List` 作为清洗前置硬依赖。
- 当 `entity/relation` JSON-lines 解析失败时，进入 degraded 模式：
  - 打 warning；
  - 保留原区块；
  - 不回退 raw messages。

### 4) JSON-lines 调试日志
- 新增 `entity/relation` 行级解析日志：
  - section 名称；
  - 行号；
  - 失败行片段（snippet）；
  - 异常信息。

### 5) 待继续验证项
- `context parsing failed` 的触发链路是否已完全可定位。
- 超长重试是否需要扩展到 `0` 图兜底策略。

## 增量更新（2026-02-23，Token Budget 解释）

### 预算公式
available_chunk_tokens = max_total_tokens - (sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens)

### 各项含义
- max_total_tokens：总预算（LightRAG QueryParam.max_total_tokens）。
- sys_prompt_tokens：系统提示模板本身的 token 消耗（上下文留空，仅计算模板和响应类型等固定开销）。
- kg_context_tokens：KG 区块 token 消耗（由 entity + relation 组成，按模板拼接后再编码统计；不含 chunks，不含 reference list）。
- query_tokens：用户问题文本 token 消耗。
- buffer_tokens：固定安全预留（当前实现为 200），用于 reference list 和波动余量。

### 与实体/关系裁剪的关系
- 先按 max_entity_tokens、max_relation_tokens 对实体和关系做裁剪。
- 再基于裁剪后的 entity/relation 拼接 pre_kg_context 并计算 kg_context_tokens。
- 所以 kg_context_tokens 不是“配置上限本身”，而是“裁剪后真实格式化文本的 token 占用”。

### 代码位置
- 实体/关系裁剪：lightrag/lightrag/operate.py（约 3685-3718 行）。
- 预算计算与 available_chunk_tokens：lightrag/lightrag/operate.py（约 3923-3945 行）。

## 增量计划（2026-02-23，Ingest 非 image 也启用 JSON Schema）

> 说明：本节仅记录计划，暂未执行代码改动。

### 目标
- 仅修改 `raganything/services/local_rag.py`。
- 不修改官方核心文件（`modalprocessors.py`、`query.py`、`prompt.py`）。
- 在入库阶段（ingest）让 table/equation/generic 的结构化 JSON 任务也可使用 `response_format=json_schema`。

### 计划改动点（仅 local_rag）
1. 抽取“入库结构化任务识别”函数（显式判定 ingest + JSON 结构任务）。
2. 抽取统一 schema 构造函数，复用当前 `detailed_description + entity_info` 结构。
3. 将“无 image_data 的 ingest 调用”从直接回退文本调用改为：
   - 先尝试带 schema 调用；
   - 若失败，自动移除 `response_format` 再重试一次。
4. 保持 query 分支逻辑不变（不影响 VLM enhanced query 的现有行为）。

### 参数与兼容策略
- 初始保持 `strict=False`、`additionalProperties=True`，优先保证兼容与成功率。
- 若后续稳定，再评估 `strict=True` 与 `additionalProperties=False` 的收紧策略。

### 可观测性补充
- 增加 ingest 侧日志：
  - 任务类型（image / table / equation / generic）
  - 是否使用 schema
  - schema 首次成功/失败
  - fallback 重试结果
- 增加阶段性统计：
  - schema_success_rate
  - schema_fallback_rate

### 风险与预期
- 预期不影响 query 速度与行为。
- ingest 速度可能小幅下降；若 schema 失败率高，单条请求可能出现一次重试开销。
- 改动范围小，回滚简单（仅 local_rag 单文件）。

## 增量计划补充（2026-02-23，统一 Query 输出上限）

> 说明：本节仅记录计划，暂未执行代码改动。

### 目标
- 将 enhanced 与 non-enhanced 两条 query 回答路径的 `max_tokens` 输出上限统一。
- 优先方案：统一为 `2048`，降低 400 风险并保持可读回答长度。

### 现状
- non-enhanced（文本 query）使用 `settings.max_tokens`（当前默认 8192）。
- enhanced（VLM query）使用 `settings.vision_max_tokens`（当前默认 2048）。
- 两条路径上限不一致，会导致行为差异和排障复杂度增加。

### 计划改动（仅 local_rag）
1. 为 query 回答统一单一输出上限配置（建议统一到 2048）。
2. non-enhanced 与 enhanced 均读取同一上限值。
3. ingest 路径单独统一输出上限为 8192（image / 非 image 保持一致，不与 query 共用）。
4. 增加启动日志，明确打印两条 query 路径使用的最终 `max_tokens`。

### 取值建议
- 先使用 `2048`：在大多数问答场景下够用，且可明显降低超长风险。
- 如后续需要长文回答，可按场景放宽到 `3072`，不建议默认回到 `8192`。

### 当前决议（2026-02-23）
- query：enhanced 与 non-enhanced 统一为 2048（降低回答阶段超长风险）。
- ingest：image 与非 image 统一为 8192（保证多模态入库描述信息充分）。

## 增量计划补充（2026-02-23，local_rag 侧兼容 `\n[VLM_IMAGE_n]`，不改 query.py）

> 说明：本节仅记录方案，暂未执行代码改动。

### 背景判断（复核结论）
- 当前链路是：`query.py` 先把 `Image Path: ...` 替换成 `Image Path: ...\n[VLM_IMAGE_n]`，再交给 `local_rag.py` 二次解析。
- `local_rag.py` 的 entity/relation 解析是 JSON-lines（每行一个 JSON），若标记换行进入 JSON 行，`json.loads(line)` 会失败。
- 因此，路径 entity 删除/relation 端点替换目前是“有条件生效”（解析成功时才生效），并非稳定生效。

### 目标
- 不修改官方 `query.py`。
- 仅修改 `local_rag.py`，让现有 `\n[VLM_IMAGE_n]` 也能稳定进入清洗流程。
- 保持“chunk 作为图例真源”，避免重复打标和重复改写。

### 计划改动（仅 local_rag）
1. **预归一化（解析前）**
   - 在 context 解析前，将 `\n[VLM_IMAGE_n]` 规范为同一行标记（例如空格连接），避免打断 JSON-lines。
2. **先 chunk 后 entity/relation**
   - 先解析 chunk，建立 `path -> marker` 映射（图例真源）。
   - 再解析 entity/relation：
     - entity：路径型实体删除；
     - relation：路径端点优先映射到 `[VLM_IMAGE_n]`，失败回退 `[IMAGE_ENTITY]`。
3. **去掉重复打标**
   - 不再对 chunk 二次重打标；优先复用上游已有 marker，减少二次改写副作用。
4. **降级策略**
   - entity/relation 任一解析失败时，不 raw 回退；
   - 保留可解析区块的处理结果，未解析区块保留原文，保证主流程可用。

### 可观测性
- 增加日志字段：
  - `markers_normalized_count`
  - `chunk_parse_ok`
  - `entity_parse_ok`
  - `relation_parse_ok`
  - `path_marker_map_size`

### 预期收益
- 降低 `entity JSON-lines invalid / relation JSON-lines invalid` 触发率。
- 在不改 `query.py` 的前提下，稳定恢复“路径 entity 删除 + relation 端点替换”能力。
- 降低链路复杂度（不再先去标记再重打标）。

## 执行记录（2026-02-23，已落地到 local_rag.py）

### 1) Ingest 非 image 启用 JSON Schema（已执行）
- `vision_model_func` 中新增 ingest 任务识别（`image/table/equation/generic/query`）。
- 非 image ingest（table/equation/generic）也会按条件启用 `response_format=json_schema`。
- schema 失败会自动移除 `response_format` 并重试一次。
- 保持 query 分支行为不变（messages 分支仍为 enhanced query 路径）。

### 2) Query / Ingest 输出上限统一（已执行）
- 新增：
  - `query_max_tokens`（默认 2048）
  - `ingest_max_tokens`（默认 8192）
- query（enhanced + non-enhanced）统一使用 `query_max_tokens`。
- ingest（image + 非 image）统一使用 `ingest_max_tokens`。
- 启动日志新增：`Token caps configured: query_max_tokens=..., ingest_max_tokens=..., vlm_enable_json_schema=...`。

### 3) 兼容 `\\n[VLM_IMAGE_n]`（仅改 local_rag，已执行）
- 解析前预归一化：将 `\\n[VLM_IMAGE_n]` 规范为同一行 marker，避免 JSON-lines 断行。
- 先解析 chunk，建立 `path -> marker` 图例映射，再处理 entity/relation。
- 去掉 chunk 二次重打标，优先复用上游 marker。
- entity/relation 任一解析失败时，不 raw 回退，保留原区块继续流程。

### 4) 新增可观测日志字段（已执行）
- context 清洗日志包含：
  - `markers_normalized_count`
  - `chunk_parse_ok`
  - `entity_parse_ok`
  - `relation_parse_ok`
  - `path_marker_map_size`
- ingest schema 侧日志包含：
  - `task_type`
  - `using_schema`
  - `schema_success_rate`
  - `schema_fallback_rate`

## 最终复核（2026-02-23）

### 覆盖文件
- `raganything/services/local_rag.py`（已改）
- `evaluate_local/DocBench/evaluate.py`（已改）
- `examples/raganything_local_v2.py`（本轮无需改）

### 结果
- 语法检查通过：
  - `python -m py_compile raganything/services/local_rag.py`
  - `python -m py_compile evaluate_local/DocBench/evaluate.py`
  - `python -m py_compile examples/raganything_local_v2.py`
- token 配置已统一到新字段：
  - query：`query_max_tokens=2048`
  - ingest：`ingest_max_tokens=8192`
- 兼容 `\n[VLM_IMAGE_n]` 的 local_rag 清洗链路与可观测日志已落地。
- evaluate 已切换到新字段（不再写入旧字段 `max_tokens/vision_max_tokens`）。

### 当前代码行数
- `raganything/services/local_rag.py`: 1322
- `evaluate_local/DocBench/evaluate.py`: 902
- `examples/raganything_local_v2.py`: 85

## 执行记录补充（2026-02-23，non-enhanced 与 enhanced 消息组织统一）

### 变更目标
- 统一 non-enhanced 文本 query 与 enhanced query 的消息组织方式：
  - `system`：仅放 Role/Goal/Instructions 等系统指令
  - `user`：放 Context + User Question
- 不新增 `text_query_repacked=true/false` 日志（按要求关闭）。

### 已执行修改（仅 `local_rag.py`）
- 新增 `_clean_context_for_user_text(...)`：统一清理 `User Question` 和末尾提示句。
- 新增 `_try_repack_text_query(system_prompt, prompt)`：
  - 从 LightRAG 文本 query 的 `system_prompt` 中拆分 `---Context---`
  - 组装为新的 `final_system` + `final_user`
- 在 `build_llm_model_func(...)` 中接入重组逻辑：
  - 可重组时按“system 指令 + user 上下文/问题”发送
  - 不可重组时保持原发送路径（兼容回退）
- enhanced 分支复用 `_clean_context_for_user_text(...)`，去掉重复清理代码。
- 修正配置注释语义：
  - `vlm_max_images` 标注为 enhanced query 图片上限
  - `vlm_enable_json_schema` 标注为 ingest 结构化输出开关

### 相较未改动前的功能变化
- non-enhanced 不再是“几乎全部内容放 system、query 放 user”。
- non-enhanced 与 enhanced 现在在消息结构上对齐，行为更一致，便于排障和对比。
- 未引入额外冗余开关与日志字段，保留最小必要改动。

### 本轮检查结论
- 三个目标文件功能正常、无语法错误：
  - `raganything/services/local_rag.py`
  - `evaluate_local/DocBench/evaluate.py`
  - `examples/raganything_local_v2.py`
- 检查命令：
  - `python -m py_compile raganything/services/local_rag.py evaluate_local/DocBench/evaluate.py examples/raganything_local_v2.py`
- 文本编码清理：已去除 `local_rag.py` 与本记录文件的 UTF-8 BOM 头。
- 冗余性复查：
  - `evaluate.py` 已使用新 token 字段（`query_max_tokens/ingest_max_tokens`），无旧字段残留调用。
  - `raganything_local_v2.py` 保持示例最小参数集，无需额外改动。

## 执行记录补充（2026-02-23，keyword_extraction 官方对齐）

### 本次已执行改动（仅 `raganything/services/local_rag.py`）
- 在 `build_llm_model_func` 中按官方语义处理 `keyword_extraction`：先从 kwargs 中读取并移除该内部参数。
- 当 `keyword_extraction=True` 时，设置 `response_format=GPTKeywordExtractionFormat`（与 LightRAG 官方关键词抽取契约一致）。
- 关键词抽取请求改为在 `response_format` 存在时走 `client.chat.completions.parse(...)`，其余请求继续走 `create(...)`。
- 对 parse 结果统一序列化为 JSON 字符串返回（优先 `model_dump_json()`），保持与上游 `json_repair.loads(...)` 兼容。
- 保留 `_strip_internal_openai_kwargs`，不删除该函数；仅修正其在关键词抽取路径上的使用方式。

### 相较改动前的行为变化
- 改动前：`keyword_extraction` 被过滤后无任何专门处理，关键词抽取无结构化约束。
- 改动后：关键词抽取具备官方一致的结构化输出约束，降低 `low_level_keywords is empty` 的非预期触发概率。

### 本次校验
- 语法校验通过：
  - `python -m py_compile raganything/services/local_rag.py`

## 官方对齐审计（基于当前代码）

### 仍为自定义实现（不属于 LightRAG 默认 wrapper）
- 文本 query 的 system/user 重组。
- VLM query 前 context 二次清洗与重排。
- ingest schema 判定、注入、失败回退。
- InternVL2 prompt override。
- 自定义 `llm_model_func/vision_model_func` 注入。
- text/vision 双端点解析与双客户端初始化（`_resolve_text_endpoint/_resolve_vision_endpoint`）。
- 本地 embedding/rerank 模型加载与 `rerank_model_func` 注入。
- query 阶段 VLM 上下文过长时的降图重试策略。

### 其余主要链路仍使用官方
- 关键词提取、KG 检索、实体/关系/chunk 构建、query 主流程由 LightRAG 官方 `operate.py` 执行。
- `raganything/query.py`、`modalprocessors.py`、`prompt.py` 本次未改，继续沿用官方逻辑。

## 最终复检（2026-02-23，keyword_extraction 对齐后）

### 本次相较未改动前的功能变更
- `raganything/services/local_rag.py`：关键词抽取路径改为官方语义对齐。
  - 读取并消费内部参数 `keyword_extraction`。
  - 关键词抽取时启用 `response_format=GPTKeywordExtractionFormat`。
  - 结构化输出时使用 `chat.completions.parse(...)` 并返回可被上游 JSON 解析的字符串。
- `evaluate_local/DocBench/evaluate.py`：本轮未新增功能改动，仅与现有参数路径复核兼容。
- `examples/raganything_local_v2.py`：本轮未新增功能改动，仅复核调用兼容。

### 三文件最终检查结果
- 语法检查通过：
  - `python -m py_compile raganything/services/local_rag.py`
  - `python -m py_compile evaluate_local/DocBench/evaluate.py`
  - `python -m py_compile examples/raganything_local_v2.py`
- 关键路径检查通过：
  - 关键词抽取结构化开关、parse 分支与 ingest schema 分支互不冲突。
  - query 重组、VLM 清洗、ingest schema 回退、InternVL2 prompt override、自定义注入保持原有行为。

### 说明
- 本轮保持”最小改动”原则：仅调整关键词抽取链路，不扩展新配置项，不引入额外分支复杂度。

## 待行动补充（2026-02-25，统一尾句与 system/user 组装）

> 说明：本节是待执行计划，当前尚未提交代码改动。

### 目标
- enhanced（有图）尾句使用官方文案：`Please answer based on the context and images provided.`
- non-enhanced 与 enhanced（无图）尾句统一为：`Please answer based on the context provided.`
- 两条链路保持同一组装骨架，仅保留“有图/无图”必要差异。
- 禁止尾句重复（同一请求中不能同时出现两种尾句）。

### 计划改动点（仅 `raganything/services/local_rag.py`）
1. 增加统一尾句选择逻辑（按是否携带图片二选一）。
2. non-enhanced 路径固定使用 `context provided` 尾句。
3. enhanced 路径：
   - 有图：使用 `context and images provided` 尾句；
   - 无图：与 non-enhanced 对齐，使用 `context provided` 尾句。
4. 在拼接最终 user 文本前，先清理旧尾句模板，避免重复追加。
5. system/user 重组规则保持一致：仅取最后一个 `---Context---`，并清理嵌入的 `---User Query---`、`User Question:`、历史尾句模板。

### 我们自己追加的 prompt（计划保留）
- `User Question: {query}`（统一追加在 user 文本末尾的问题位）。
- 单一尾句（按模式二选一）：
  - enhanced（有图）：`Please answer based on the context and images provided.`
  - non-enhanced / enhanced（无图）：`Please answer based on the context provided.`

### 计划改动后组装示例（详细）

#### A) non-enhanced（文本问答）

- `system`（示例结构）
```text
---Role---
...（来自 LightRAG 的角色说明）

---Goal---
...（来自 LightRAG 的任务目标）

---Instructions---
...（来自 LightRAG 的指令约束；空 Additional Instructions 会被清理）
```

- `user`（示例结构）
```text
---Context---
...（Knowledge Graph Data / Document Chunks / Reference Document List）

User Question: {query}

Please answer based on the context provided.
```

#### B) enhanced（有图）

- `system`（示例结构）
```text
You are a helpful assistant that can analyze both text and image content to provide comprehensive answers.

---Role---
...（来自 LightRAG 的角色说明）

---Goal---
...（来自 LightRAG 的任务目标）

---Instructions---
...（来自 LightRAG 的指令约束）
```

- `user`（示例结构，multimodal content）
```text
[
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
  ...,
  {
    "type": "text",
    "text": "---Context---\n...\n[VLM_IMAGE_1]\n...\n\nUser Question: {query}\n\nPlease answer based on the context and images provided."
  }
]
```

#### C) enhanced（无图，回退文本）

- 对齐要求：与 non-enhanced 完全一致（同一套 system/user 重组规则与同一尾句）。

- `system`（示例结构）
```text
---Role---
...（来自 LightRAG 的角色说明）

---Goal---
...（来自 LightRAG 的任务目标）

---Instructions---
...（来自 LightRAG 的指令约束）
```

- `user`（示例结构）
```text
---Context---
...（Knowledge Graph Data / Document Chunks / Reference Document List）

User Question: {query}

Please answer based on the context provided.
```

### 一致性判定标准（验收）
- non-enhanced 与 enhanced（无图）的 system/user 文本应一致。
- enhanced（有图）与 non-enhanced 的唯一必要差异：
  1) user 前部包含 `image_url` 列表；
  2) system 前置一条多模态能力提示。
- 不应再出现同一请求中两种尾句并存。

## 执行记录（2026-02-25，多模态检索增强 + 配置统一 + WebUI 重设计）

### 本次改动范围（5 个子任务）

| 编号 | 内容 | 涉及文件 |
|------|------|---------|
| T1 | 多模态 chunk 名额上限（`multimodal_top_k`） | `lightrag/base.py`、`lightrag/utils.py`、`query.py` |
| T2 | 索引阶段路径实体过滤 | `lightrag/operate.py` |
| T3 | VLM query 分支直通简化 | `local_rag.py` |
| T4 | rag-anything 配置统一（`constants.py`） | `constants.py`（新建）+ 5 个文件同步 |
| T5 | WebUI 全新设计 | `server/templates/index.html` |

---

### T1：多模态图片上限（`multimodal_top_k`）

#### 背景
多模态 chunk 的内容字段含 `Image Path: ...` 行，VLM 增强查询时会把这些路径替换为 base64 图片。无上限时大量图片会被发给 VLM，占用 context window 并增加延迟。此外，多模态 chunk 与文本 chunk 共享 `chunk_top_k` 名额，若多模态 chunk 排名靠前则会挤出高相关性纯文本原文。

#### 最终语义（v2，初始实现已更新见下方增量记录）

- `multimodal_top_k`：**仅控制发给 VLM 的图片数量上限**，不限制 context 中的 chunk 数量
- 所有多模态 chunk 均进入 context（排名前 `multimodal_top_k` 个附图，其余降级为纯文本 VLM 描述）
- 纯文本 chunk 获得独立配额：`chunk_top_k - multimodal_top_k` 个，从全量 rerank 结果中取最相关的
- 最终 context 总量 = 全部 mm_chunks + min(text_pool, chunk_top_k - multimodal_top_k) 个文本 chunk

#### 初始改动（`base.py`、`utils.py`、`query.py`）

**`lightrag/lightrag/base.py`（L171）**
- 在 `QueryParam` 中新增字段：
  ```python
  multimodal_top_k: int | None = None
  ```
  默认 `None`（向后兼容）。

**`rag-anything/raganything/query.py`（`aquery_vlm_enhanced()`）**
- 在 `QueryParam` 构造前注入默认值，并传递 `max_images` 给图片扫描函数：
  ```python
  kwargs.setdefault(“multimodal_top_k”, DEFAULT_MULTIMODAL_TOP_K)  # = 3
  image_cap = kwargs[“multimodal_top_k”]
  enhanced_prompt, images_found = await self._process_image_paths_for_vlm(
      raw_prompt, max_images=image_cap
  )
  ```
- `_process_image_paths_for_vlm(prompt, max_images=None)`：新增 `max_images` 参数，在 `replace_image_path` 回调中按出现顺序计数，达到上限后剩余路径保留为纯文本，不附图。
- `aquery()` 中补充默认值注入（确保直接调用也遵循 constants 配置）：
  ```python
  kwargs.setdefault(“top_k”, DEFAULT_TOP_K)
  kwargs.setdefault(“chunk_top_k”, DEFAULT_CHUNK_TOP_K)
  ```

> 注：`utils.py` 的 Step 2.5 逻辑自初始提交后已完全重写，见下方增量记录。

---

### T2：索引阶段路径实体过滤（v2，已修订）

#### 背景
多模态 chunk 模板含 `Image Path: /path/to/file.jpg`，LLM 在抽取实体/关系时会把文件路径、工作空间目录当作普通实体存入知识图谱，产生大量噪声节点。对 Ubuntu 系统尤为明显（`/home/user/rag_workspace/...` 等绝对路径）。

#### 方案选择

采用 **post-extraction 过滤**（LLM 提取完成后、写入 KG 前过滤）：
- 对 LightRAG 管道**零侵入**，不修改 prompt 或 chunk 分发逻辑
- 过滤时机正确：`sanitize_and_normalize` 之后、KG 写入之前
- 关系过滤位置正确：在 `source == target` 检查之前，避免无效的 description 处理
- 与 `_handle_single_entity_extraction` 现有空值检测模式一致（均使用 `logger.info`）

唯一更优的方案（在 chunk 进入 entity extraction LLM 前 mask `Image Path:` 行）需 hook 进 LightRAG 内部 chunk 分发，侵入性高，暂不采用。

**固有限制**：裸文件夹名（如 `hybrid_auto`、`rag_workspace`，无路径分隔符、无扩展名）在 regex 层面无法与正常实体名区分，属方案边界，需通过 prompt 层指令解决（不在本轮范围内）。

#### 改动

**`lightrag/lightrag/operate.py`（L77–L146）**

初版（T2 首次提交）存在两个问题后经复查修复：
1. `_FILE_PATH_RE` 定义后从未被 `_is_file_or_folder_path()` 使用（死代码，已删除）
2. URL 保留逻辑 `not cleaned.startswith(“http”)` 过于窄，`ftp://`、`s3://`、`doi://` 等会被误过滤（已修复）

最终实现：

```python
# Matches any valid URL scheme (http, https, ftp, s3, git, doi, …)
_URL_SCHEME_RE = re.compile(r”^[a-zA-Z][a-zA-Z0-9+\-.]*://”, re.IGNORECASE)

# Bare filename with a known extension but no directory separators
_BARE_FILENAME_RE = re.compile(
    r”^[^\s\\\/]*\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif|svg|ico”
    r”|pdf|doc|docx|txt|xlsx|xls|pptx|ppt|html|htm|md|csv”
    r”|py|js|ts|java|cpp|c|h|go|rs|rb|sh|bat”
    r”|yaml|yml|json|xml|toml|cfg|ini|conf|log|env|lock)$”,
    re.IGNORECASE,
)

def _is_file_or_folder_path(name: str) -> bool:
    if not name: return False
    cleaned = name.strip().strip(“\”'`”)
    if not cleaned: return False
    if _URL_SCHEME_RE.match(cleaned): return False        # 保留有效 URL
    if re.match(r”^[A-Za-z]:[\\\/]”, cleaned): return True  # Windows 绝对路径
    if cleaned.startswith(“/”): return True              # Unix/Linux 绝对路径
    if len(cleaned) > 1 and cleaned[0] == “~” and cleaned[1] in (“/”, “\\”): return True  # ~/...
    if re.search(r”[\\\/]”, cleaned): return True        # 含目录分隔符的相对路径
    if _BARE_FILENAME_RE.match(cleaned): return True     # 裸文件名含扩展名
    return False
```

**`_handle_single_entity_extraction()`（L476–L481）**
```python
if _is_file_or_folder_path(entity_name):
    logger.info(f”Filtered file path entity: '{entity_name}'”)
    return None
```

**`_handle_single_relationship_extraction()`（L566–L571）**
```python
if _is_file_or_folder_path(source) or _is_file_or_folder_path(target):
    logger.info(f”Filtered relationship with path entity: '{source}' -> '{target}'”)
    return None
```

#### 测试覆盖（31 个用例，全部通过）

| 类别 | 示例 | 结果 |
|------|------|------|
| Unix 绝对路径 | `/home/user/docs/paper.pdf`、`/workspace/proj` | 过滤 ✓ |
| Ubuntu 模型路径 | `/data/h50056787/models/bge-m3` | 过滤 ✓ |
| Windows 绝对路径 | `C:\Users\file.png`、`C:/foo/bar.jpg` | 过滤 ✓ |
| 相对路径 | `./output/img.jpg`、`../data/train.csv`、`foo/bar/baz` | 过滤 ✓ |
| `~` 路径 | `~/documents/thesis.pdf`、`~/projects/rag` | 过滤 ✓ |
| 裸文件名（含扩展） | `figure1.png`、`config.yaml`、`run.sh` | 过滤 ✓ |
| 相对工作区路径 | `output/paper_id/hybrid_auto/page.png`、`rag_workspace/output` | 过滤 ✓ |
| `http/https` URL | `https://github.com/HKUDS/LightRAG` | 保留 ✓ |
| **`ftp` URL（原来漏掉）** | `ftp://server.com/data/file.csv` | 保留 ✓ |
| **`s3` URL（原来漏掉）** | `s3://my-bucket/prefix/key` | 保留 ✓ |
| **`doi` URL（原来漏掉）** | `doi://10.1145/1234` | 保留 ✓ |
| 正常实体名 | `Knowledge Graph`、`BERT`、`bge-m3`、`RAGAnything` | 保留 ✓ |
| 裸文件夹名（固有限制） | `hybrid_auto`、`rag_workspace` | 保留（无法区分）|

---

### T3：VLM query 分支直通简化

#### 背景
`local_rag.py` 的 `vision_model_func` query 分支（`if messages:` 段落）包含约 70 行二次处理逻辑（`_sanitize_context_for_vlm()` 清洗知识图谱路径实体、`_build_content_parts_from_markers()` 重排图文顺序），这些是对索引阶段遗留问题的 patch。T2 在索引阶段根治问题后，这些 patch 不再必要。

`query.py` 的 `_build_vlm_messages_with_images()` 已经构建了格式正确的多模态消息（交错文本 + base64 图像），无需二次处理。

#### 改动

**`rag-anything/raganything/services/local_rag.py`（`vision_model_func` → `if messages:` 分支）**
- 删除约 70 行二次处理，替换为直通调用：
  ```python
  if messages:
      response = await client.chat.completions.create(
          model=model_name,
          messages=messages,
          temperature=settings.temperature,
          max_tokens=settings.query_max_tokens,
          **cleaned_kwargs,
      )
      return response.choices[0].message.content
  ```
- 函数定义（`_sanitize_context_for_vlm`、`_build_content_parts_from_markers` 等）保留，不删除（避免影响其他潜在调用）。

---

### T4：rag-anything 配置统一（`constants.py`）

#### 背景
配置默认值分散在 `config.py`、`local_rag.py`、`server/app.py` 三处，存在 9 个重复/冲突项（如 `DEFAULT_TOP_K=15`、`DEFAULT_CHUNK_TOP_K=30`），且 `local_rag.py` 中含有硬编码服务器路径（`/data/h50056787/models/bge-m3`）。

#### 改动

**`rag-anything/raganything/constants.py`（新建，~121 行）**
- 集中定义所有默认值，分为 8 个区段：
  - 目录：`DEFAULT_OUTPUT_DIR`、`DEFAULT_WORKING_DIR_ROOT`、`DEFAULT_LOG_DIR`
  - 解析器：`DEFAULT_PARSER`、`DEFAULT_PARSE_METHOD`、`DEFAULT_CONTENT_FORMAT` 等
  - 多模态：`DEFAULT_ENABLE_*_PROCESSING`、`DEFAULT_MULTIMODAL_TOP_K=3`
  - 批处理：`DEFAULT_MAX_CONCURRENT_FILES`、`DEFAULT_SUPPORTED_FILE_EXTENSIONS`
  - 上下文提取：`DEFAULT_CONTEXT_WINDOW`、`DEFAULT_CONTEXT_MODE` 等
  - 图像验证：`DEFAULT_MAX_IMAGE_SIZE_MB=50`、`SUPPORTED_IMAGE_EXTENSIONS`
  - 查询：`DEFAULT_TOP_K=20`、`DEFAULT_CHUNK_TOP_K=10`
  - 本地部署：模型路径（通用名替换硬编码服务器路径）、端点、token 上限、VLM 参数、日志

**同步更新的文件**

| 文件 | 变更内容 |
|------|---------|
| `raganything/config.py` | 所有 `default=...` 字面量替换为 `constants` 导入 |
| `raganything/services/local_rag.py` | `LocalRagSettings` 所有 `field(default=...)` 替换；硬编码模型路径替换为 `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3` |
| `server/app.py` | `DEFAULT_TOP_K`、`DEFAULT_CHUNK_TOP_K`、`DEFAULT_SUPPORTED_FILE_EXTENSIONS` 来自 constants |
| `raganything/utils.py` | `validate_image_file()` 默认参数和 `image_extensions` 列表来自 constants |
| `raganything/query.py` | `DEFAULT_MULTIMODAL_TOP_K` 来自 constants |

---

### T5：WebUI 全新设计

**`server/templates/index.html`（完全重写，约 470 行）**

#### 设计方向
编辑室/研究期刊风格：左侧暖色调文档阅读器（类 Notion/Google Docs 质感）+ 右侧深色精致聊天区，琥珀色点缀系统。

#### 视觉特征
- **字体**：DM Serif Display（标题）+ IBM Plex Sans（正文）+ IBM Plex Mono（代码/API Key）
- **色彩**：琥珀色（`#d4940a`）点缀，深色/浅色两套完整主题，CSS 变量全局切换，持久化到 `localStorage`
- **背景**：深色主题用多层暖棕黑底色区分面板（`#111113` app / `#22211d` 阅读区 / `#141416` 聊天区）

#### 功能增强
- **可拖拽分割线**：鼠标拖拽调整左右面板比例（20%–75%），悬停有视觉反馈
- **消息动画**：每条消息 `fadeInUp` 入场；等待回答时显示三点脉冲动画替代静态文字
- **页面加载动画**：header → 左面板 → 右面板依次 `fadeInUp`
- **Textarea 自动扩展**：随输入内容自动增高（最大 120px）
- **响应式**：768px 以下切换为上下堆叠布局

#### 功能保留（无变化）
- API 端点：`GET /workspaces`、`GET /files/{doc_id}`、`GET /content/{doc_id}`、`POST /query`
- 引用链接点击 → 高亮并切换源文件
- marked.js CDN Markdown 渲染（含 fallback）
- Enter 发送 / Shift+Enter 换行

---

### 本次验证

- `lightrag/lightrag/base.py`：`multimodal_top_k` 字段添加位置正确，类型注解完整。
- `lightrag/lightrag/utils.py`：步骤 2.5 插入位置在 rerank 过滤之后、`chunk_top_k` 截断之前，逻辑正确。
- `lightrag/lightrag/operate.py`：`_is_file_or_folder_path()` 独立单元测试 23 个用例全部通过（包含 Windows/Unix/相对路径/裸文件名/模型名/空字符串）。
- `raganything/constants.py`：独立导入测试通过，无循环依赖。
- `raganything/query.py`、`config.py`、`utils.py`、`server/app.py`、`local_rag.py`：`py_compile` 语法检查通过。
- WebUI：所有原有 API 调用路径保留，功能完整。

### 本次结束后各文件行数

| 文件 | 行数 |
|------|------|
| `raganything/constants.py` | ~117（新建；注：`DEFAULT_UPLOAD_DIR` 在 7b483f3 commit message 中声称删除但实际漏删，后续增量记录中已修正）|
| `lightrag/lightrag/base.py` | +6 行（新增字段） |
| `lightrag/lightrag/utils.py` | +12 行（步骤 2.5 初始版本，后续已重写） |
| `lightrag/lightrag/operate.py` | +32 行（函数 + 两处过滤调用） |
| `raganything/query.py` | +2 行（导入 + setdefault） |
| `server/templates/index.html` | ~470（完全重写）|

---

## 增量更新（2026-02-25，server/app.py 后端优化）

### 改动范围

唯一修改文件：`rag-anything/server/app.py`

### 问题根因

parser 写出路径 ≠ server 查找路径：server 假设文件在 `{output_dir}/{doc_id}/hybrid_auto/`，但 parser 实际写入 `{output_dir}/{file_stem}/hybrid_auto/`。当 `doc_id != file_stem` 时文件找不到（用户传入自定义 `doc_id`、文件名含特殊字符等场景）。

### 核心修复

- `/ingest` 传入 `workspace_output = output_dir/{final_doc_id}` 作为 parser 的 `output_dir`，parser 写入 `output/{doc_id}/{stem}/hybrid_auto/`
- `_find_md_in_hybrid_auto()` 和 `/files/{doc_id}` 改用 `rglob("hybrid_auto/*.md")` 递归查找，同时兼容旧格式数据（`{doc_id}/hybrid_auto/` 也能被 rglob 命中）

### 其他修复

| 问题 | 修复方式 |
|------|---------|
| `_find_md_in_hybrid_auto()`、`list_workspace_files()`、`list_workspaces()` 每次请求重复 `LocalRagSettings.from_env()` | 改用 `service.settings`（通过依赖注入传入） |
| 模糊匹配返回 `candidates[0]` 前未排序，结果不稳定 | 改为 `sorted(rglob(...))` 后再过滤取第一个 |
| `doc_id` 直接拼入路径，存在路径穿越风险 | 所有接受 `doc_id` 的函数均加 `".."/"/"\\` 校验，返回 400 |
| 上传文件未做扩展名校验 | 对照 `DEFAULT_SUPPORTED_FILE_EXTENSIONS` 构建 `SUPPORTED_EXTENSIONS` 集合，不合法返回 400 |
| ingest 成功/失败后上传文件未清理，磁盘持续增长 | 改用 `tempfile.mkstemp(suffix=file_ext)` 写临时文件，`finally` 块确保清理 |
| `/workspaces` 不显示是否有解析结果 | 新增 `has_files` 字段，通过 `rglob` 检测是否存在 `hybrid_auto/*.md` |

### `DEFAULT_UPLOAD_DIR` 删除

`app.py` 改用 `tempfile.mkstemp()` 后不再需要持久化上传目录。`DEFAULT_UPLOAD_DIR` 从 `app.py` 和 `constants.py` 中一并删除，`UPLOAD_DIR` 全局变量及 `mkdir()` 调用同步移除。
**注**：7b483f3 commit message 描述了此删除，但 constants.py 的实际改动未包含在该提交内，已在后续增量记录中补正。

### 新增辅助函数

`_compute_doc_id(name: str) -> str`：与 `local_rag.py` 中 `_safe_doc_id()` 逻辑一致，将文件名转为合法 doc_id，全为非法字符时 fallback MD5 hex。

---

## 增量更新（2026-02-25，multimodal_top_k 语义重设计 + is_multimodal 传播修复）

### 背景

调试发现 T1 初始实现存在三个 bug，导致 `multimodal_top_k` 功能完全失效：

1. **`is_multimodal` 字段在检索时被丢弃**（`operate.py`）：round-robin merge 重建 chunk dict 时只保留 `content`、`file_path`、`chunk_id` 三个字段，`is_multimodal` 被截断，`process_chunks_unified` 始终看到 `is_multimodal=False`
2. **`_process_image_paths_for_vlm` 无图片上限**（`query.py`）：即使 `process_chunks_unified` 正确过滤了 mm_chunks，该函数仍会扫描 raw_prompt 全文中所有 `Image Path:` 模式并全部发送图片，绕过了 multimodal_top_k 的约束
3. **T1 初始逻辑语义错误**（`utils.py`）：将超出 multimodal_top_k 的多模态 chunk 整体丢弃（总 context 从 10 缩减到 4），而不是保留为纯文本降级

### `is_multimodal` 传播修复（`lightrag/lightrag/operate.py`）

**4 处 dict 重建位置均补充字段**：

- `_get_vector_context()`（L3496）：`chunk_with_metadata` 字典新增 `"is_multimodal": result.get("is_multimodal", False)`
- `_merge_all_chunks()` round-robin（L3911、L3926、L3941，三处分支）：每处 `merged_chunks.append({...})` 新增 `"is_multimodal": chunk.get("is_multimodal", False)`

### `multimodal_top_k` 语义重设计（`lightrag/lightrag/utils.py`）

**新语义**：`multimodal_top_k` 仅控制发给 VLM 的图片数量，不限制 context chunk 数量。

**`process_chunks_unified()` 三处改动**：

**Step 1**（rerank top_n）：
```python
if getattr(query_param, "multimodal_top_k", None) is not None:
    rerank_top_k = len(unique_chunks)   # 保留全量，确保文本 chunk 有足够候选
else:
    rerank_top_k = query_param.chunk_top_k or len(unique_chunks)
```

**Step 2.5**（预算分配，完全重写）：

核心语义：先切出 top `chunk_top_k` 的 window（不分类型），window 里有几个 mm 就选几个；文本 budget 先从 window 内取，不足则从 window 外按 rerank 顺序补充。

```python
top_window     = unique_chunks[:chunk_top_k]      # 先切 top-10（类型无关）
remaining_pool = unique_chunks[chunk_top_k:]       # 10名以外备用

selected_mm    = [c for c in top_window if c.get("is_multimodal")]
text_in_window = [c for c in top_window if not c.get("is_multimodal")]

text_budget    = max(chunk_top_k - multimodal_top_k, 0)   # = 7
selected_text  = text_in_window[:text_budget]
# 若 window 里文本不足，从 remaining_pool 中补充文本 chunk
if len(selected_text) < text_budget:
    extra = [c for c in remaining_pool if not c.get("is_multimodal")]
    selected_text += extra[:text_budget - len(selected_text)]

unique_chunks = sorted(selected_mm + selected_text,
                       key=lambda c: c.get("rerank_score", 0), reverse=True)
budgets_applied = True
# 日志示例：Context composition: 10 multimodal (3 with images, 7 text-only) + 7/7 text chunks
```

**Step 3**（chunk_top_k 截断，条件跳过）：
```python
if not budgets_applied:   # multimodal_top_k 激活时跳过
    if query_param.chunk_top_k is not None and query_param.chunk_top_k > 0:
        ...
```

### 图片上限硬封顶（`rag-anything/raganything/query.py`）

`_process_image_paths_for_vlm(prompt, max_images=None)` 新增 `max_images` 参数：

```python
def replace_image_path(match):
    nonlocal images_processed
    if max_images is not None and images_processed >= max_images:
        return match.group(0)   # 达到上限：保留路径文字，不附图
    ...
```

`aquery_vlm_enhanced()` 传递 `image_cap = kwargs["multimodal_top_k"]` 给该函数。

### 最终 context 组成（chunk_top_k=10, multimodal_top_k=3）

设 top-10 window 含 9 mm + 1 text，remaining pool 含更多 text：

| 部分 | 数量 | 来源 |
|------|------|------|
| 多模态 chunk（附图） | min(9, 3) = 3 | top-10 window 内的 mm，rerank 最高的 3 个 |
| 多模态 chunk（纯文本降级） | 9 - 3 = 6 | top-10 window 内的 mm，rank 4-9 |
| 文本 chunk | 7 | window 内 1 个 + remaining pool 补 6 个 |
| **总计** | **16** | |

设 top-10 全为 mm（如实测 80 chunks 中 72 mm 的场景）：

| 部分 | 数量 | 来源 |
|------|------|------|
| 多模态 chunk（附图） | 3 | top-10 window mm，rank 1-3 |
| 多模态 chunk（纯文本降级） | 7 | top-10 window mm，rank 4-10 |
| 文本 chunk | 7 | 全部来自 remaining pool |
| **总计** | **17** | |

### `constants.py` 补正

- 路径前缀更新：`./output` → `./rag-anything/output`（工作区根目录运行时正确解析）
- `DEFAULT_UPLOAD_DIR` 真正删除（7b483f3 的 commit message 声称删除，但实际未改文件）

### 文件移动

`raganything/raganything_local.py` → `raganything/examples/raganything_local.py`（示例脚本移至 examples 目录）

### 改动文件汇总

| 文件 | 改动 |
|------|------|
| `lightrag/lightrag/utils.py` | Step 1/2.5/3 重写（含 top-window 语义修正，见下方补丁记录）|
| `lightrag/lightrag/operate.py` | `is_multimodal` 传播（+4 行） |
| `rag-anything/raganything/query.py` | max_images 封顶、top_k defaults（+39 -12 行） |
| `rag-anything/raganything/constants.py` | 路径更新、DEFAULT_UPLOAD_DIR 删除 |
| `rag-anything/raganything/raganything_local.py` | 删除（移至 examples） |
| `rag-anything/examples/raganything_local.py` | 新增（从 raganything/ 移入） |
| `.vscode/launch.json` | zhb debug 配置路径修正 |

---

## 补丁（0039c63，multimodal budget split top-window 修正）

### 问题

上一版 Step 2.5 从整个 rerank pool 取所有 mm_chunks（实测 80 chunks 中 72 个均为多模态，全部进入 context），而非从 top `chunk_top_k` 的 window 内取。导致日志出现 `72 multimodal (3 with images, 69 text-only)`，context 共 77 chunks。

### 修正（`lightrag/lightrag/utils.py`）

将"先分 mm/text 再分别取"改为"先切 top-window 再分"：

- `selected_mm = top_window 内的 mm chunks`（上限自然为 chunk_top_k）
- `selected_text`：先取 window 内文本，不足 text_budget 时从 window 外的 remaining_pool 按 rerank 顺序补文本 chunk

修正后实测（80 chunks，72mm+8text）：`Context composition: 10 multimodal (3 with images, 7 text-only) + 7/7 text chunks`

## 执行记录补充（2026-02-26，完成 2026-02-25 待行动项）

### 本轮目标（已完成）
- enhanced 与 non-enhanced 统一使用同一套 `system/user` 重组规则。
- 尾句规则统一为单一 helper 控制：
  - 有图：`Please answer based on the context and images provided.`
  - 无图：`Please answer based on the context provided.`
- enhanced 无图回退时与 non-enhanced 保持一致（不再保留额外控图参数影响检索预算）。

### 代码落地
- 新增 `raganything/query_message_repack.py`：
  - 抽取公共 helper：`repack_query_messages(...)`、`build_answer_suffix(...)`。
  - 统一清理逻辑：仅取最后一个 `---Context---`，清理 `---User Query---`、`User Question/User Query` 行、历史尾句模板、空 `Additional Instructions`。
- `raganything/services/local_rag.py`：
  - 删除本地重复的文本重组函数，改为复用 `query_message_repack.py`。
  - non-enhanced 文本问答走统一 helper，减少重复代码和分叉行为。
- `raganything/query.py`：
  - enhanced 消息构建改为复用同一 helper。
  - 保持“有图时图片在 user content 最前”。
  - 修复无图回退一致性：fallback 到文本 query 时移除 `multimodal_top_k`，保证与 non-enhanced 检索预算一致。
- `evaluate_local/DocBench/evaluate.py`：
  - 删除失效字段 `settings.vlm_max_images`。
  - 改为在 `DOCBENCH_QUERY_PARAMS` 中使用 `multimodal_top_k` 控图（当前值 `5`）。

### 与未改动前相比的功能变化
- 由“两套消息组装逻辑”变为“一套共享消息组装逻辑”。
- enhanced（无图）与 non-enhanced 的消息文本与尾句保持一致。
- 控图参数从旧的本地设置项迁移为 query 参数语义（`multimodal_top_k`）。
- 减少重复清理与重复拼接代码，调用链更短、更易排障。

### 本轮校验
- 语法检查通过：
  - `raganything/query_message_repack.py`
  - `raganything/query.py`
  - `raganything/services/local_rag.py`
  - `evaluate_local/DocBench/evaluate.py`
- 行为校验（轻量）：
  - `query_message_repack.py` 的重组与尾句选择已用样例断言验证。
  - `query.py` 已静态确认 fallback 分支会去除 `multimodal_top_k`。

## 三文件对照复核（2026-02-26，按“改前 vs 改后”）

### 1) `raganything/raganything/services/local_rag.py`
- 改前：
  - non-enhanced 文本重组逻辑内嵌在本文件，多段 helper 与 enhanced 路径存在重复能力。
- 改后：
  - 删除本地重复 helper，统一复用 `query_message_repack.repack_query_messages(...)`。
  - non-enhanced 问答的 system/user 重组、尾句规则与 enhanced 共用同一规则源。
- 结果：
  - 代码更短，重复逻辑减少，行为一致性更高。

### 2) `raganything/raganything/query.py`
- 改前：
  - enhanced 的消息拼装与 non-enhanced 规则不一致，且无图回退会保留 `multimodal_top_k`。
- 改后：
  - `_build_vlm_messages_with_images(...)` 改为复用 `repack_query_messages(...)`，统一尾句规则。
  - 保留增强模式“图片在前、文本在后”。
  - 无图回退时显式移除 `multimodal_top_k`，避免影响文本检索预算。
- 结果：
  - enhanced / non-enhanced 在“无图场景”对齐，逻辑线条更清晰。

### 3) `rag-anything/evaluate_local/DocBench/evaluate.py`
- 改前：
  - 使用失效字段 `settings.vlm_max_images`。
  - `dump_raw_prompt` 的 QueryParam 与真实 query 参数集不完全一致。
- 改后：
  - 改为通过 `DOCBENCH_QUERY_PARAMS["multimodal_top_k"]` 控图。
  - `dump_raw_prompt` 增加 `multimodal_top_k/max_total_tokens/max_entity_tokens/max_relation_tokens`，与真实 query 参数对齐。
- 结果：
  - 调试观测结果与实际执行链路一致，不会“看错 prompt”。

### 本轮逻辑校验结论
- 不是只做编译检查；已对关键行为做链路级复核：
  - 尾句二选一与清理规则统一。
  - enhanced 无图回退参数清理正确。
  - evaluate 的 raw prompt 抓取参数与真实 query 对齐。
- 结论：三文件当前改动保持“最小必要改动 + 行为一致 + 代码简化”。

## 增量补充（2026-02-26，按当前决策去除旧 token 兼容 + 统一 optional system_prompt 空值语义）

### 1) 去除旧 token 环境变量兼容链（`local_rag.py`）
- 变更前：`query_max_tokens` 读取链为
  `RAGANYTHING_QUERY_MAX_TOKENS -> RAGANYTHING_VISION_MAX_TOKENS -> RAGANYTHING_MAX_TOKENS`。
- 变更后：仅保留
  `RAGANYTHING_QUERY_MAX_TOKENS`（默认 `DEFAULT_QUERY_MAX_TOKENS`）。
- 同时删除 legacy token 字段告警分支（`max_tokens/vision_max_tokens`）。

### 2) enhanced optional system_prompt 与 Additional Instructions 空值语义对齐
- 新增统一清理 helper：`normalize_optional_instruction_text(...)`。
- 对 `{user_prompt}` / `none` / `null` / `n/a` / 空串按“无有效附加指令”处理。
- enhanced 链路中，`optional system_prompt` 为空占位时不再拼接到最终 system。

### 3) 当前建议与影响
- 若环境里仍在设置 `RAGANYTHING_VISION_MAX_TOKENS` 或 `RAGANYTHING_MAX_TOKENS`，需要迁移为 `RAGANYTHING_QUERY_MAX_TOKENS`。
- `raganything_local_v2.py` 与当前评测脚本均已使用新字段，不受影响。

## 最终检查补充（2026-02-26，代码精简与逻辑复核）

### 本轮对照“改前 vs 改后”结论
- `local_rag.py`：
  - 改前：文本重组 helper 分散在本文件，存在重复逻辑。
  - 改后：统一复用 `query_message_repack.py`，删除重复 helper，保留单一路径。
- `query.py`：
  - 改前：enhanced 消息构建与 non-enhanced 规则存在分叉；无图回退会保留 `multimodal_top_k`。
  - 改后：复用同一重组规则；无图回退显式移除 `multimodal_top_k`，与 non-enhanced 检索预算一致。
- `evaluate.py`：
  - 改前：使用无效字段 `settings.vlm_max_images`，raw prompt 参数与真实 query 不完全一致。
  - 改后：统一使用 `DOCBENCH_QUERY_PARAMS["multimodal_top_k"]` 控图；`raw_prompt` 参数补齐 `multimodal_top_k/max_total_tokens/max_entity_tokens/max_relation_tokens`。
- 环境变量读取：
  - 改前：`query_max_tokens` 兼容链过长（含旧字段）。
  - 改后：仅保留 `RAGANYTHING_QUERY_MAX_TOKENS`，去除旧兼容分支。

### 本轮代码简洁性检查
- 删除冗余函数与重复清理逻辑，避免双实现。
- 保持 enhanced 仅保留必要差异（有图时图片前置）；无图场景与 non-enhanced 对齐。
- 未新增临时测试脚本、未保留调试专用改动到业务目录。

### 本轮验证记录
- 语法检查通过：
  - `python -m py_compile rag-anything/raganything/query_message_repack.py`
  - `python -m py_compile rag-anything/raganything/query.py`
  - `python -m py_compile rag-anything/raganything/services/local_rag.py`
  - `python -m py_compile rag-anything/evaluate_local/DocBench/evaluate.py`
- 逻辑检查通过（轻量断言）：
  - `repack_query_messages(...)`：空 `Additional Instructions` 清理、尾句选择、`---User Query---` 清理符合预期。
  - `query.py`：已确认 fallback 分支存在 `fallback_kwargs.pop("multimodal_top_k", None)`。

## 增量更新（2026-02-27，Rerank-First + Image-Cap-Only）

- chunk 选择改为：rerank -> chunk_top_k -> token 截断。
- multimodal_top_k 仅用于图片转换阶段的图片上限，不再参与 chunk 分池。
- 无图回退时移除 multimodal_top_k，回退行为与 non-enhanced 一致。
- 已做逻辑测试：
  - 反例验证通过（高分文本不再被提前排除）。
  - 候选数量不再超过 chunk_top_k。
  - 图片上限按 prompt 顺序生效。

##  增量更新（2026-02-27，抽取与查询 token 上限分离）

- 目标：将“索引实体抽取”上限提升到 8192，同时保持“query 回答”上限 2048。
- 原状态（本地封装）：
  - `llm_model_func` 固定使用 `query_max_tokens`，导致实体抽取也被限制在 2048。
- 本次改动：
  - 在 `raganything/services/local_rag.py` 增加 `_is_entity_extraction_call(system_prompt, prompt)` 识别 LightRAG 的 extract/glean 抽取请求。
  - `llm_model_func` 的 `max_tokens` 选择逻辑改为：
    - 若显式传入 `max_tokens`：优先使用传入值；
    - 否则若判定为实体抽取：使用 `ingest_max_tokens`（当前默认 8192）；
    - 其他情况：使用 `query_max_tokens`（当前默认 2048）。
  - 同时先从 `cleaned_kwargs` 弹出 `max_tokens` 再统一注入，避免重复关键字冲突。
- 行为结果：
  - 索引实体抽取链路（含 continue/glean）默认按 8192 上限；
  - query 文本/多模态回答链路保持 2048；
  - 入库多模态分析链路继续使用 `ingest_max_tokens`（8192），不变。
- 校验：
  - `python -m py_compile raganything/services/local_rag.py` 通过。

## 待改动需求（2026-02-27，Entity 去重与 entity_key 评估）

### 需求背景与问题

- 当前图谱主键仍以 `entity_name` 直接驱动（节点聚合、关系端点、向量索引均使用实体名）。
- 同义写法/大小写/标点差异会造成“同一实体拆成多个节点”，影响：
  - 图谱 merge 一致性（同实体被重复建点）；
  - query 检索命中率（相关关系被分散）；
  - 上下文拼装质量（实体信息割裂）。

### 风险判断（为什么不能直接改主键）

- 若直接把主键从 `entity_name` 切换到 `entity_key`，会影响全链路：
  - `merge_nodes_and_edges` 的节点聚合键；
  - 关系 `src_id/tgt_id` 的端点一致性；
  - graph/vdb 的 upsert/get 键；
  - query 侧实体/关系检索与最终展示映射。
- 属于高风险重构，若一次到位易引入断链或回归。

### 改动目标（先低风险）

- 目标不是立即替换主键，而是“先提升去重质量且不破坏现有链路”：
  - 保留 `entity_name` 作为当前主键与展示名；
  - 新增 `entity_key` 作为内部归一键（NFKC + casefold + 空白/标点归一）用于别名对齐与去重判断；
  - 在 merge 前做名称归并映射，减少重复实体落库。

### 计划改动方案（分阶段）

1. Phase-1（低风险，优先）
- 新增 `entity_key` 计算函数（仅用于归并判定，不替换主键）。
- 在 `merge_nodes_and_edges` 前增加轻量归并步骤：
  - 同 `entity_key` 的候选进行合并；
  - 关系端点同步按映射重写；
  - 最终仍以统一后的 `entity_name` 进入现有 upsert 流程。
- 日志增加归并统计（合并前后实体数、命中规则）。

2. Phase-2（可选，谨慎）
- 增加 alias 存储（可持久化）：
  - 自动确认高置信别名映射；
  - 低置信映射保留人工确认入口。
- 用 alias 提升跨文档增量入库的一致性。

3. Phase-3（暂不执行）
- 评估是否将 `entity_key` 升级为主键（高风险，需要全链路联调后再决策）。

### 验收标准（本需求）

- 不破坏当前 query/ingest 行为与接口；
- 图谱实体重复率下降（同义名拆点减少）；
- 关系端点不丢失、查询结果不回退；
- 通过语法检查 + 逻辑反例验证（别名、大小写、标点差异样本）。

##  增量更新（2026-02-27，Entity 抽取精度清洗 + 去除 soft_path）

### 改动目标

- 解决 index 阶段 precision 不足：路径碎片、版面/结构元数据、表图编号、通用词残片进入图谱。
- 移除 `soft_path` 规则，避免把 `foo/bar/baz` 这类非路径文本误判为路径。

### 代码改动

- 文件：`lightrag/lightrag/operate.py`
- 1) 路径判定收敛
  - 删除 `SOFT_PATH` 分支，仅保留硬规则：
    - Windows/UNC/Unix 绝对路径
    - 相对路径前缀 `./ ../ ~/`
    - 裸文件名+扩展名
  - 保留 `A/B`、`input/output` 等白名单短语不过滤。

- 2) 实体名轻量归一（不改主键机制）
  - 新增 `_normalize_entity_surface(...)`：
    - `_` 连接词转空格（`Game_Boy -> Game Boy`）
    - 词间连字符空格规范化（`side - scrolling -> side-scrolling`）
    - NFKC 归一 + 规则化大小写（无硬编码词典）：
      - 缩写样式（如 `kglm`）按规则归一为 `KGLM`
      - 普通小写短语按词形归一（避免死板映射表）

- 3) 低质量实体过滤
  - 新增 `_classify_low_quality_entity(...)`，过滤类别：
    - 路径碎片：基于当前 chunk `content` 内可识别路径动态提取并过滤（关注“碎片命中”，非完整路径字符串匹配）
    - 版面结构元数据：`Page Number/Footer Content/Bounding Box/...`
    - 表图编号（仅编号，不过滤超长标题）：
      - `Table 1`
      - `Figure 2`
    - 通用词/残片：`a/et/None/xt/...`
    - `published by` 类发布语残留
  - 实体过滤记录 reason 到日志。

- 4) 关系端点同步过滤
  - 在关系抽取中，对 `src/tgt` 同步执行归一与低质量判定；
  - 任一端点命中低质量规则则丢弃该关系并记录 reason。

### 逻辑校验

- 语法：`python -m py_compile lightrag/lightrag/operate.py` 通过。
- 内联样例检查通过（未新增测试文件）：
  - `projects/evaluate_local/docbench_results/y50056788` 这类路径碎片被按 chunk `content` 中路径动态命中并过滤；
  - `images/mineru outputs/hybrid auto` 作为路径碎片同样可命中过滤（不再放行）；
  - `Page Number` / `Table 1` / `a` 被正确过滤；
  - `kglm -> KGLM`，`Game_Boy -> Game Boy` 归一正确；
  - `foo/bar/baz` 在移除 soft_path 后不再按路径过滤。

### 说明

- 你确认“超长标题正常”，因此不再对 `Table 1: ...` 这类 caption 标题做实体层硬过滤。

## 增量更新（2026-02-27，路径碎片过滤与 chunk content 对齐）

### 背景

- 发现真实链路中，实体/关系抽取来源于 chunk `content`，而不是 `file_path` 文本本身。
- 仅基于 `file_path` 的碎片过滤会漏掉 `content` 中出现的路径碎片（如 `docbench_results`、`evaluate_local`）。

### 本次修正

- 文件：`lightrag/lightrag/operate.py`
- 路径碎片键计算改为：仅从 chunk `content` 中的可识别路径提取，不再混入 `file_path`。
- 识别来源包括：
  - `Image Path: ...` 行；
  - 文本中的 Windows/Unix/相对路径片段。
- 抽取结果处理阶段对实体与关系端点复用同一组 `path_fragment_keys`，避免重复计算。

### 上游抽取约束补充

- 文件：`lightrag/lightrag/prompt.py`
- 在 `entity_extraction_system_prompt` 中新增：
  - 实体排除规则：不抽取路径/路径碎片/文件名/纯版面元数据；
  - 关系排除规则：任一端点违规则不输出关系。
  - 将关系规则中的 `above` 改为显式锚点：`Entity Exclusion Rules in Section 1 (Entity Extraction & Output)`，降低中小模型理解歧义。

### 校验

- 语法：
  - `python -m py_compile lightrag/lightrag/operate.py`
  - `python -m py_compile lightrag/lightrag/prompt.py`
- 逻辑（内联，不落地测试文件）：
  - 当 `file_path` 仅为 basename（如 `P19-1598.pdf`）时，仍可从 chunk `content` 的路径行提取 `evaluate_local/docbench_results/mineru_outputs/hybrid_auto/images` 等碎片并命中过滤；
  - 不提供 `content` 路径时，上述碎片不会被误判；
  - `input/output` 仍由白名单放行，`/images` 仍按路径命中，数学表达式白名单不受影响。

## 增量更新（2026-02-27，接入 history/image token 预算 + enhanced 历史对齐）

### 改动目标

- 将 `history_tokens` / `image_tokens` 真正纳入 `available_chunk_tokens` 预算，不改变 rerank 与 chunk 选取逻辑顺序。
- 为后续“summary history”预留正式接口，避免只能传整段原始多轮对话。
- 让 enhanced 与 non-enhanced 在 history 行为上保持一致（都可带历史上下文）。

### 代码改动

- 文件：`lightrag/lightrag/base.py`
  - `QueryParam` 新增：
    - `history_summary`：历史摘要文本（可选）。
    - `image_token_estimate_per_image`：读图失败时的单图 token 兜底值。
    - 动态估算参数：
      - `image_size_for_token_estimate`
      - `patch_size_for_token_estimate`
      - `downsample_ratio_for_token_estimate`
      - `min_dynamic_patch_for_token_estimate`
      - `max_dynamic_patch_for_token_estimate`
      - `use_thumbnail_for_token_estimate`
      - `image_wrapper_tokens_per_image`

- 文件：`lightrag/lightrag/constants.py`
  - 新增 `DEFAULT_IMAGE_TOKEN_ESTIMATE_PER_IMAGE = 1024`（读不到图片尺寸时使用）。

- 文件：`lightrag/lightrag/operate.py`
  - 新增统一历史构造：
    - `_build_effective_history_messages(query_param)`：
      - `history_summary`（若有）先转为一条 summary 历史消息；
      - 再拼接清洗后的 `conversation_history`。
  - 新增 token 估算：
    - `_estimate_history_tokens(...)`
    - `_estimate_image_tokens_for_budget(...)`
      - 按 `Image Path:` 从 chunk 内容提取候选图；
      - 按 `multimodal_top_k` 截断；
      - 可读图时按动态 patch 估算：
        - `num_image_token = (image_size // patch_size)^2 * downsample_ratio^2`
        - `image_tokens_i = num_patches_i * num_image_token + wrapper_tokens`
        - `use_thumbnail=true` 且 `blocks != 1` 时加 thumbnail patch
      - 读图失败时回退 `image_token_estimate_per_image`。
  - 两条查询链预算均已接入：
    - KG 查询预算：
      - `available_chunk_tokens = max_total_tokens - (sys + kg + query + history + image + buffer)`
    - naive 查询预算：
      - `available_chunk_tokens = max_total_tokens - (sys + query + history + image + buffer)`
  - 缓存一致性补充：
    - query cache 的 `args_hash` / `queryparam` 增加 `history_signature`，避免不同历史命中同一缓存。
  - LLM 调用统一改为发送 `effective_history_messages`（不仅 raw conversation）。

- 文件：`lightrag/lightrag/api/routers/query_routes.py`
  - `QueryRequest` 新增 `history_summary` 入参（API 接口可直接传）。

- 文件：`raganything/raganything/query.py`
  - enhanced 路径读取并传递 `conversation_history/history_summary`。
  - `_build_vlm_messages_with_images(...)` 支持注入历史消息：
    - system 在前；
    - history（summary + 对话）中间；
    - 当前 user 在后（含图片时仍是 `image_url(base64)+text` 结构）。
  - 不改变图像推理输入格式：仍是 `data:image/jpeg;base64,...`。

### 关键行为说明

- rerank、chunk_top_k、token 截断顺序保持不变，只是把 history/image 预算提前扣减。
- 目前若未传 `conversation_history/history_summary`，这两项 token 默认是 0（你当前“没开 history”场景不受影响）。
- enhanced 与 non-enhanced 都支持历史：
  - non-enhanced：走 LightRAG `history_messages`；
  - enhanced：最终多模态 `messages` 也会带同一份历史。

### 校验

- 语法通过：
  - `lightrag/lightrag/constants.py`
  - `lightrag/lightrag/base.py`
  - `lightrag/lightrag/operate.py`
  - `lightrag/lightrag/api/routers/query_routes.py`
  - `raganything/raganything/query.py`
- 逻辑校验：
  - 已完成静态链路检查（预算项、history 注入点、enhanced/non-enhanced 对齐点）。
  - 本机当前 shell 缺少运行依赖（如 `python-dotenv`），未完成端到端运行态回归；后续建议在项目 venv 下跑你现有 `evaluate.py` 的 0-1 样本进行最终行为确认。

## 增量更新（2026-02-27，改动后复核结论）

### 复核范围

- `lightrag/lightrag/constants.py`
- `lightrag/lightrag/base.py`
- `lightrag/lightrag/api/routers/query_routes.py`
- `lightrag/lightrag/operate.py`
- `rag-anything/raganything/query.py`

### 复核结果（相较未改动前）

- 已接入预算项：
  - `available_chunk_tokens` 现已扣减 `history_tokens` 与 `image_tokens`。
  - KG/naive 两条查询链都已接入，且为同一预算思路。
- History 接口已对齐：
  - 新增 `history_summary`，支持“摘要 + 原始 history”混合输入。
  - enhanced 与 non-enhanced 都可带历史上下文。
- 图片消息格式保持不变：
  - 仍使用 `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`。
  - 本次只新增“图片 token 预算估算”，不改推理消息协议。
- 缓存一致性增强：
  - query cache 增加 `history_signature`，避免不同 history 误命中同一缓存。

### 简洁性与冗余说明

- 本轮没有新增独立测试文件，也未引入临时运行文件。
- 变更集中在预算与 history 接口接入，未改动 rerank 主逻辑顺序。

## 增量更新（2026-02-27，再次复检）

### 复检结论

- 本轮改动文件再次完成语法检查（`py_compile`）并通过。
- 预算链路确认：
  - KG/naive 的 `available_chunk_tokens` 都已扣减 `history_tokens + image_tokens`。
  - 未开启 history 时 `history_tokens = 0`；无图场景 `image_tokens = 0`。
- 协议确认：
  - enhanced 发送图片仍为 base64 `image_url`，未改消息协议。
- 代码简洁性：
  - 本轮未新增冗余分支与临时测试文件，保持最小增量修改。

### 未覆盖项说明

- 当前环境仅完成静态与编译级复检，未在项目虚拟环境做端到端跑数。
- 建议你在现有环境执行一次 0-1 样本（enhanced/non-enhanced 各一轮）做最终运行态确认。

## 增量更新（2026-02-27，运行态测试执行结果）

### 已执行并通过（本机可复现）

- 边界测试（离线）：
  - `multimodal_top_k=0` 时：`image_tokens=0`。
  - `multimodal_top_k=1` 时：图片计数严格封顶为 1。
  - `history` 为空时：`history_tokens=0`。
  - 仅传 `history_summary`：可正常进入历史消息。
- 反例测试（离线）：
  - 无效图片路径：触发 `image_token_estimate_per_image` fallback。
  - 超长 `history_summary`（5 万字符）：token 估算稳定，无异常。
  - enhanced 无图回退：最终 user content 为纯文本。
  - enhanced 有图：最终 user content 为 `image_url(base64)` 在前、文本在后。

### 端到端测试阻断（本机环境）

- `evaluate.py` 运行链路需要本机完整依赖与服务：
  - 初始阻断：`sentence_transformers` 缺失（已通过 stub 方式仅验证 argparse/help 可起）。
  - 关键阻断：本机 `localhost:8001/8002/8000` 均未启动（连接被拒绝），无法完成真实 E2E 推理回合。
  - 数据阻断：`evaluate_local/DocBench` 下未见完整 DocBench 数据目录，无法直接跑 0-1 文档生成。

### 本次补充工具

- 新增图片 token 对齐脚本（仅测 image token）：
  - `rag-anything/examples/image_token_alignment_test.py`
  - 用途：对比“本地估算”与“vLLM prompt_tokens 差分”。

## 增量更新（2026-02-27，移除调用层 max_tokens 覆盖）

### 背景

- 当前链路已明确分离：
  - query 走 `query_max_tokens`
  - ingest/抽取走 `ingest_max_tokens`
- 用户要求移除调用层显式 `max_tokens` 覆盖能力，避免冗余与行为不透明。

### 代码调整

- 文件：`rag-anything/raganything/services/local_rag.py`
  - `build_llm_model_func(...)` 中删除 `requested_max_tokens` 分支。
  - 新增 `max_tokens` 到 `_INTERNAL_OPENAI_KWARGS`，统一丢弃调用层传入的 `max_tokens`。
  - 保持默认策略：
    - 实体抽取调用：`ingest_max_tokens`
    - 其余问答调用：`query_max_tokens`

### 校验

- `python -m py_compile raganything/services/local_rag.py` 通过。
- 代码检索确认已无调用层 `max_tokens` 覆盖入口。
