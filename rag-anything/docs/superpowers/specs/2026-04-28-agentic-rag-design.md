# 自适应 Agentic RAG 设计文档

**日期**: 2026-04-28  
**状态**: 待实现  
**作者**: Jobfromearth

---

## 1. 背景与目标

### 现状

`mode="auto"` 是当前系统的智能路由入口，流程为：

```
QueryClassifier（LLM 分类一次）→ profile 选择 → 并行检索路径 → RRF 合并 → Rerank → 单次生成
```

这是"单程票"设计：分类一次、检索一次、生成一次，无迭代能力。

### 目标

新增 `mode="agentic"`，构建自适应智能体工作流：

- **简单查询**：快速直达，不引入额外延迟
- **中等查询**：生成后自省评估，不足则补充检索一次
- **复杂查询**：问题分解 → 并行检索 → 迭代评估，最多 3 轮

### 设计约束

- 现有代码**零修改**：通过新增 `mode="agentic"` 分支集成
- 本地 vLLM 部署，无调用成本约束
- 可观测性：Arize Phoenix 本地部署，B 级别（实时追踪，无需测试集）
- 离线评估脚本（`evaluate_multihop.py`、`evaluate_surge_fast.py`）保持独立，不纳入本设计

---

## 2. 架构总览

### 新增文件（现有文件不修改）

```
rag-anything/raganything/
  observability.py                      ← Phoenix OTEL 初始化
  retrieval/
    complexity.py                       ← 复杂度分类器
    evaluator.py                        ← Evaluator-Optimizer 节点
    agent_graph.py                      ← LangGraph 图定义（主体）
```

### 集成点

```python
# raganything/query.py — 新增一个 elif 分支，其余不变
if mode == "agentic":
    from .retrieval.agent_graph import AdaptiveAgentGraph
    graph = AdaptiveAgentGraph(self.lightrag)
    return await graph.run(query, **kwargs)
```

---

## 3. 工作流模式映射

五种 Anthropic 工作流模式全部覆盖，各有明确落地位置：

| 模式 | 节点 | 说明 |
|------|------|------|
| **路由 (Routing)** | `classify_complexity` | 输出 simple/medium/complex，决定走哪条轨道 |
| **并行化 (Parallelization)** | `parallel_retrieve` | complex 路径中各子问题并发调用 `RetrievalRouter` |
| **提示链 (Prompt Chaining)** | complex 完整路径 | decompose → retrieve → generate → evaluate，逐步传递 |
| **Orchestrator-Workers** | `decompose`（orchestrator）+ `run_path`（workers） | Orchestrator 生成 2-4 个子问题，复用现有原子检索单元 |
| **Evaluator-Optimizer** | `evaluate` + conditional edge | LLM 评分 0-1，< 0.7 时输出 `eval_gap` 驱动补充检索 |

---

## 4. 图结构

### 三条执行轨道

```
simple:
  classify → retrieve → generate → END
  约 1-2 次 LLM 调用

medium:
  classify → retrieve → generate → evaluate
                                       ├─ score ≥ 0.7 → END
                                       └─ retry (≤1 次) → targeted_retrieve → generate → END
  约 2-4 次 LLM 调用

complex:
  classify → decompose → parallel_retrieve → generate → evaluate
                               ↑                            ├─ score ≥ 0.7 → END
                               └──── targeted_retrieve ←───┘ retry (≤2 次)
  约 4-7 次 LLM 调用
```

### 路由规则

| 条件 | 跳转 |
|------|------|
| `complexity == "simple"` | classify → retrieve |
| `complexity == "medium"` | classify → retrieve（同路径，evaluate 节点激活） |
| `complexity == "complex"` | classify → decompose |
| `eval_score >= 0.7 OR iteration >= MAX_ITER` | evaluate → END |
| `eval_score < 0.7 AND iteration < MAX_ITER` | evaluate → targeted_retrieve |

`MAX_ITER`：medium = 1，complex = 2。

---

## 5. 状态定义

```python
from typing import Annotated
import operator
from typing_extensions import TypedDict

class AgentState(TypedDict):
    query: str                                              # 原始问题，不可变，作为 generate/evaluate 主轴
    complexity: str                                         # "simple" | "medium" | "complex"
    sub_questions: list[str]                                # complex 路径首次分解结果（不在 retry 时重新生成）
    retrieved_chunks: Annotated[list[dict], operator.add]  # Reducer：追加模式，跨轮累积，不覆盖
    answer: str
    eval_score: float                                       # Evaluator 评分 0-1
    eval_gap: str                                           # "答案缺少X方面" — 驱动 targeted_retrieve
    current_search_query: str                               # 当前检索词；初始 = query，retry 时 = query + eval_gap
    iteration: int                                          # 当前迭代次数，硬上限保护
    routing_trace: dict                                     # 保留现有 trace 结构，Phoenix 消费
```

**关键设计决策**：

- `retrieved_chunks` 使用 `operator.add` Reducer：retry 时新检索到的 chunk **追加**而非覆盖，generate 节点始终看到所有轮次的证据
- generate 节点在使用前按 `chunk_id` 去重，保留 `rrf_score` 最高的版本
- `query` 字段只传给 generate 和 evaluate；`current_search_query` 只传给检索节点

---

## 6. 核心节点逻辑

### 6.1 classify_complexity

扩展现有 `QueryClassifier`，输出三级而非 profile 名：

```
simple:   单跳事实查询，一次检索可完整回答
medium:   单实体深度查询或轻度推理，可能需要一次补充检索
complex:  多实体链条、因果分析、跨文档比较，需要分解
```

分类置信度 < 0.6 时，**降级为 medium**（不升级为 complex，避免浪费）。

### 6.2 decompose（complex 路径专用）

Orchestrator 提示词要求：
- 将原始问题拆解为 2-4 个**可独立检索**的子问题
- 子问题之间不重叠
- 每个子问题应可由现有 `RetrievalRouter` 独立回答

输出：`sub_questions: list[str]`，写入 state。**retry 时不重新调用此节点。**

### 6.3 parallel_retrieve

```python
_SUB_SEM = asyncio.Semaphore(MAX_CONCURRENT_SUBQUESTIONS)  # 默认 3

async def parallel_retrieve(state: AgentState) -> dict:
    async def _one(sub_q: str):
        async with _SUB_SEM:
            chunks, trace = await router.route(sub_q, param)
            return chunks
    results = await asyncio.gather(*[_one(q) for q in state["sub_questions"]])
    # 展平后返回，由 Reducer 追加到 state["retrieved_chunks"]
    return {"retrieved_chunks": [c for batch in results for c in batch]}
```

外层信号量 `MAX_CONCURRENT_SUBQUESTIONS` 控制峰值并发上限，峰值 = `MAX_CONCURRENT_SUBQUESTIONS × profile.max_concurrent_paths`，防止连接池压力。

### 6.4 targeted_retrieve（retry 专用节点）

```python
async def targeted_retrieve(state: AgentState) -> dict:
    new_query = f"{state['query']} — 补充检索：{state['eval_gap']}"
    chunks, trace = await router.route(new_query, param)
    return {
        "retrieved_chunks": chunks,  # Reducer 自动追加
        "current_search_query": new_query,
        "iteration": state["iteration"] + 1,
    }
```

单路径、不分解，只针对 `eval_gap` 指出的缺口补充证据。

### 6.5 evaluate

```python
# 评分提示词核心
"""
原始问题：{query}
当前答案：{answer}

评估：答案是否完整回答了问题？给出 0-1 分，并说明缺失的关键信息（如果有）。
输出 JSON：{{"score": float, "gap": "缺失信息描述，若无则为空字符串"}}
"""
```

- `score >= 0.7`：触发 END 边
- `score < 0.7 AND iteration < MAX_ITER`：触发 targeted_retrieve 边，`eval_gap` 写入 state
- `iteration >= MAX_ITER`：强制 END，不再迭代

`MAX_ITER` 由 `complexity` 字段在 evaluate 节点内动态决定：

```python
MAX_ITER_BY_COMPLEXITY = {"simple": 0, "medium": 1, "complex": 2}
max_iter = MAX_ITER_BY_COMPLEXITY[state["complexity"]]
```

---

## 7. 可观测性

### Phoenix 本地部署

```python
# raganything/observability.py
from phoenix.otel import register

def setup_phoenix(project_name: str = "rag-agentic", port: int = 6006):
    register(
        project_name=project_name,
        auto_instrument=True,
        # LangGraph instrumentor 自动捕获所有节点
    )
    # 访问 http://localhost:6006
```

在 `RAGAnything.__init__` 或服务启动时调用一次 `setup_phoenix()`。

### 自动追踪的指标（无需测试集，任意实时查询）

| 指标 | 来源 |
|------|------|
| 每节点延迟（ms） | OTEL span duration |
| 每次 LLM 调用 token 数 | OTEL LLM attributes |
| 复杂度分布（simple/medium/complex） | `classify_complexity` 节点输出 |
| Profile 分布 | `routing_trace.profile` |
| 各路径 chunk rerank 分数分布 | `routing_trace.chunks_per_path` + rerank scores |
| Evaluator 评分分布 | `evaluate` 节点输出 `eval_score` |
| 迭代次数分布 | `iteration` 字段 |
| 条件边跳转路径 | LangGraph instrumentor 自动记录 |

### asyncio context 传播注意事项

`asyncio.gather(*coroutines)` 自动继承 `contextvars.Context`，OpenTelemetry trace context 正常传播，span 不会断裂。

若将来将协程改为 `asyncio.create_task()`，需手动传递 context：

```python
ctx = contextvars.copy_context()
task = asyncio.create_task(ctx.run(coro))
```

当前实现统一使用协程模式，不引入此风险。

---

## 8. 依赖

```
langgraph>=0.2
arize-phoenix[otel]>=4.0
opentelemetry-sdk
opentelemetry-exporter-otlp
```

离线评估脚本（`evaluate_multihop.py`、`evaluate_surge_fast.py`）不引入上述依赖，保持独立运行。

---

## 9. 不在本次范围内

- 离线评估结果推送到 Phoenix（保持独立脚本）
- CrewAI 多 Agent 协作（过度设计，当前场景不需要）
- LangSmith / W&B 集成（Phoenix 本地部署已满足需求）
- `mode="auto"` 的任何修改

---

## 10. 后续实现入口

实现计划将由 `writing-plans` skill 生成，覆盖：

1. `observability.py` — Phoenix 初始化
2. `complexity.py` — 复杂度分类器
3. `evaluator.py` — Evaluator 节点
4. `agent_graph.py` — LangGraph 图主体
5. `query.py` 集成入口（仅新增 elif 分支）
6. 单元测试与集成测试
